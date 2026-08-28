"""The post-aggregation facts invariant gate, shared by both entry paths.

Two callers reach this gate with the same contract and different capabilities:

* the document graph, at ``VALIDATE_FACTS`` -- many units, so the un-merge
  repair is meaningful;
* the single-unit path (``/process_unit`` and the CLI ``--use-unit-pipeline``
  batch), which does not run the graph and calls in after aggregation.

Un-merging re-aggregates *retained units against each other*, which has no
meaning for one unit, so the single-unit caller passes ``merge_repair=False``.
Everything else -- validation arguments, SHACL autofix, conformance summary and
the emitted metrics -- is identical, and used to be written out twice; a new
``validate_aggregated_facts`` argument added to one copy silently skipped the
other entry path.
"""

from __future__ import annotations

import logging

from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import FactsValidationFinding, FactsValidationFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool.agg.aggregate import AggregationResult
from ontocast.tool.facts_validation import (
    FactsValidationReport,
    ShaclRepairResult,
    ValidationPolicy,
    apply_shacl_repairs,
    collect_shacl_shapes,
    record_facts_gate_metrics,
    shacl_catalog_contradictions,
    summarize_conformance,
    validate_aggregated_facts,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: Finding kinds the un-merge repair can actually act on. Scoring the loop on
#: *all* errors let SHACL findings -- which un-merging cannot fix -- decide
#: whether a pass was an improvement.
MERGE_SIGNATURE_KINDS = frozenset(
    {
        FactsValidationFindingKind.FUNCTIONAL_VIOLATION,
        FactsValidationFindingKind.SUSPECT_MULTI_VALUE,
        FactsValidationFindingKind.DEGENERATE_COREFERENCE,
    }
)


def vetoes_from_findings(
    findings: list[FactsValidationFinding],
    clusters: dict[str, list[str]],
) -> set[frozenset[URIRef]]:
    """Full-cluster pair vetoes for merge-signature error findings.

    Both the finding's subject and its IRI-valued objects are candidate merge
    victims. DEGENERATE_COREFERENCE reports the *pointing* node as subject and
    the over-merged endpoint in ``values`` (``range1 hasLowerBound v1 ;
    hasUpperBound v1`` -- ``v1`` is the collapsed cluster, ``range1`` usually
    is not merged at all), so a subject-only lookup could never repair it.
    The same holds for the IRI-object branch of SUSPECT_MULTI_VALUE.

    Args:
        findings: Error findings from the validation report.
        clusters: Identity-merge clusters keyed by representative IRI.

    Returns:
        Pairs of IRIs that must not be merged on a re-aggregation pass.
    """
    vetoes: set[frozenset[URIRef]] = set()
    for finding in findings:
        if finding.kind not in MERGE_SIGNATURE_KINDS:
            continue
        candidates = [finding.subject, *finding.values]
        for candidate in candidates:
            members = clusters.get(candidate, [])
            if len(members) < 2:
                continue
            refs = [URIRef(member) for member in members]
            for index, left in enumerate(refs):
                for right in refs[index + 1 :]:
                    vetoes.add(frozenset((left, right)))
    return vetoes


def run_facts_gate(
    state: AgentState,
    ontology_graph: RDFGraph,
    tools: ToolBox,
    *,
    merge_repair: bool,
    document_metadata: dict | None = None,
) -> None:
    """Validate the aggregated facts and apply the two LLM-free repair stages.

    When ``merge_repair`` is set, merge-signature error findings whose subject
    resulted from an identity merge turn the offending cluster into pair vetoes
    and the retained facts units are re-aggregated, up to
    ``FACTS_MERGE_REPAIR_PASSES`` times. Then, on both paths, SHACL autofix
    (``FACTS_SHACL_AUTOFIX``) repairs constraint violations in code. Residual
    findings stay on the state as telemetry.

    Args:
        state: Document state carrying ``aggregated_facts`` and the facts units.
        ontology_graph: Ontology context the facts are validated against.
        tools: Dependency container; only the aggregator is used, and only when
            ``merge_repair`` is set.
        merge_repair: Whether to run the un-merge repair loop. False for the
            single-unit path, where re-aggregating one unit is a no-op.
        document_metadata: Provenance metadata for re-aggregation. Required
            when ``merge_repair`` is set.

    Raises:
        ValueError: If ``merge_repair`` is set without ``document_metadata``.
    """
    if merge_repair and document_metadata is None:
        raise ValueError("merge_repair=True requires document_metadata")

    facts_validation = tools.config.get_tool_config().facts_validation
    shapes_graph = collect_shacl_shapes(ontology_graph, facts_validation.shapes_dir)
    contradictions = shacl_catalog_contradictions(
        shapes_graph,
        ontology_graph,
        policy=ValidationPolicy(
            additional_standard_namespaces=tuple(
                facts_validation.additional_standard_namespaces
            ),
            quantity_fallback_vocabulary=facts_validation.quantity_fallback_vocabulary,
            code_predicates=tuple(facts_validation.code_predicates),
        ),
    )
    if contradictions:
        # Data cannot satisfy both sides: the shapes demand these properties
        # while the unit validator's mandatory findings order the renderer to
        # remove them. This is a catalog/configuration error, and it silently
        # destroys extracted data (observed live: qudt:numericValue on the
        # matsci catalog). Loud, per document, until the catalog declares the
        # term or the deployment exempts its namespace.
        logger.error(
            "SHACL shapes require %d propert%s the term validator would flag "
            "as unknown: %s. Declare %s in the catalog, or exempt via "
            "FACTS_ADDITIONAL_STANDARD_NAMESPACES / the quantity fallback "
            "vocabulary.",
            len(contradictions),
            "y" if len(contradictions) == 1 else "ies",
            ", ".join(f"<{term}>" for term in contradictions),
            "it" if len(contradictions) == 1 else "them",
        )
    fact_namespaces = [DEFAULT_IRI, str(state.doc_iri), state.doc_namespace or ""]

    def run_validation() -> FactsValidationReport:
        return validate_aggregated_facts(
            state.aggregated_facts,
            ontology_graph,
            shapes_graph=shapes_graph,
            fact_namespaces=fact_namespaces,
            suspect_multi_value_severity=facts_validation.suspect_multi_value_severity,
            functional_min_single_support=(
                facts_validation.functional_min_single_support
            ),
            quantity_fallback_vocabulary=facts_validation.quantity_fallback_vocabulary,
            shacl_inference=facts_validation.shacl_inference,
            shacl_advanced=facts_validation.shacl_advanced,
            shacl_max_triples=facts_validation.shacl_max_triples,
        )

    def merge_signature_errors(report: FactsValidationReport) -> int:
        return sum(
            1
            for finding in report.error_findings
            if finding.kind in MERGE_SIGNATURE_KINDS
        )

    report = run_validation()
    vetoes: set[frozenset[URIRef]] = set()
    repair_passes = 0
    rejected_repairs = 0
    # Last *accepted* re-aggregation, i.e. the one whose graph is served.
    last_result: AggregationResult | None = None

    while merge_repair and repair_passes < facts_validation.merge_repair_passes:
        previous_errors = merge_signature_errors(report)
        if not previous_errors:
            break
        new_vetoes = vetoes_from_findings(
            report.error_findings, state.aggregation_clusters
        )
        if not (new_vetoes - vetoes):
            break
        vetoes |= new_vetoes
        logger.info(
            "Facts validation gate: %d merge-signature error finding(s), "
            "re-aggregating with %d merge veto pair(s)",
            previous_errors,
            len(vetoes),
        )
        result = tools.aggregator.postprocess_facts_units(
            units=state.facts_units,
            ontology_graph=ontology_graph,
            doc_iri=state.doc_iri,
            document_metadata=document_metadata,
            doc_namespace=state.doc_namespace,
            merge_vetoes=vetoes,
        )
        # Un-merging is destructive: a veto dissolves a whole cluster, so a
        # pass that does not strictly reduce the error count has traded real
        # coreference for nothing and must not be kept.
        previous_graph = state.aggregated_facts
        previous_clusters = state.aggregation_clusters
        state.aggregated_facts = result.graph
        state.aggregation_clusters = result.merged_clusters
        candidate_report = run_validation()
        candidate_errors = merge_signature_errors(candidate_report)
        if candidate_errors >= previous_errors:
            logger.warning(
                "Facts validation gate: repair pass %d did not reduce "
                "merge-signature errors (%d -> %d); reverting to the "
                "pre-repair graph",
                repair_passes + 1,
                previous_errors,
                candidate_errors,
            )
            state.aggregated_facts = previous_graph
            state.aggregation_clusters = previous_clusters
            rejected_repairs += 1
            break
        repair_passes += 1
        last_result = result
        report = candidate_report

    # Shape-driven repair, in code: retype literals against sh:datatype,
    # resolve codes to catalog IRIs, drop placeholder nodes. Runs after
    # un-merging so it sees the graph that will actually be served, and
    # reuses the violations the reporting pass just computed.
    repair_result = ShaclRepairResult(graph=state.aggregated_facts)
    if shapes_graph is not None:
        repair_result = apply_shacl_repairs(
            state.aggregated_facts,
            shapes_graph,
            ontology_graph,
            mode=facts_validation.shacl_autofix,
            passes=facts_validation.shacl_autofix_passes,
            fact_namespaces=fact_namespaces,
            code_predicates=facts_validation.code_predicates,
            inference=facts_validation.shacl_inference,
            advanced=facts_validation.shacl_advanced,
            max_triples=facts_validation.shacl_max_triples,
            initial_violations=(
                report.shacl_violations if report.shacl_evaluated else None
            ),
        )
        if repair_result.records:
            state.aggregated_facts = repair_result.graph
            state.facts_gate_repairs = repair_result.records
            report = run_validation()

    state.facts_validation_findings = report.findings
    state.facts_conformance = summarize_conformance(
        report.findings,
        shacl_evaluated=report.shacl_evaluated,
        repairs=state.facts_gate_repairs,
    )
    record_facts_gate_metrics(
        state.retrieval_metrics,
        report=report,
        repair_result=repair_result,
        ontology_context_empty=not len(ontology_graph),
    )
    if merge_repair:
        state.retrieval_metrics[RetrievalMetric.FACTS_MERGE_REPAIR_PASSES] = (
            repair_passes
        )
        state.retrieval_metrics[RetrievalMetric.FACTS_MERGE_VETOES] = len(vetoes)
        state.retrieval_metrics[RetrievalMetric.FACTS_MERGE_REPAIRS_REJECTED] = (
            rejected_repairs
        )
    if repair_passes and last_result is not None:
        # merge_facts recorded this against the pre-repair aggregation, so
        # it is stale once a veto pass has re-aggregated. Republish the
        # guard count for the graph that is actually served -- not the veto
        # count, which is a different quantity and is already published as
        # facts_merge_vetoes.
        state.retrieval_metrics[RetrievalMetric.FACTS_REJECTED_MERGES] = (
            last_result.rejected_merge_count
        )
    if report.error_findings:
        logger.warning(
            "Facts validation gate: %d error finding(s) remain after "
            "%d repair pass(es)",
            len(report.error_findings),
            repair_passes,
        )
