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
from collections.abc import Sequence
from itertools import combinations

from rdflib import BNode, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import (
    FactsGateRepairKind,
    FactsValidationFinding,
    FactsValidationFindingKind,
    GraphRepairRecord,
)
from ontocast.onto.rdfgraph import RDFGraph, drop_reifiers_mentioning
from ontocast.onto.state import AgentState
from ontocast.tool.agg.aggregate import AggregationResult
from ontocast.tool.agg.signatures import (
    canonical_literal,
    normalize_string_value,
    string_values_compatible,
)
from ontocast.tool.agg.value_names import carrierless_values, name_values
from ontocast.tool.facts_validation import (
    FactsValidationReport,
    ShaclRepairResult,
    ValidationPolicy,
    apply_shacl_repairs,
    collect_shacl_shapes,
    count_shacl_focus_nodes,
    dedupe_literal_variants,
    facts_connectivity,
    instance_links,
    links_lost,
    materialize_entailed_types,
    record_facts_gate_metrics,
    shacl_catalog_contradictions,
    silent_instances,
    summarize_conformance,
    twin_labels,
    untyped_instances,
    validate_aggregated_facts,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: How many dropped instance IRIs the state keeps; the metric holds the count.
_UNTYPED_DROPPED_STATE_CAP = 200

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


def _value_key(term: Node, canonical: dict[str, str]) -> tuple[str, str]:
    """``(kind, value)`` a member's object is compared by.

    IRIs by the cluster they belong to, numbers and dates by value, strings by
    their normalized form (compared for alias compatibility, not equality).
    """
    if isinstance(term, Literal):
        value = canonical_literal(term)
        if value is not None:
            return value[1], value[0]
        return "string", normalize_string_value(str(term))
    return "node", canonical.get(str(term), str(term))


def _values_conflict(left: tuple[str, str], right: tuple[str, str]) -> bool:
    if left[0] != right[0]:
        return False
    if left[0] == "string":
        return not string_values_compatible(left[1], right[1])
    return left[1] != right[1]


def _conflicting_pairs(
    members: Sequence[str],
    values: dict[str, set[tuple[str, str]]],
) -> set[frozenset[URIRef]]:
    """Member pairs whose values on a predicate share nothing compatible."""
    pairs: set[frozenset[URIRef]] = set()
    for left, right in combinations(sorted(members), 2):
        left_values, right_values = values.get(left), values.get(right)
        if not left_values or not right_values:
            continue
        if all(_values_conflict(a, b) for a in left_values for b in right_values):
            pairs.add(frozenset((URIRef(left), URIRef(right))))
    return pairs


def vetoes_from_findings(
    findings: list[FactsValidationFinding],
    clusters: dict[str, list[str]],
    units: Sequence[ContentUnit] = (),
) -> set[frozenset[URIRef]]:
    """Pair vetoes for merge-signature error findings.

    A finding implicates the cluster whose merge could have produced it. A
    merged subject collects the objects of all its members, so
    FUNCTIONAL_VIOLATION and SUSPECT_MULTI_VALUE implicate the subject; its
    objects are innocent (merging objects can only reduce the values a subject
    carries). DEGENERATE_COREFERENCE reports the *pointing* node as subject and
    the over-merged endpoint in ``values`` (``range1 hasLowerBound v1 ;
    hasUpperBound v1``); either merge can produce it, so it implicates both.

    Within an implicated cluster only the members that disagree are split:
    two members are vetoed when every value one carried on the finding's
    predicate, in its own unit, conflicts with every value the other carried
    (for DEGENERATE_COREFERENCE: when they use, or are reached by, different
    ones of its predicates). Members that carried nothing there keep their
    merges. When ``units`` cannot attribute the disagreement, or are not
    given, every pair of the implicated cluster is vetoed.

    Args:
        findings: Error findings from the validation report.
        clusters: Identity-merge clusters keyed by final IRI; members are the
            entity IRIs of the unit graphs.
        units: The aggregated units, whose graphs say which member carried
            which value.

    Returns:
        Pairs of IRIs that must not be merged on a re-aggregation pass.
    """
    canonical = {
        member: min(members) for members in clusters.values() for member in members
    }
    graphs = [unit.graph for unit in units if unit.graph is not None]
    vetoes: set[frozenset[URIRef]] = set()
    for finding in findings:
        if finding.kind not in MERGE_SIGNATURE_KINDS:
            continue
        degenerate = finding.kind == FactsValidationFindingKind.DEGENERATE_COREFERENCE
        implicated = (
            [finding.subject, *finding.values] if degenerate else [finding.subject]
        )
        for candidate in implicated:
            members = clusters.get(candidate, [])
            if len(members) < 2:
                continue
            values: dict[str, set[tuple[str, str]]] = {}
            if degenerate:
                # A member's "value" is which of the finding's predicates it
                # uses (pointing side) or is reached by (endpoint side): one
                # endpoint is a lower bound, the other an upper.
                predicates = [URIRef(p) for p in finding.predicate.split(", ") if p]
                pointing = candidate == finding.subject
                for graph in graphs:
                    for member in members:
                        node = URIRef(member)
                        for predicate in predicates:
                            pattern = (
                                (node, predicate, None)
                                if pointing
                                else (None, predicate, node)
                            )
                            if pattern in graph:
                                values.setdefault(member, set()).add(
                                    ("node", str(predicate))
                                )
            else:
                predicate = URIRef(finding.predicate)
                for graph in graphs:
                    for member in members:
                        for obj in graph.objects(URIRef(member), predicate):
                            values.setdefault(member, set()).add(
                                _value_key(obj, canonical)
                            )
            pairs = _conflicting_pairs(members, values)
            if not pairs:
                pairs = {
                    frozenset((URIRef(left), URIRef(right)))
                    for left, right in combinations(sorted(members), 2)
                }
            vetoes |= pairs
    return vetoes


def _pairs(pairs: Sequence[tuple[str, str]]) -> set[frozenset[URIRef]]:
    return {frozenset((URIRef(left), URIRef(right))) for left, right in pairs}


def stored_pairs(pairs: set[frozenset[URIRef]]) -> list[tuple[str, str]]:
    """Entity pairs as the state stores them: sorted ``(left, right)`` strings."""
    out: list[tuple[str, str]] = []
    for pair in pairs:
        left, right = sorted(str(iri) for iri in pair)
        out.append((left, right))
    return sorted(out)


def _remove_orphaned_blank_nodes(
    graph: RDFGraph, candidates: set[BNode]
) -> tuple[set[BNode], int]:
    """Remove blank nodes nothing references any more, cascading.

    A blank node is reachable only through its incoming triples; once those
    are gone, its own triples describe nothing, and the blank nodes *they*
    point at may be orphaned in turn.

    Returns:
        Tuple of (removed blank nodes, removed triple count).
    """
    removed: set[BNode] = set()
    removed_triples = 0
    pending = list(candidates)
    while pending:
        node = pending.pop()
        if node in removed or (None, None, node) in graph:
            continue
        removed.add(node)
        for triple in list(graph.triples((node, None, None))):
            graph.remove(triple)
            removed_triples += 1
            if isinstance(triple[2], BNode):
                pending.append(triple[2])
    return removed, removed_triples


def drop_untyped_instances(
    graph: RDFGraph, fact_namespaces: Sequence[str]
) -> tuple[RDFGraph, list[str]]:
    """Remove every instance that has no concrete type, with its triples.

    Runs on the aggregated graph so a type asserted in any unit counts. An
    untyped instance (:func:`untyped_instances`: a fact-namespace IRI in
    subject or IRI-object position whose ``rdf:type`` names no class outside
    the fact namespaces other than a metaclass or ``owl:Thing``) loses every
    triple it is the subject or the object of; blank nodes reachable only
    through those triples go with them, and so do the RDF 1.2 reifiers
    quoting any removed statement. A typed neighbour keeps everything else.

    Args:
        graph: Aggregated facts graph, mutated in place.
        fact_namespaces: Namespaces holding minted instances.

    Returns:
        Tuple of (the same graph, dropped instance IRIs ordered by IRI).
    """
    untyped = untyped_instances(graph, fact_namespaces)
    _remove_instances(graph, untyped)
    if untyped:
        logger.info(
            "Dropped %d untyped instance(s) from the aggregated facts", len(untyped)
        )
    return graph, [str(instance) for instance in untyped]


def drop_unlinked_instances(
    graph: RDFGraph, fact_namespaces: Sequence[str], document: URIRef | None
) -> tuple[RDFGraph, list[str], int]:
    """Remove every instance that states nothing, and every value no statement carries.

    Served facts are statements: an instance linked to no other instance nor
    to ``document``, holding no value of its own
    (:func:`~ontocast.tool.facts_validation.connectivity.silent_instances`),
    is a type and a name, a mention rather than a fact; a blank-node value
    nothing points to has lost its statement. Runs on the aggregated graph,
    so a mention another unit links has merged first.

    Args:
        graph: Aggregated facts graph, mutated in place.
        fact_namespaces: Namespaces holding minted instances.
        document: The document node; a link to it counts.

    Returns:
        Tuple of (the same graph, dropped instance IRIs ordered by IRI, the
        number of carrierless values dropped).
    """
    silent = sorted(silent_instances(graph, fact_namespaces, document), key=str)
    _remove_instances(graph, silent)
    orphaned, _ = _remove_orphaned_blank_nodes(graph, carrierless_values(graph))
    drop_reifiers_mentioning(graph, orphaned)
    if silent or orphaned:
        logger.info(
            "Dropped %d instance(s) stating nothing and %d value(s) with no "
            "carrier from the aggregated facts",
            len(silent),
            len(orphaned),
        )
    return graph, [str(instance) for instance in silent], len(orphaned)


def _remove_instances(graph: RDFGraph, instances: Sequence[URIRef]) -> None:
    """Remove every triple naming ``instances``, the blank nodes only those
    triples reached, and the reifiers quoting any removed statement."""
    if not instances:
        return
    orphan_candidates: set[BNode] = set()
    removed_nodes: set[Node] = set(instances)
    for instance in instances:
        for triple in [
            *graph.triples((instance, None, None)),
            *graph.triples((None, None, instance)),
        ]:
            graph.remove(triple)
            for term in (triple[0], triple[2]):
                if isinstance(term, BNode):
                    orphan_candidates.add(term)
    orphaned, _ = _remove_orphaned_blank_nodes(graph, orphan_candidates)
    removed_nodes |= orphaned
    drop_reifiers_mentioning(graph, removed_nodes)


def run_facts_gate(
    state: AgentState,
    ontology_graph: RDFGraph,
    tools: ToolBox,
    *,
    merge_repair: bool,
    document_metadata: dict | None = None,
) -> None:
    """Validate the aggregated facts and apply the LLM-free repair stages.

    Literal variants differing only in language tag or datatype are collapsed
    first (``FACTS_LITERAL_VARIANT_DEDUPE``), so validation counts values
    rather than spellings.

    When ``merge_repair`` is set, merge-signature error findings whose subject
    resulted from an identity merge turn the offending cluster into pair vetoes
    and the retained facts units are re-aggregated, up to
    ``FACTS_MERGE_REPAIR_PASSES`` times. Then, on both paths, SHACL autofix
    (``FACTS_SHACL_AUTOFIX``) repairs constraint violations in code, instances
    still untyped take the types their links' domains and ranges entail in the
    whole ontology context (``facts_untyped_inferred``), and those still
    untyped are dropped (:func:`drop_untyped_instances`;
    ``facts_untyped_dropped``), and so are instances that state nothing
    (:func:`drop_unlinked_instances`; ``facts_unlinked_dropped``). The links
    these repairs removed
    (``facts_links_removed``), how the served instances hang together
    (``facts_components``, ``facts_isolated``) and the labels several of them
    share (``facts_twin_labels``) are recorded. Residual findings stay on the
    state as telemetry.

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
    shapes_graph = collect_shacl_shapes(ontology_graph, tools.shapes_catalog.graph())
    # The check must see what the unit validator sees, or it reports
    # contradictions no unit ever raised: the whole catalog rather than the
    # union of retrieved snapshots (a term a snapshot omitted is still a
    # term), and the shapes-contract exemptions the unit loop applies.
    _, contract_terms, _ = tools.shapes_prompt_contract()
    catalog_terms = tools.ontology_manager.catalog_terms()
    contradictions = shacl_catalog_contradictions(
        shapes_graph,
        ontology_graph,
        policy=ValidationPolicy(
            additional_standard_namespaces=tuple(
                facts_validation.additional_standard_namespaces
            ),
            quantity_fallback_vocabulary=facts_validation.quantity_fallback_vocabulary,
            code_predicates=tuple(facts_validation.code_predicates),
            contract_exempt_terms=contract_terms,
            vocabulary=facts_validation.vocabulary,
        ),
        catalog_terms=catalog_terms,
    )
    if contradictions:
        # Data cannot satisfy both sides: the shapes demand these properties
        # while the unit validator's mandatory findings order the renderer to
        # remove them. This is a catalog/configuration error, and it silently
        # destroys extracted data (observed live: qudt:numericValue on the
        # catalog). Loud, per document, until the catalog declares the
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
            key_supported_subjects=state.aggregation_key_clusters,
            cross_unit_pairs=(
                state.aggregation_cross_unit_pairs
                if facts_validation.suspect_multi_value_require_cross_unit
                else None
            ),
            non_catalog_severity=(
                "error" if facts_validation.vocabulary == "strict" else "warning"
            ),
            additional_standard_namespaces=tuple(
                facts_validation.additional_standard_namespaces
            ),
            catalog_terms=catalog_terms,
        )

    def merge_signature_errors(report: FactsValidationReport) -> int:
        return sum(
            1
            for finding in report.error_findings
            if finding.kind in MERGE_SIGNATURE_KINDS
        )

    # Collapse literal variants ("X"@en / "X"^^xsd:string / "X") before
    # validating, so multi-value checks count values rather than spellings.
    dedupe_records = (
        dedupe_literal_variants(state.aggregated_facts, fact_namespaces)
        if facts_validation.literal_variant_dedupe
        else []
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
            report.error_findings, state.aggregation_clusters, state.facts_units
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
            merge_vetoes=vetoes | _pairs(state.coreference_vetoes),
            confirmed_pairs=_pairs(state.coreference_confirmations),
        )
        # Un-merging is destructive: a veto splits correct merges too, so a
        # pass that does not strictly reduce the error count has traded real
        # coreference for nothing and must not be kept.
        previous_graph = state.aggregated_facts
        previous_clusters = state.aggregation_clusters
        previous_key_clusters = state.aggregation_key_clusters
        previous_cross_unit_pairs = state.aggregation_cross_unit_pairs
        state.aggregated_facts = result.graph
        state.aggregation_clusters = result.merged_clusters
        state.aggregation_key_clusters = result.key_supported_clusters
        # Re-aggregation rewrites subject IRIs, so the pairs the validator
        # keys on have to come from the pass it is about to score.
        state.aggregation_cross_unit_pairs = result.cross_unit_object_pairs
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
            state.aggregation_key_clusters = previous_key_clusters
            state.aggregation_cross_unit_pairs = previous_cross_unit_pairs
            rejected_repairs += 1
            break
        repair_passes += 1
        last_result = result
        report = candidate_report

    if repair_passes and facts_validation.literal_variant_dedupe:
        # An accepted un-merge pass re-aggregated from the raw units, which
        # reintroduces the literal variants the pre-validation pass removed.
        dedupe_records.extend(
            dedupe_literal_variants(state.aggregated_facts, fact_namespaces)
        )

    # Shape-driven repair, in code: retype literals against sh:datatype,
    # resolve codes to catalog IRIs, drop placeholder nodes. Runs after
    # un-merging so it sees the graph that will actually be served, and
    # reuses the violations the reporting pass just computed.
    links_before = instance_links(state.aggregated_facts, fact_namespaces)
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

    # Every instance the served graph holds is typed. An instance still
    # untyped once all units have contributed their types takes the class its
    # links entail; whatever stays untyped is removed, with its links.
    # Units materialise entailed types against their own context, which may
    # lack the domain or range the whole context declares.
    inferred_records: list[GraphRepairRecord] = []
    untyped_before = set(untyped_instances(state.aggregated_facts, fact_namespaces))
    if untyped_before:
        _, inferred_records = materialize_entailed_types(
            state.aggregated_facts, ontology_graph
        )
    state.retrieval_metrics[RetrievalMetric.FACTS_UNTYPED_INFERRED] = len(
        untyped_before - set(untyped_instances(state.aggregated_facts, fact_namespaces))
    )
    _, untyped_dropped = drop_untyped_instances(state.aggregated_facts, fact_namespaces)
    untyped_records = [
        GraphRepairRecord(
            kind=FactsGateRepairKind.UNTYPED_DROPPED, source=iri, target=""
        )
        for iri in untyped_dropped
    ]
    state.facts_untyped_dropped = untyped_dropped[:_UNTYPED_DROPPED_STATE_CAP]
    state.retrieval_metrics[RetrievalMetric.FACTS_UNTYPED_DROPPED] = len(
        untyped_dropped
    )
    _, unlinked_dropped, values_dropped = drop_unlinked_instances(
        state.aggregated_facts, fact_namespaces, state.doc_iri
    )
    untyped_records.extend(
        GraphRepairRecord(
            kind=FactsGateRepairKind.UNLINKED_DROPPED, source=iri, target=""
        )
        for iri in unlinked_dropped
    )
    state.retrieval_metrics[RetrievalMetric.FACTS_UNLINKED_DROPPED] = (
        len(unlinked_dropped) + values_dropped
    )

    state.retrieval_metrics[RetrievalMetric.FACTS_LINKS_REMOVED] = links_lost(
        links_before, state.aggregated_facts
    )
    connectivity = facts_connectivity(
        state.aggregated_facts, fact_namespaces, document=state.doc_iri
    )
    state.retrieval_metrics[RetrievalMetric.FACTS_COMPONENTS] = connectivity.components
    state.retrieval_metrics[RetrievalMetric.FACTS_ISOLATED] = connectivity.isolated
    state.retrieval_metrics[RetrievalMetric.FACTS_TWIN_LABELS] = len(
        twin_labels(state.aggregated_facts, fact_namespaces, state.doc_iri)
    )
    # Values are blank nodes under their carriers until here, so the counts
    # above see them as parts of statements; the served graph names them.
    name_values(state.aggregated_facts)

    if repair_result.records or untyped_records or inferred_records:
        report = run_validation()

    if dedupe_records or repair_result.records or untyped_records or inferred_records:
        state.facts_gate_repairs = (
            dedupe_records + repair_result.records + inferred_records + untyped_records
        )

    state.facts_validation_findings = report.findings
    state.facts_conformance = summarize_conformance(
        report.findings,
        shacl_evaluated=report.shacl_evaluated,
        repairs=state.facts_gate_repairs,
        focus_nodes=count_shacl_focus_nodes(state.aggregated_facts, shapes_graph),
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
