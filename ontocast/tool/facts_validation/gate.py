"""Document-level validation for the post-aggregation facts gate.

Merge-signature findings, SHACL conformance summary, non-catalog vocabulary
telemetry, dangling references, and the gate metrics record.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping, Sequence
from itertools import combinations

from pydantic import BaseModel, Field
from rdflib import OWL, RDF, RDFS, Literal, URIRef
from rdflib.namespace import PROV, SH, XSD

from ontocast.onto.constants import PROVENANCE_METADATA_TERMS
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import (
    FactsValidationFinding,
    FactsValidationFindingKind,
    GraphRepairRecord,
)
from ontocast.onto.rdfgraph import (
    RDFGraph,
)
from ontocast.tool.agg.signatures import (
    canonical_literal,
    harvest_max_one_predicates,
    normalize_string_value,
    string_values_compatible,
)
from ontocast.tool.facts_validation.shacl import (
    ShaclRepairResult,
    ShaclViolation,
    _violation_in_fact_scope,
    run_shacl,
)
from ontocast.tool.facts_validation.terms import (
    _in_fact_scope,
    _local_name,
    _namespace_of,
    collect_catalog_terms,
)

logger = logging.getLogger(__name__)


class FactsValidationReport(BaseModel):
    """Invariant findings over one aggregated facts graph."""

    model_config = {"arbitrary_types_allowed": True}

    findings: list[FactsValidationFinding] = Field(default_factory=list)
    shacl_evaluated: bool | None = Field(
        default=None,
        description=(
            "True when SHACL ran, False when shapes were configured but it "
            "could not (pyshacl missing, graph over the size guard), None when "
            "no shapes were in play. 'No SHACL findings' means nothing without "
            "this: it reads identically for 'conforms' and 'never checked'."
        ),
    )
    shacl_violations: list["ShaclViolation"] = Field(
        default_factory=list,
        exclude=True,
        repr=False,
        description=(
            "Raw, unfiltered pyshacl violations, kept so the autofix pass can "
            "reuse them instead of re-running validation. Internal: never "
            "serialized."
        ),
    )

    @property
    def error_findings(self) -> list[FactsValidationFinding]:
        """Error-severity findings, whatever their kind."""
        return [finding for finding in self.findings if finding.severity == "error"]


def count_shacl_focus_nodes(
    data_graph: RDFGraph, shapes_graph: RDFGraph | None
) -> int | None:
    """Count data-graph nodes any shape actually targets.

    This is the denominator ``conforms`` is silent about. A validation run over
    zero focus nodes reports no violations for the same reason an empty query
    returns no rows -- nothing was examined. Reported alongside ``conforms`` so
    a clean result cannot be read as a passing one when the shapes and the data
    never met.

    Class targeting follows ``rdfs:subClassOf`` because SHACL's
    ``sh:targetClass`` is subclass-aware.

    Args:
        data_graph: The graph that was validated.
        shapes_graph: The shapes it was validated against; ``None`` when SHACL
            did not run.

    Returns:
        int | None: Number of distinct focus nodes, or ``None`` if no shapes.
    """
    if shapes_graph is None or not len(shapes_graph):
        return None
    focus: set = set()
    for target_class in shapes_graph.objects(None, SH.targetClass):
        classes = {target_class}
        # Walk down the hierarchy: a shape on a superclass targets instances
        # of every subclass too.
        pending = [target_class]
        while pending:
            current = pending.pop()
            for sub in data_graph.subjects(RDFS.subClassOf, current):
                if sub not in classes:
                    classes.add(sub)
                    pending.append(sub)
        for klass in classes:
            focus.update(data_graph.subjects(RDF.type, klass))
    focus.update(shapes_graph.objects(None, SH.targetNode))
    for predicate in shapes_graph.objects(None, SH.targetSubjectsOf):
        focus.update(data_graph.subjects(predicate, None))
    for predicate in shapes_graph.objects(None, SH.targetObjectsOf):
        focus.update(data_graph.objects(None, predicate))
    return len(focus)


def summarize_conformance(
    findings: Sequence[FactsValidationFinding],
    *,
    shacl_evaluated: bool | None = None,
    repairs: Sequence[GraphRepairRecord] = (),
    focus_nodes: int | None = None,
) -> dict:
    """Roll findings up into the shape a report or a client can read.

    Counting by constraint component is what separates "168 violations" from
    "two systematic defects": 71 missing-qualifier violations on one shape are
    one modelling gap, not 71 problems to triage.

    Args:
        findings: Residual findings after any repair.
        shacl_evaluated: Whether SHACL actually ran (see
            :class:`FactsValidationReport`).
        repairs: LLM-free repairs the gate applied.
        focus_nodes: Data-graph nodes the shapes actually target, from
            :func:`count_shacl_focus_nodes`. Zero means the shapes examined
            nothing, so a violation-free result says nothing about the data.

    Returns:
        ``conforms`` (None when SHACL did not run **or** when it ran over an
        empty focus set), counts by severity, by finding kind, by SHACL
        constraint component and shape, and the applied repair counts by kind.
        ``shacl_vacuous`` flags the empty-focus-set case, which is a
        measurement failure rather than a passing grade.
    """
    shacl_findings = [
        finding
        for finding in findings
        if finding.kind == FactsValidationFindingKind.SHACL
    ]
    by_kind: dict[str, int] = {}
    for finding in findings:
        by_kind[str(finding.kind)] = by_kind.get(str(finding.kind), 0) + 1
    by_component: dict[str, int] = {}
    by_shape: dict[str, int] = {}
    for finding in shacl_findings:
        if finding.component:
            key = _local_name(finding.component) or finding.component
            by_component[key] = by_component.get(key, 0) + 1
        if finding.source_shape:
            by_shape[finding.source_shape] = by_shape.get(finding.source_shape, 0) + 1
    repairs_by_kind: dict[str, int] = {}
    for record in repairs:
        repairs_by_kind[str(record.kind)] = repairs_by_kind.get(str(record.kind), 0) + 1

    vacuous = bool(shacl_evaluated) and focus_nodes == 0
    return {
        "shacl_evaluated": shacl_evaluated,
        "conforms": (None if not shacl_evaluated or vacuous else not shacl_findings),
        "shacl_focus_nodes": focus_nodes,
        "shacl_vacuous": vacuous,
        "findings": len(findings),
        "errors": sum(1 for finding in findings if finding.severity == "error"),
        "warnings": sum(1 for finding in findings if finding.severity == "warning"),
        "by_kind": dict(sorted(by_kind.items())),
        "shacl_violations": len(shacl_findings),
        "shacl_by_constraint": dict(
            sorted(by_component.items(), key=lambda item: (-item[1], item[0]))
        ),
        "shacl_by_shape": dict(
            sorted(by_shape.items(), key=lambda item: (-item[1], item[0]))
        ),
        "repairs_applied": dict(sorted(repairs_by_kind.items())),
    }


def _distinct_object_keys(objects: set) -> set[str]:
    """Distinct-value keys for a mixed object set (canonical for literals)."""
    keys: set[str] = set()
    for obj in objects:
        if isinstance(obj, Literal):
            canonical = canonical_literal(obj)
            keys.add(f"lit:{canonical[0]}" if canonical else f"lex:{obj}")
        else:
            keys.add(f"iri:{obj}")
    return keys


def _dominant_single_valued_predicates(
    iri_groups: dict[tuple[URIRef, URIRef], set],
    *,
    min_single_support: int,
) -> set[URIRef]:
    """Predicates whose IRI objects are single-valued for a dominant majority.

    Unlike the strict corpus inference used by merge guards, this tolerates a
    minority of multi-valued subjects — those are exactly the violation
    candidates the gate reports (e.g. one quantity node carrying two
    ``qudt:unit`` IRIs after a bad merge, while every other quantity node
    carries one).
    """
    single: dict[URIRef, int] = {}
    multi: dict[URIRef, int] = {}
    for (_, predicate), objects in iri_groups.items():
        bucket = single if len(objects) == 1 else multi
        bucket[predicate] = bucket.get(predicate, 0) + 1
    return {
        predicate
        for predicate, count in single.items()
        if count >= min_single_support and count >= 3 * multi.get(predicate, 0)
    }


def record_facts_gate_metrics(
    metrics: MutableMapping[str, int | float | str | dict],
    *,
    report: FactsValidationReport,
    repair_result: ShaclRepairResult,
    ontology_context_empty: bool = False,
) -> None:
    """Write the validation-gate metrics both entry paths share.

    The graph pipeline's ``VALIDATE_FACTS`` node and the single-unit gate behind
    ``/process_unit`` run the same checks minus the un-merge repair, and had
    drifted into two hand-maintained copies of these writes — so a metric added
    to one path was silently absent from the other, and batch dumps stopped
    being comparable across entry paths, which is the one thing they exist for.
    Merge-specific counters stay with the graph pipeline: they have no meaning
    for a single unit.

    Takes a plain mapping rather than ``AgentState`` so the tool layer stays
    ignorant of the state graph.

    Args:
        metrics: ``AgentState.retrieval_metrics``, mutated in place.
        report: Validation report describing the graph that will be served.
        repair_result: Outcome of :func:`apply_shacl_repairs`. Its counters are
            written only when the pass actually ran, so "SHACL did not run"
            stays distinguishable from "ran and found nothing".
        ontology_context_empty: Whether the facts were validated with no
            catalog vocabulary at all. The per-term non-catalog check cannot
            see this — with no context there is nothing to compare against — so
            it is reported here, where an empty context is known to be
            unexpected. Only the document path used to report it, which left
            ``/process_unit`` silently unable to say the same thing.
    """
    if ontology_context_empty:
        reason = metrics.get(
            RetrievalMetric.EMPTY_SNAPSHOT_REASON, "no ontology context was assembled"
        )
        logger.warning(
            "Validating facts against an empty ontology context (%s); every "
            "extracted term is outside the catalog.",
            reason,
        )
        metrics[RetrievalMetric.VALIDATED_WITHOUT_ONTOLOGY_CONTEXT] = True
    if repair_result.ran:
        metrics[RetrievalMetric.FACTS_SHACL_VIOLATIONS_BEFORE] = (
            repair_result.violations_before
        )
        metrics[RetrievalMetric.FACTS_SHACL_VIOLATIONS_AFTER] = (
            repair_result.violations_after
        )
        metrics[RetrievalMetric.FACTS_SHACL_REPAIRS] = len(repair_result.records)
        metrics[RetrievalMetric.FACTS_SHACL_AUTOFIX_PASSES] = (
            repair_result.passes_applied
        )
        metrics[RetrievalMetric.FACTS_SHACL_AUTOFIX_REVERTED] = repair_result.reverted
    metrics[RetrievalMetric.FACTS_VALIDATION_FINDINGS] = len(report.findings)
    metrics[RetrievalMetric.FACTS_VALIDATION_ERRORS] = len(report.error_findings)


_SCAFFOLDING_NAMESPACES = (str(RDF), str(RDFS), str(OWL), str(XSD))

# String values longer than this are prose payloads (descriptions, notes),
# not names or identifiers, and stay out of the string multi-value check —
# two chunks legitimately describe one entity in different words.
_NAME_LIKE_MAX_VALUE_LENGTH = 64


def _non_catalog_vocabulary_findings(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    namespaces: list[str],
    quantity_fallback_vocabulary: dict[str, str] | None = None,
) -> list[FactsValidationFinding]:
    """Report terms the facts graph uses that the ontology context never supplied.

    When the renderer cannot find a term for something the text states, the
    prompt's documented fallback is to reach for a well-known vocabulary
    instead. That is the right behaviour -- the alternative is dropping the
    fact -- but it is silent, and it looks identical to success: a clean graph
    scoring in the nineties, expressed in vocabulary the catalog does not
    contain and downstream queries will not match. A fallback firing is
    evidence of a *retrieval* miss, so it is worth a finding even though the
    facts graph itself is well-formed.

    Warning severity: this is telemetry about context assembly, not the
    signature of a bad merge, and must not drive the un-merge repair.

    Args:
        graph: Aggregated facts graph.
        ontology_graph: Merged ontology context that was offered to the renderer.
        namespaces: Fact namespaces; terms minted there are the extraction, not
            borrowed vocabulary.

    Returns:
        list: One finding per non-catalog term, ordered by IRI.
    """
    # A pure graph-vs-graph check: with nothing to compare against there is no
    # per-term finding to make. The *deployment* condition this used to mask --
    # extraction proceeding with no catalog vocabulary at all -- is reported by
    # the validate node, which knows whether an empty context was expected.
    if ontology_graph is None or not len(ontology_graph):
        return []

    fallback_namespaces = _fallback_namespaces(graph, quantity_fallback_vocabulary)
    # Must match UNKNOWN_TERM's notion of "in the catalog" (collect_catalog_terms,
    # all three triple positions). A subjects()-only view calls a term supplied
    # solely as an rdfs:range object non-catalog -- exactly the terms the
    # domain/range schema closure is designed to admit.
    supplied = collect_catalog_terms(ontology_graph)
    used: dict[str, set[str]] = {}
    for subject, predicate, obj in graph:
        if not isinstance(subject, URIRef) or str(predicate).startswith(str(PROV)):
            continue
        if predicate == RDF.type:
            if isinstance(obj, URIRef):
                used.setdefault(str(obj), set()).add(str(subject))
            continue
        used.setdefault(str(predicate), set()).add(str(subject))

    findings: list[FactsValidationFinding] = []
    for term in sorted(used):
        if term in supplied:
            continue
        if term.startswith(_SCAFFOLDING_NAMESPACES):
            continue
        # Chunk-metadata terms the pipeline mints itself. The prov: guard above
        # only covers predicates, so schema:position, schema:identifier and the
        # chunk node's own prov:Entity / schema:Text types were reported as
        # vocabulary the renderer improvised, on every run whose catalog does
        # not happen to include prov and schema.org. No catalog will ever
        # supply them: they are scaffolding, like rdfs:label.
        if URIRef(term) in PROVENANCE_METADATA_TERMS:
            continue
        if any(term.startswith(ns) for ns in namespaces):
            continue
        subjects = sorted(used[term])
        # A term the deployment itself named as the fallback is a configured
        # choice the prompt instructed. Still a retrieval miss worth reporting,
        # but not the same signal as vocabulary invented out of nowhere -- the
        # message must not accuse the renderer of improvising what it was told.
        configured = _is_configured_fallback(term, fallback_namespaces)
        findings.append(
            FactsValidationFinding(
                kind=FactsValidationFindingKind.NON_CATALOG_VOCABULARY,
                severity="warning",
                message=(
                    f"<{term}> is used by {len(subjects)} subject(s) but was "
                    "absent from the ontology context — the renderer took the "
                    "configured fallback vocabulary, which points at a "
                    "retrieval miss rather than a modeling choice."
                    if configured
                    else f"<{term}> is used by {len(subjects)} subject(s) but "
                    "was absent from the ontology context — the renderer fell "
                    "back to outside vocabulary, which points at a retrieval "
                    "miss rather than a modeling choice."
                ),
                subject=subjects[0],
                predicate=term,
                values=subjects,
            )
        )
    return findings


def _fallback_namespaces(
    graph: RDFGraph, fallback_vocabulary: dict[str, str] | None
) -> tuple[str, ...]:
    """Resolve configured fallback terms to namespace prefixes.

    Configured terms are CURIEs (``qudt:QuantityValue``) or full IRIs. CURIEs
    are expanded against the graph's own prefix bindings, which is where the
    renderer declared them.

    Args:
        graph: Facts graph, used for its prefix bindings.
        fallback_vocabulary: Configured role -> term mapping.

    Returns:
        tuple: Namespace IRIs, usable with ``str.startswith``.
    """
    if not fallback_vocabulary:
        return ()
    bindings = {prefix: str(uri) for prefix, uri in graph.namespaces()}
    namespaces: set[str] = set()
    for term in fallback_vocabulary.values():
        if not term:
            continue
        if term.startswith("http://") or term.startswith("https://"):
            namespaces.add(_namespace_of(term))
            continue
        prefix, separator, _ = term.partition(":")
        if separator and prefix in bindings:
            namespaces.add(bindings[prefix])
    return tuple(sorted(namespaces))


def _is_configured_fallback(term: str, fallback_namespaces: tuple[str, ...]) -> bool:
    """True when *term* lives in a namespace the deployment configured."""
    return bool(fallback_namespaces) and term.startswith(fallback_namespaces)


def validate_aggregated_facts(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    *,
    shapes_graph: RDFGraph | None = None,
    fact_namespaces: list[str] | None = None,
    suspect_multi_value_severity: str = "error",
    functional_min_single_support: int = 3,
    quantity_fallback_vocabulary: dict[str, str] | None = None,
    shacl_inference: str = "rdfs",
    shacl_advanced: bool = True,
    shacl_max_triples: int = 0,
    key_supported_subjects: Sequence[str] | None = None,
    cross_unit_pairs: Sequence[tuple[str, str]] | None = None,
) -> FactsValidationReport:
    """Check post-merge invariants over the aggregated facts graph.

    Deterministic defense-in-depth behind the merge guards: merge-signature
    violations here are almost always a bad identity merge, and error-severity
    findings of those kinds on merged subjects drive the un-merge repair.
    SHACL findings are reported but never drive it: a constraint violation
    says a node is under-specified, not that two entities were wrongly
    identified.

    Checks:
        - ``FUNCTIONAL_VIOLATION``: >= 2 distinct objects on a predicate the
          schema constrains to at most one value (``owl:FunctionalProperty``
          or an OWL max-cardinality-1 restriction).
        - ``SUSPECT_MULTI_VALUE``: >= 2 distinct canonical numeric values on
          one (subject, predicate); >= 2 mutually irreconcilable short string
          values on a predicate that is string-single-valued for a dominant
          majority (distinct names collapsed into one node); or >= 2 IRI
          objects on a predicate that is single-valued for a dominant
          majority of other subjects. Severity is configurable — legitimate
          multi-value modeling exists, bad merges are far more common. The
          IRI branch additionally accepts ``cross_unit_pairs``, which
          separates the two by provenance rather than by frequency.
        - ``DEGENERATE_COREFERENCE``: one IRI object shared by >= 2 distinct
          functional-ish predicates of one subject (collapsed range bounds).
        - ``SHACL``: optional, when ``pyshacl`` is installed and shapes exist.
        - ``NON_CATALOG_VOCABULARY``: warning-only telemetry for terms the
          ontology context never supplied, which mark a retrieval miss the
          renderer papered over with a documented fallback.
        - ``MIXED_OBJECT_KINDS``: warning-only telemetry for predicates used
          with both IRI and literal objects across the graph.

    Args:
        graph: Aggregated facts graph.
        ontology_graph: Merged ontology context (functionality harvest).
        shapes_graph: Optional SHACL shapes graph.
        fact_namespaces: When set, only subjects under these namespaces are
            reported (ontology entities are not the gate's business).
        suspect_multi_value_severity: ``"error"`` or ``"warning"`` for
            SUSPECT_MULTI_VALUE findings.
        functional_min_single_support: Minimum single-valued subjects before a
            predicate counts as dominantly single-valued.
        shacl_inference: pyshacl pre-inference mode (see :func:`run_shacl`).
        shacl_advanced: Enable SHACL Advanced Features.
        shacl_max_triples: Skip SHACL above this graph size; 0 disables.
        cross_unit_pairs: Canonical (subject, predicate) pairs whose IRI
            objects came from more than one unit. When supplied, an
            IRI-branch SUSPECT_MULTI_VALUE finding on a pair *not* listed
            here is reported as a warning and never vetoes a cluster: a
            single unit asserting two objects on one predicate is reading
            one sentence, not the residue of a bad identity decision.
            None disables the distinction.
        key_supported_subjects: Final URIs of merge clusters backed by
            natural-key evidence. Irreconcilable *string* values on these
            subjects are reported as warnings, not errors: "Application no.
            36760/06" and "Case of Stanev v. Bulgaria" are two names for one
            key-confirmed case, and an error here would drive the un-merge
            repair to split a correct merge.

    Returns:
        Report with all findings, ordered by subject.
    """
    namespaces = [ns for ns in (fact_namespaces or []) if ns]
    key_supported = set(key_supported_subjects or ())
    functional = harvest_max_one_predicates(ontology_graph)

    # Provenance machinery (chunk nodes, derivation annotations) is
    # legitimately multi-valued and never the gate's business.
    provenance_subjects = {
        subject
        for subject in graph.subjects(RDF.type, PROV.Entity)
        if isinstance(subject, URIRef)
    }

    object_groups: dict[tuple[URIRef, URIRef], set] = {}
    iri_groups: dict[tuple[URIRef, URIRef], set[URIRef]] = {}
    string_groups: dict[tuple[URIRef, URIRef], set[str]] = {}
    predicate_object_kinds: dict[URIRef, dict[str, int]] = {}
    for subject, predicate, obj in graph:
        if (
            not isinstance(subject, URIRef)
            or not isinstance(predicate, URIRef)
            or predicate == RDF.type
            # owl:sameAs is the aggregator's own merge bookkeeping: the rewriter
            # emits `canonical owl:sameAs original` per remapped entity, so it
            # carries 1 object for an unmerged entity and N-1 for a cluster.
            # Left in, it reads as "dominantly single-valued" and every large
            # cluster becomes an error that drives the repair to un-merge it.
            or predicate == OWL.sameAs
            or subject in provenance_subjects
            or str(predicate).startswith(str(PROV))
        ):
            continue
        object_groups.setdefault((subject, predicate), set()).add(obj)
        if isinstance(obj, URIRef):
            iri_groups.setdefault((subject, predicate), set()).add(obj)
        elif isinstance(obj, Literal) and canonical_literal(obj) is None:
            normalized = normalize_string_value(str(obj))
            if 0 < len(normalized) <= _NAME_LIKE_MAX_VALUE_LENGTH:
                string_groups.setdefault((subject, predicate), set()).add(normalized)
        if _in_fact_scope(subject, namespaces) and predicate != RDFS.label:
            kinds = predicate_object_kinds.setdefault(predicate, {})
            kind = "iri" if isinstance(obj, URIRef) else "literal"
            kinds[kind] = kinds.get(kind, 0) + 1

    dominant_single = _dominant_single_valued_predicates(
        iri_groups, min_single_support=functional_min_single_support
    )
    # String-valued analogue, over short name-like values: the signature of
    # distinct entities collapsed into one node is a naming predicate that is
    # single-valued everywhere else suddenly carrying several irreconcilable
    # names.
    dominant_single_strings = _dominant_single_valued_predicates(
        string_groups, min_single_support=functional_min_single_support
    )
    functional_ish = functional | dominant_single

    findings: list[FactsValidationFinding] = []
    flagged_pairs: set[tuple[URIRef, URIRef]] = set()
    cross_unit_object_keys = (
        set(cross_unit_pairs) if cross_unit_pairs is not None else None
    )

    for (subject, predicate), objects in sorted(
        object_groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        if not _in_fact_scope(subject, namespaces):
            continue
        distinct = _distinct_object_keys(objects)
        if predicate in functional and len(distinct) >= 2:
            flagged_pairs.add((subject, predicate))
            findings.append(
                FactsValidationFinding(
                    kind=FactsValidationFindingKind.FUNCTIONAL_VIOLATION,
                    message=(
                        f"<{subject}> holds {len(distinct)} distinct values for "
                        f"<{predicate}>, which the schema constrains to at most "
                        "one."
                    ),
                    subject=str(subject),
                    predicate=str(predicate),
                    values=sorted(str(obj) for obj in objects),
                )
            )
            continue

        numeric_values = {
            canonical[0]
            for obj in objects
            if isinstance(obj, Literal)
            and (canonical := canonical_literal(obj)) is not None
            and canonical[1] == "numeric"
        }
        if len(numeric_values) >= 2:
            flagged_pairs.add((subject, predicate))
            findings.append(
                FactsValidationFinding(
                    kind=FactsValidationFindingKind.SUSPECT_MULTI_VALUE,
                    severity=(
                        "error"
                        if suspect_multi_value_severity == "error"
                        else "warning"
                    ),
                    message=(
                        f"<{subject}> carries {len(numeric_values)} distinct "
                        f"numeric values on <{predicate}> — the signature of "
                        "distinct quantities collapsed into one node."
                    ),
                    subject=str(subject),
                    predicate=str(predicate),
                    values=sorted(numeric_values),
                )
            )
            continue

        string_forms = sorted(string_groups.get((subject, predicate), set()))
        if len(string_forms) >= 2 and predicate in dominant_single_strings:
            # Name variants of one entity are alias-compatible ("mr beer" /
            # "mr karlheinz beer"); irreconcilable values ("mrs e palm" /
            # "mrs w thomassen") mark distinct entities collapsed into one
            # node — the failure the numeric branch cannot see. On a merge
            # backed by a shared identifier value, disagreement is name
            # variance of one confirmed entity: warning, never a veto.
            if any(
                not string_values_compatible(left, right)
                for left, right in combinations(string_forms, 2)
            ):
                subject_key_supported = str(subject) in key_supported
                if not subject_key_supported:
                    flagged_pairs.add((subject, predicate))
                findings.append(
                    FactsValidationFinding(
                        kind=FactsValidationFindingKind.SUSPECT_MULTI_VALUE,
                        severity=(
                            "error"
                            if suspect_multi_value_severity == "error"
                            and not subject_key_supported
                            else "warning"
                        ),
                        message=(
                            f"<{subject}> carries {len(string_forms)} "
                            f"mutually irreconcilable string values on "
                            f"<{predicate}>, which is single-valued for a "
                            "dominant majority of other subjects — the "
                            "signature of distinct entities collapsed into "
                            "one node."
                        ),
                        subject=str(subject),
                        predicate=str(predicate),
                        values=string_forms,
                    )
                )
                continue

        iri_objects = iri_groups.get((subject, predicate), set())
        if (
            len(iri_objects) >= 2
            and predicate not in functional
            and predicate in dominant_single
        ):
            # Frequency says this predicate is usually single-valued, which on
            # its own is evidence of nothing: a genuinely multi-valued
            # statement is rare by construction. Provenance is what separates
            # the two cases -- only objects arriving from different units could
            # have been brought together by an identity decision.
            merge_created = (
                cross_unit_object_keys is None
                or (str(subject), str(predicate)) in cross_unit_object_keys
            )
            if merge_created:
                flagged_pairs.add((subject, predicate))
            findings.append(
                FactsValidationFinding(
                    kind=FactsValidationFindingKind.SUSPECT_MULTI_VALUE,
                    severity=(
                        "error"
                        if suspect_multi_value_severity == "error" and merge_created
                        else "warning"
                    ),
                    message=(
                        f"<{subject}> points at {len(iri_objects)} objects via "
                        f"<{predicate}>, which is single-valued for every other "
                        "subject in this graph."
                    ),
                    subject=str(subject),
                    predicate=str(predicate),
                    values=sorted(str(obj) for obj in iri_objects),
                )
            )

    coreference: dict[tuple[URIRef, URIRef], set[URIRef]] = {}
    for (subject, predicate), objects in iri_groups.items():
        if predicate not in functional_ish or not _in_fact_scope(subject, namespaces):
            continue
        for obj in objects:
            coreference.setdefault((subject, obj), set()).add(predicate)
    for (subject, obj), predicates in sorted(
        coreference.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        if len(predicates) < 2:
            continue
        findings.append(
            FactsValidationFinding(
                kind=FactsValidationFindingKind.DEGENERATE_COREFERENCE,
                message=(
                    f"<{subject}> reaches <{obj}> through "
                    f"{len(predicates)} single-valued predicates "
                    f"({', '.join(sorted(str(p) for p in predicates))}) — "
                    "distinct endpoints (e.g. range bounds) likely merged."
                ),
                subject=str(subject),
                predicate=", ".join(sorted(str(p) for p in predicates)),
                values=[str(obj)],
            )
        )

    shacl_evaluated: bool | None = None
    shacl_violations: list[ShaclViolation] = []
    if shapes_graph is not None and len(shapes_graph):
        violations = run_shacl(
            graph,
            shapes_graph,
            ontology_graph=ontology_graph,
            inference=shacl_inference,
            advanced=shacl_advanced,
            max_triples=shacl_max_triples,
        )
        shacl_evaluated = violations is not None
        shacl_violations = list(violations or [])
        # Filter on the violation, which still holds the focus as an RDF term.
        # Filtering the projected finding instead compared a stringified blank
        # node against namespace prefixes, so it matched nothing and every
        # blank-node violation was dropped from the report — including the ones
        # the repair pass had just acted on.
        findings.extend(
            violation.as_finding()
            for violation in shacl_violations
            if _violation_in_fact_scope(graph, violation.focus, namespaces)
        )

    findings.extend(
        _non_catalog_vocabulary_findings(
            graph, ontology_graph, namespaces, quantity_fallback_vocabulary
        )
    )

    findings.extend(_dangling_reference_findings(graph, namespaces))

    # Object-kind self-consistency: a predicate carrying IRI objects on some
    # subjects and literal objects on others ("worksFor <org>" here, "worksFor
    # 'Ministry of Justice'" there) is un-queryable by shape. Warning-only
    # telemetry — never a merge signature.
    for predicate in sorted(predicate_object_kinds, key=str):
        kinds = predicate_object_kinds[predicate]
        iri_count = kinds.get("iri", 0)
        literal_count = kinds.get("literal", 0)
        if iri_count and literal_count:
            findings.append(
                FactsValidationFinding(
                    kind=FactsValidationFindingKind.MIXED_OBJECT_KINDS,
                    severity="warning",
                    message=(
                        f"<{predicate}> is used with {iri_count} IRI object(s) "
                        f"and {literal_count} literal object(s) — the same "
                        "relation is asserted as a link on some subjects and "
                        "as a string on others, so no single query shape "
                        "matches both."
                    ),
                    predicate=str(predicate),
                )
            )

    return FactsValidationReport(
        findings=findings,
        shacl_evaluated=shacl_evaluated,
        shacl_violations=shacl_violations,
    )


_DANGLING_REFERENCE_REPORT_CAP = 20

# Identity/provenance bookkeeping: their objects are alias or lineage handles
# that legitimately carry no description of their own.
_DANGLING_EXEMPT_PREDICATES = frozenset({OWL.sameAs, PROV.wasDerivedFrom})


def _dangling_reference_findings(
    graph: RDFGraph,
    fact_namespaces: list[str],
) -> list[FactsValidationFinding]:
    """Warning telemetry: fact-namespace objects that are never described.

    A fact-namespace IRI referenced as an object but never appearing as a
    subject and carrying no ``rdf:type``/``rdfs:label`` is a phantom node —
    usually a hallucinated or renamed reference. Warning severity only:
    it never drives un-merge.
    """
    subjects: set[URIRef] = {
        subject for subject in graph.subjects() if isinstance(subject, URIRef)
    }
    dangling: dict[URIRef, set[URIRef]] = {}
    for subject, predicate, obj in graph:
        if predicate == RDF.type or not isinstance(obj, URIRef):
            continue
        if predicate in _DANGLING_EXEMPT_PREDICATES:
            continue
        if obj in subjects:
            continue
        if not any(str(obj).startswith(ns) for ns in fact_namespaces):
            continue
        if not isinstance(predicate, URIRef):
            continue
        dangling.setdefault(obj, set()).add(predicate)

    findings: list[FactsValidationFinding] = []
    for obj in sorted(dangling, key=str)[:_DANGLING_REFERENCE_REPORT_CAP]:
        predicates = sorted(str(p) for p in dangling[obj])
        findings.append(
            FactsValidationFinding(
                kind=FactsValidationFindingKind.DANGLING_REFERENCE,
                severity="warning",
                message=(
                    f"<{obj}> is referenced via {', '.join(predicates)} but is "
                    "never described (no triples as subject, no rdf:type or "
                    "rdfs:label) — likely a hallucinated or renamed node."
                ),
                subject=str(obj),
                predicate=", ".join(predicates),
            )
        )
    remainder = len(dangling) - _DANGLING_REFERENCE_REPORT_CAP
    if remainder > 0:
        logger.info(
            "Dangling-reference report capped at %s; %s more not reported",
            _DANGLING_REFERENCE_REPORT_CAP,
            remainder,
        )
    return findings
