"""Deterministic per-unit findings on a rendered facts graph.

Only unresolved, machine-verified issues go back to the LLM, as MANDATORY
fix instructions that must be resolved by rewriting — never by deleting the
statement.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from rdflib import RDF, RDFS, SKOS, Literal, URIRef

from ontocast.onto.model import (
    FactsUnitFinding,
    FactsUnitFindingKind,
)
from ontocast.onto.rdfgraph import (
    RDFGraph,
    RejectedLiteralTriple,
)
from ontocast.tool.agg.signatures import canonical_literal, harvest_max_one_predicates
from ontocast.tool.facts_validation.terms import (
    _FORBIDDEN_NAMESPACES,
    _STANDARD_NAMESPACES,
    ValidationPolicy,
    _alias_candidates,
    _declared_domains,
    _described_classes,
    _local_name,
    _namespace_of,
    _resolve_type_literal,
    _superclass_closure,
    _vocabulary_role_subset,
    collect_catalog_terms,
    collect_declared_namespaces,
    expand_vocabulary_terms,
)
from ontocast.util.numeric_inventory import canonical_number, missing_numeric_mentions

logger = logging.getLogger(__name__)


def domain_violation_findings(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
) -> list[FactsUnitFinding]:
    """Report subjects whose asserted type contradicts a predicate's domain.

    Asserting a triple whose predicate declares an ``rdfs:domain`` *entails*
    that the subject belongs to that domain, so an untyped subject is never a
    violation -- the type is simply left to inference. It becomes one when the
    subject carries an asserted type that is unrelated to the declared domain:
    inference then adds the domain class on top of an incompatible one, and
    the contradiction surfaces later as a confusing failure somewhere else
    (SHACL reporting a missing property on a class the graph never meant to
    assert) rather than at the triple that caused it.

    Conservative by construction, since a false accusation costs a render pass.
    A subject is reported only when it has at least one asserted type and every
    asserted type is *unrelated* to every declared domain -- neither a subtype
    nor a supertype of it, following ``rdfs:subClassOf`` and
    ``owl:equivalentClass`` intersections in both directions. Typing a subject
    with a supertype of the domain (``sosa:Observation`` where the domain is
    ``obs:QuantitativeObservation``) is consistent: inference specializes it,
    it contradicts nothing, and flagging it would bury the real violations.

    Args:
        graph: Rendered facts graph for one unit.
        ontology_graph: Ontology context the renderer was given.

    Returns:
        list: One mandatory finding per offending (subject, predicate) pair,
        ordered by subject then predicate.
    """
    if ontology_graph is None or not len(ontology_graph):
        return []
    domains = _declared_domains(ontology_graph)
    if not domains:
        return []
    described = _described_classes(ontology_graph)

    closures: dict[URIRef, set[URIRef]] = {}
    findings: list[FactsUnitFinding] = []
    reported: set[tuple[str, str]] = set()

    for subject, predicate, _ in sorted(graph, key=lambda t: (str(t[0]), str(t[1]))):
        declared = domains.get(predicate)
        if declared is None or not isinstance(subject, URIRef):
            continue
        # Only domains the context places in a hierarchy can be argued about.
        declared = {value for value in declared if value in described}
        if not declared:
            continue
        asserted = {
            value
            for value in graph.objects(subject, RDF.type)
            if isinstance(value, URIRef)
        }
        if not asserted or not asserted <= described:
            continue

        def closure(class_iri: URIRef) -> set[URIRef]:
            if class_iri not in closures:
                closures[class_iri] = _superclass_closure(class_iri, ontology_graph)
            return closures[class_iri]

        # Compatible in either direction: the asserted type specializes a
        # declared domain, or a declared domain specializes the asserted type.
        domain_closure = set().union(*(closure(value) for value in declared))
        if any(
            closure(asserted_type) & declared or asserted_type in domain_closure
            for asserted_type in asserted
        ):
            continue
        key = (str(subject), str(predicate))
        if key in reported:
            continue
        reported.add(key)
        expected = ", ".join(f"<{value}>" for value in sorted(declared, key=str))
        actual = ", ".join(f"<{value}>" for value in sorted(asserted, key=str))
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.DOMAIN_VIOLATION,
                message=(
                    f"<{subject}> is typed {actual} but carries <{predicate}>, "
                    f"whose rdfs:domain is {expected}. Either type the subject "
                    "as the declared domain, or use the property that fits the "
                    "type it has."
                ),
                subject=str(subject),
                predicate=str(predicate),
                suggestions=sorted(str(value) for value in declared),
            )
        )
    return findings


def _closed_range_suggestions(
    rejected: RejectedLiteralTriple, ontology_graph: RDFGraph | None
) -> list[str]:
    """Suggest named individuals of the expected range matching the literal."""
    if ontology_graph is None or not rejected.expected_range:
        return []
    range_ref = URIRef(rejected.expected_range)
    token = rejected.object_lexical.strip()
    suggestions: list[str] = []
    for individual in ontology_graph.subjects(RDF.type, range_ref):
        if not isinstance(individual, URIRef):
            continue
        surfaces = {
            str(value)
            for predicate in (RDFS.label, SKOS.notation, SKOS.altLabel)
            for value in ontology_graph.objects(individual, predicate)
        }
        surfaces.add(_local_name(str(individual)))
        # Character-for-character only: the facts prompt instructs that a
        # lowercase symbol and its uppercase variant denote DIFFERENT
        # individuals, so the suggester must not propose `unit:M` for "m".
        if token in surfaces:
            suggestions.append(str(individual))
    return sorted(suggestions)[:3]


def _label_only_number_findings(
    graph: RDFGraph,
    *,
    unit_properties: set[str],
    numeric_value_properties: set[str],
    fact_namespaces: Sequence[str],
) -> list[FactsUnitFinding]:
    """Flag value nodes whose only numeric content sits in a label.

    A node carrying a unit property but no numeric literal on any
    non-annotation property, while its label or comment contains a number, is
    a measurement the renderer described instead of structuring — invisible to
    every query and to SHACL alike. Driven entirely by the configured quantity
    vocabulary (the unit-role property identifies value nodes), so no domain
    terms are hardcoded.
    """
    if not unit_properties:
        return []
    annotation_predicates = {RDFS.label, RDFS.comment, SKOS.prefLabel, SKOS.altLabel}
    findings: list[FactsUnitFinding] = []
    subjects = {
        subject
        for subject, predicate, _ in graph
        if isinstance(subject, URIRef)
        and str(predicate) in unit_properties
        and any(str(subject).startswith(ns) for ns in fact_namespaces)
    }
    for subject in sorted(subjects, key=str):
        has_numeric_literal = False
        label_numbers: set[str] = set()
        label_text = ""
        for _, predicate, obj in graph.triples((subject, None, None)):
            if not isinstance(obj, Literal):
                continue
            if predicate in annotation_predicates:
                for match in _LABEL_NUMBER_PATTERN.finditer(str(obj)):
                    canonical = canonical_number(match.group(1))
                    if canonical is not None:
                        label_numbers.add(canonical)
                        if not label_text:
                            label_text = str(obj)
                continue
            canonical = canonical_number(str(obj).strip())
            if canonical is not None:
                has_numeric_literal = True
                break
        if has_numeric_literal or not label_numbers:
            continue
        numbers = ", ".join(sorted(label_numbers, key=lambda v: (len(v), v)))
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.LABEL_ONLY_NUMBER,
                message=(
                    f"<{subject}> carries a unit but no numeric literal on any "
                    f"property; its label “{label_text}” holds the "
                    f"number(s) {numbers} as prose. Extract each number into "
                    "the appropriate numeric property as a typed literal "
                    "(verbatim value, source unit), keeping the label. A "
                    "measured value that exists only inside a label is "
                    "invisible to queries."
                ),
                subject=str(subject),
                value=numbers,
                suggestions=sorted(numeric_value_properties),
            )
        )
    return findings


_LABEL_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![\w])"
)


def _scalar_as_bounds_findings(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    fact_namespaces: Sequence[str],
) -> list[FactsUnitFinding]:
    """Flag one numeric value duplicated across single-valued numeric predicates.

    An exact scalar written into two distinct schema-constrained (functional /
    max-1) numeric predicates of one node — the classic case being equal lower
    and upper bounds — encodes a single measurement as if it carried epistemic
    structure it does not have. Fully generic: predicates come from the
    ontology's own functional/cardinality declarations, values compare via
    :func:`canonical_literal`.
    """
    constrained = harvest_max_one_predicates(ontology_graph)
    if not constrained:
        return []
    findings: list[FactsUnitFinding] = []
    by_subject: dict[URIRef, dict[URIRef, set[str]]] = {}
    for subject, predicate, obj in graph:
        if not isinstance(subject, URIRef) or not isinstance(obj, Literal):
            continue
        if not isinstance(predicate, URIRef) or predicate not in constrained:
            continue
        if not any(str(subject).startswith(ns) for ns in fact_namespaces):
            continue
        canonical = canonical_literal(obj)
        if canonical is None or canonical[1] != "numeric":
            continue
        by_subject.setdefault(subject, {}).setdefault(predicate, set()).add(
            canonical[0]
        )
    for subject, per_predicate in by_subject.items():
        value_to_predicates: dict[str, list[URIRef]] = {}
        for predicate, values in per_predicate.items():
            for value in values:
                value_to_predicates.setdefault(value, []).append(predicate)
        for value, predicates in value_to_predicates.items():
            if len(predicates) < 2:
                continue
            predicate_list = ", ".join(f"<{p}>" for p in sorted(map(str, predicates)))
            findings.append(
                FactsUnitFinding(
                    kind=FactsUnitFindingKind.SCALAR_AS_BOUNDS,
                    message=(
                        f"<{subject}> carries the same numeric value {value} on "
                        f"multiple single-valued numeric properties "
                        f"({predicate_list}). This encodes one exact scalar as "
                        "if it had epistemic structure. Record the value ONCE, "
                        "on the property the ontology documents as carrying "
                        "the plain/central value for this class (see its "
                        "definitions and scope notes), and remove it from the "
                        "other properties."
                    ),
                    subject=str(subject),
                    value=value,
                )
            )
    return findings


def domain_vocabulary_share(
    graph: RDFGraph,
    catalog_terms: set[str],
    fact_namespaces: Sequence[str],
) -> tuple[int, int]:
    """Count distinct schema terms drawn from the catalog, and the total.

    Schema position only -- predicates and ``rdf:type`` objects. Instances in
    the fact namespaces are excluded (they are supposed to be minted, not
    looked up), and so are the RDF/RDFS/OWL/XSD/SKOS/DC/PROV plumbing
    namespaces, which every graph uses regardless of which catalog it was
    given and would otherwise float the ratio for free. Generic *content*
    vocabularies such as schema.org deliberately stay in the denominator:
    reaching for them instead of the catalog is exactly what this measures.

    Args:
        graph: The rendered unit graph.
        catalog_terms: Terms declared by the unit's ontology context.
        fact_namespaces: Namespaces holding minted instances.

    Returns:
        tuple[int, int]: ``(from_catalog, total)`` over distinct terms.
    """
    used: set[str] = set()
    from_catalog: set[str] = set()
    for _, predicate, obj in graph:
        candidates = [predicate]
        if predicate == RDF.type:
            candidates.append(obj)
        for term in candidates:
            if not isinstance(term, URIRef):
                continue
            text = str(term)
            if any(text.startswith(ns) for ns in fact_namespaces):
                continue
            if text.startswith(_STANDARD_NAMESPACES):
                continue
            used.add(text)
            if text in catalog_terms:
                from_catalog.add(text)
    return len(from_catalog), len(used)


def _domain_adherence_findings(
    graph: RDFGraph,
    catalog_terms: set[str],
    fact_namespaces: Sequence[str],
    min_share: float,
) -> list[FactsUnitFinding]:
    """Flag a render that barely used the vocabulary it was handed.

    Every other finding here judges one triple. This one judges the render as a
    whole, because the failure it catches is invisible term by term:
    substituting a generic vocabulary for the catalog produces triples that are
    individually well-formed, pass ``UNKNOWN_TERM`` (standard namespaces are
    exempt by default), and satisfy every shape -- their subjects are in no
    shape's target class. The graph looks extracted and answers nothing.

    A share rather than an all-or-nothing test, because a graph picks up a
    catalog term or two by accident (a unit class, a quantity wrapper) while
    still expressing the substance in generic terms.

    Silent when the unit has no catalog to adhere to, so a deliberately
    catalog-free deployment is not spammed, and when ``min_share`` is 0.

    Args:
        graph: The rendered unit graph.
        catalog_terms: Terms declared by the unit's ontology context.
        fact_namespaces: Namespaces holding minted instances.
        min_share: Floor on the catalog share, in [0, 1]. 0 disables.

    Returns:
        list[FactsUnitFinding]: At most one finding.
    """
    if not catalog_terms or not len(graph) or min_share <= 0:
        return []
    from_catalog, total = domain_vocabulary_share(graph, catalog_terms, fact_namespaces)
    if not total or from_catalog / total >= min_share:
        return []
    return [
        FactsUnitFinding(
            kind=FactsUnitFindingKind.DOMAIN_ADHERENCE,
            message=(
                f"Only {from_catalog} of {total} schema terms in this render "
                "come from the ontology you were given; the rest are generic "
                "vocabulary. Re-express the same facts with terms from the "
                "ONTOLOGY section wherever one fits -- a generic substitute is "
                "validated by no shape and reachable by no query written "
                "against the catalog. Keep the statements and their values; "
                "change the terms."
            ),
        )
    ]


def collect_unit_findings(
    *,
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    quarantined: list[RejectedLiteralTriple],
    extraction_text: str,
    fact_namespaces: list[str],
    coverage_limit: int = 30,
    policy: ValidationPolicy | None = None,
) -> list[FactsUnitFinding]:
    """Assemble all deterministic findings for one rendered unit graph.

    Mandatory: quarantined literals (with closed-range individual
    suggestions), forbidden-namespace terms (``example.org``), doc-namespace
    predicates, unresolved catalog near-misses, predicates asserted on a
    subject whose type contradicts their ``rdfs:domain``, and value nodes
    whose only numeric content sits in a label. Advisory-strong: numeric
    mentions of the source text absent from the graph — the renderer decides
    per item whether each is an extractable quantity or an artifact.

    The policy's exempt terms (the sanctioned fallback vocabulary the facts
    prompt itself names, plus code predicates) never raise UNKNOWN_TERM:
    flagging the vocabulary the prompt recommends produced mandatory findings
    that repair renders obeyed by deleting correct data.
    """
    policy = policy or ValidationPolicy()
    findings: list[FactsUnitFinding] = []
    standard_namespaces = policy.standard_namespaces()
    fallback_terms = policy.exempt_terms(graph, ontology_graph)
    quantity_fallback_vocabulary = policy.quantity_fallback_vocabulary

    for rejected in quarantined:
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.QUARANTINED_LITERAL,
                message=(
                    f"Triple excluded ({rejected.reason}): the object of "
                    f"<{rejected.predicate}> must be an IRI/valid literal, got "
                    f"'{rejected.object_lexical}'."
                ),
                subject=rejected.subject,
                predicate=rejected.predicate,
                value=rejected.object_lexical,
                suggestions=_closed_range_suggestions(rejected, ontology_graph),
            )
        )

    catalog_terms = collect_catalog_terms(ontology_graph)
    declared_namespaces = collect_declared_namespaces(ontology_graph)
    normalized_fact_namespaces = [ns for ns in fact_namespaces if ns]

    prefix_map = {
        prefix: str(namespace) for prefix, namespace in graph.namespaces() if prefix
    }
    flagged_terms: set[str] = set()
    for subject, predicate, obj in graph:
        if predicate == RDF.type and isinstance(obj, Literal):
            lexical = str(obj).strip()
            if lexical in flagged_terms:
                continue
            flagged_terms.add(lexical)
            resolved = _resolve_type_literal(lexical, prefix_map)
            findings.append(
                FactsUnitFinding(
                    kind=FactsUnitFindingKind.LITERAL_TYPE_OBJECT,
                    message=(
                        f"rdf:type object '{lexical}' is a string literal, not "
                        "an IRI; assert the type as a catalog class IRI "
                        "(`a prefix:Class`), never as a quoted string."
                    ),
                    subject=str(subject),
                    value=lexical,
                    suggestions=[resolved]
                    if resolved and resolved in catalog_terms
                    else [],
                )
            )
            continue
        for position, term in (("predicate", predicate), ("type", obj)):
            if not isinstance(term, URIRef):
                continue
            if position == "type" and predicate != RDF.type:
                continue
            text = str(term)
            if text in flagged_terms:
                continue
            if text.startswith(_FORBIDDEN_NAMESPACES):
                flagged_terms.add(text)
                findings.append(
                    FactsUnitFinding(
                        kind=FactsUnitFindingKind.UNKNOWN_TERM,
                        message=(
                            f"<{text}> uses the example.org placeholder namespace; "
                            "replace it with a catalog term or express the "
                            "statement with catalog/standard vocabulary."
                        ),
                        predicate=text,
                        suggestions=_alias_candidates(
                            term,
                            graph,
                            catalog_terms,
                            ontology_graph=ontology_graph,
                            position=position,
                        )
                        if catalog_terms
                        else [],
                    )
                )
                continue
            if any(text.startswith(ns) for ns in normalized_fact_namespaces):
                flagged_terms.add(text)
                role_message = (
                    f"Predicate <{text}> is minted in the facts/document "
                    "namespace; facts namespaces hold instances only — "
                    "use a catalog or standard-vocabulary property."
                    if position == "predicate"
                    else f"rdf:type object <{text}> is a class minted in the "
                    "facts/document namespace; facts namespaces hold instances, "
                    "not classes — type the instance with a catalog or "
                    "standard-vocabulary class."
                )
                findings.append(
                    FactsUnitFinding(
                        kind=FactsUnitFindingKind.UNKNOWN_TERM,
                        message=role_message,
                        predicate=text,
                        suggestions=_alias_candidates(
                            term,
                            graph,
                            catalog_terms,
                            ontology_graph=ontology_graph,
                            position=position,
                        )
                        if catalog_terms
                        else [],
                    )
                )
                continue
            namespace = _namespace_of(text)
            if (
                catalog_terms
                and namespace in declared_namespaces
                and text not in catalog_terms
                and text not in fallback_terms
                and not namespace.startswith(standard_namespaces)
            ):
                flagged_terms.add(text)
                findings.append(
                    FactsUnitFinding(
                        kind=FactsUnitFindingKind.UNKNOWN_TERM,
                        message=(
                            f"<{text}> does not exist in its ontology; rewrite "
                            "the term IN PLACE to the closest correct term from "
                            "the ontology chapter (or a suggested candidate), "
                            "keeping the statement and its value. Do NOT delete "
                            "the statement."
                        ),
                        predicate=text,
                        suggestions=_alias_candidates(
                            term,
                            graph,
                            catalog_terms,
                            ontology_graph=ontology_graph,
                            position=position,
                        ),
                    )
                )

    findings.extend(
        _scalar_as_bounds_findings(graph, ontology_graph, normalized_fact_namespaces)
    )
    findings.extend(domain_violation_findings(graph, ontology_graph))
    findings.extend(
        _domain_adherence_findings(
            graph,
            catalog_terms,
            normalized_fact_namespaces,
            policy.domain_adherence_min_share,
        )
    )
    findings.extend(
        _label_only_number_findings(
            graph,
            unit_properties=expand_vocabulary_terms(
                _vocabulary_role_subset(quantity_fallback_vocabulary, "unit"),
                graph,
                ontology_graph,
            ),
            numeric_value_properties=expand_vocabulary_terms(
                _vocabulary_role_subset(quantity_fallback_vocabulary, "numeric_value"),
                graph,
                ontology_graph,
            ),
            fact_namespaces=normalized_fact_namespaces,
        )
    )

    missing = missing_numeric_mentions(
        extraction_text,
        graph,
        ignore_identifier_fragments=policy.numeric_identifier_guard,
        limit=coverage_limit,
    )
    if missing:
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.NUMERIC_COVERAGE,
                mandatory=False,
                message=(
                    "These numbers appear in the source text but not in the "
                    "graph. For each: extract it as a typed literal on an "
                    "appropriate node (verbatim value and source unit — never "
                    "convert units) if it is a factual quantity, or ignore it "
                    "if it is a page/citation/figure artifact: " + ", ".join(missing)
                ),
                value=", ".join(missing),
            )
        )

    return findings
