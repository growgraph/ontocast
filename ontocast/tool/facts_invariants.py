"""Deterministic normalization, repair, and findings for rendered facts.

The renderer LLM is treated as a transcriber, not a guarantor: every check
here detects or repairs a concrete violation in code, and only unresolved
items go back to the LLM as mandatory fix instructions. All mechanisms are
schema-driven and domain-agnostic — alias tables and known-term sets are
derived from the ontology context at runtime, never hardcoded per
vocabulary.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal as TypingLiteral

from pydantic import BaseModel, Field
from rdflib import OWL, RDF, RDFS, SKOS, Literal, URIRef
from rdflib.namespace import DCTERMS, PROV, SH, XSD
from rdflib.term import Node

from ontocast.onto.model import (
    FactsGateRepairKind,
    FactsUnitFinding,
    FactsUnitFindingKind,
    FactsValidationFinding,
    FactsValidationFindingKind,
    GraphRepairRecord,
)
from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple, copy_triples
from ontocast.tool.agg.signatures import canonical_literal, harvest_max_one_predicates
from ontocast.util.numeric_inventory import canonical_number, missing_numeric_mentions

logger = logging.getLogger(__name__)

_FORBIDDEN_NAMESPACES = ("http://example.org/", "https://example.org/")

_NUMERIC_RANGE_DATATYPES = {
    XSD.decimal,
    XSD.integer,
    XSD.float,
    XSD.double,
    XSD.nonNegativeInteger,
    XSD.positiveInteger,
}

# Meta-vocabularies: the RDF/OWL substrate plus the annotation and provenance
# terms every facts graph carries regardless of catalog. Exempting these from
# UNKNOWN_TERM keeps the signal readable -- flagging rdfs:label would bury it.
#
# Domain vocabularies do NOT belong here. SOSA/SSN (sensor observations),
# CSVW (tabular metadata), FOAF and schema.org (people, organizations,
# creative works) model a subject area; a catalog that does not declare them
# should hear about it. They are exempted by configuration
# (FACTS_ADDITIONAL_STANDARD_NAMESPACES) when a deployment wants them, not by
# being compiled in.
_STANDARD_NAMESPACES = (
    str(RDF),
    str(RDFS),
    str(OWL),
    str(XSD),
    str(SKOS),
    str(DCTERMS),
    "http://purl.org/dc/elements/1.1/",
    "http://www.w3.org/ns/prov#",
    "http://www.w3.org/XML/1998/namespace",
)

_LOCAL_SPLIT = re.compile(r"[#/]")

# Bounds on the advisory alias-candidate list shown to the renderer. Not
# applied to the graph, so these trim prompt noise rather than gate a rewrite.
_ALIAS_MIN_SCORE = 0.5
_ALIAS_MAX_SUGGESTIONS = 3


def _namespace_of(iri: str) -> str:
    """Split an IRI into its namespace part (up to the last '#' or '/')."""
    for separator in ("#", "/"):
        head, sep, local = iri.rpartition(separator)
        if sep and local:
            return head + sep
    return iri


def _local_name(iri: str) -> str:
    for separator in ("#", "/"):
        head, sep, local = iri.rpartition(separator)
        if sep and local:
            return local
    return iri


def _name_tokens(local: str) -> set[str]:
    tokens = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local)
    tokens = re.sub(r"[_\-]", " ", tokens)
    return {token.lower() for token in tokens.split() if token}


def collect_catalog_terms(ontology_graph: RDFGraph | None) -> set[str]:
    """All IRIs appearing anywhere in the ontology context."""
    terms: set[str] = set()
    if ontology_graph is None:
        return terms
    for triple in ontology_graph:
        for term in triple:
            if isinstance(term, URIRef):
                terms.add(str(term))
    return terms


def collect_catalog_namespaces(ontology_graph: RDFGraph | None) -> set[str]:
    """Namespaces of every IRI appearing in the ontology context."""
    return {_namespace_of(term) for term in collect_catalog_terms(ontology_graph)}


def normalize_literals_against_schema(
    graph: RDFGraph, ontology_graph: RDFGraph | None
) -> int:
    """Retype untyped numeric literals whose predicate declares a numeric range.

    Fixes the ``qudt:numericValue 230`` vs ``"230"^^xsd:decimal`` drift at
    parse time: when the schema says the range is numeric and the lexical form
    parses as a number, the literal is rewritten with the declared datatype.

    Returns:
        Number of retyped literals.
    """
    if ontology_graph is None:
        return 0
    declared_ranges: dict[URIRef, URIRef] = {}
    for predicate, range_iri in ontology_graph.subject_objects(RDFS.range):
        if (
            isinstance(predicate, URIRef)
            and isinstance(range_iri, URIRef)
            and range_iri in _NUMERIC_RANGE_DATATYPES
        ):
            declared_ranges[predicate] = range_iri

    if not declared_ranges:
        return 0

    replacements: list[tuple[tuple, tuple]] = []
    for subject, predicate, obj in graph:
        if not isinstance(obj, Literal) or not isinstance(predicate, URIRef):
            continue
        target_datatype = declared_ranges.get(predicate)
        if target_datatype is None or obj.datatype == target_datatype:
            continue
        if obj.datatype is not None and obj.datatype not in _NUMERIC_RANGE_DATATYPES:
            continue
        if canonical_number(str(obj).strip()) is None:
            continue
        replacements.append(
            (
                (subject, predicate, obj),
                (
                    subject,
                    predicate,
                    Literal(str(obj).strip(), datatype=target_datatype),
                ),
            )
        )

    for old, new in replacements:
        graph.remove(old)
        graph.add(new)
    return len(replacements)


_COMPACT_IRI = re.compile(r"^([A-Za-z][\w.-]*):([^\s/][^\s]*)$")


def _resolve_type_literal(lexical: str, prefix_map: dict[str, str]) -> str | None:
    """Resolve a literal ``rdf:type`` object to a full IRI, when unambiguous.

    Accepts absolute IRIs and compact IRIs whose prefix is bound in the graph.
    Returns None when the lexical form cannot be resolved deterministically.
    """
    if lexical.startswith(("http://", "https://", "urn:")) and " " not in lexical:
        return lexical
    match = _COMPACT_IRI.match(lexical)
    if match is None:
        return None
    namespace = prefix_map.get(match.group(1))
    if not namespace:
        return None
    return f"{namespace}{match.group(2)}"


def repair_literal_type_objects(
    graph: RDFGraph,
) -> tuple[int, list[FactsUnitFinding], list[GraphRepairRecord]]:
    """Coerce literal ``rdf:type`` objects into IRIs.

    The renderer sometimes emits ``a "prefix:Class"^^xsd:string`` instead of
    ``a prefix:Class`` (JSON-LD bare-string type values parse the same way).
    A literal-typed node is invisible to SPARQL class queries, reasoning, and
    the aggregator's URI minting/entity matching, all of which guard on
    ``isinstance(obj, URIRef)``. Absolute IRIs and compact IRIs bound in the
    graph are rewritten deterministically; unresolvable forms become MANDATORY
    findings.

    Returns:
        Tuple of (number of rewritten triples, unresolved findings,
        applied-repair records).
    """
    prefix_map = {
        prefix: str(namespace) for prefix, namespace in graph.namespaces() if prefix
    }
    rewritten = 0
    findings: list[FactsUnitFinding] = []
    applied: list[GraphRepairRecord] = []
    for subject, predicate, obj in list(graph.triples((None, RDF.type, None))):
        if not isinstance(obj, Literal):
            continue
        lexical = str(obj).strip()
        resolved = _resolve_type_literal(lexical, prefix_map)
        if resolved is not None:
            graph.remove((subject, predicate, obj))
            graph.add((subject, RDF.type, URIRef(resolved)))
            rewritten += 1
            applied.append(
                GraphRepairRecord(
                    kind=FactsUnitFindingKind.LITERAL_TYPE_OBJECT,
                    source=lexical,
                    target=resolved,
                )
            )
            logger.info(
                "Repaired literal rdf:type object %r -> <%s>", lexical, resolved
            )
            continue
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.LITERAL_TYPE_OBJECT,
                message=(
                    f"rdf:type object '{lexical}' is a string literal, not an "
                    "IRI; assert the type as a catalog class IRI "
                    "(`a prefix:Class`), never as a quoted string."
                ),
                subject=str(subject),
                value=lexical,
            )
        )
    return rewritten, findings, applied


def _alias_candidates(
    alias: URIRef,
    graph: RDFGraph,
    catalog_terms: set[str],
) -> list[str]:
    """Rank replacement candidates for a near-miss predicate.

    Candidate pool: catalog terms sharing the alias namespace, plus predicates
    of that namespace the graph itself uses on >= 2 subjects (the renderer's
    own dominant usage defines the alias target — this is how
    ``qudt:value`` resolves to ``qudt:numericValue`` even when the snapshot
    does not spell the property out).
    """
    namespace = _namespace_of(str(alias))
    pool: set[str] = {
        term for term in catalog_terms if _namespace_of(term) == namespace
    }
    predicate_subjects: dict[str, set[str]] = {}
    for subject, predicate, _ in graph:
        text = str(predicate)
        if text != str(alias) and _namespace_of(text) == namespace:
            predicate_subjects.setdefault(text, set()).add(str(subject))
    pool.update(
        predicate
        for predicate, subjects in predicate_subjects.items()
        if len(subjects) >= 2
    )
    pool.discard(str(alias))

    alias_local = _local_name(str(alias))
    alias_tokens = _name_tokens(alias_local)
    scored: list[tuple[float, str]] = []
    for candidate in pool:
        candidate_local = _local_name(candidate)
        candidate_tokens = _name_tokens(candidate_local)
        if alias_tokens and (
            alias_tokens <= candidate_tokens or candidate_tokens <= alias_tokens
        ):
            score = 1.0
        else:
            score = SequenceMatcher(
                None, alias_local.lower(), candidate_local.lower()
            ).ratio()
        scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    # Suggestions only: these are shown to the renderer as candidates, never
    # applied. The 0.5 floor drops noise and the top-3 cap keeps the prompt
    # readable -- both are presentation bounds on an advisory list, so
    # truncation here cannot silently change the graph.
    qualifying = [candidate for score, candidate in scored if score >= _ALIAS_MIN_SCORE]
    return qualifying[:_ALIAS_MAX_SUGGESTIONS]


def repair_property_aliases(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    *,
    min_ratio: float = 0.85,
) -> tuple[int, list[FactsUnitFinding], list[GraphRepairRecord]]:
    """Rewrite near-miss predicates in catalog namespaces; report ambiguity.

    A predicate whose namespace belongs to the ontology context but which is
    not itself a catalog term is a near-miss (``qqval:lowerBound`` for
    ``qqval:hasLowerBound``). When exactly one candidate scores above
    ``min_ratio`` (token containment counts as 1.0) the rewrite is applied
    deterministically; otherwise a mandatory finding carries the top
    suggestions.

    Returns:
        Tuple of (number of rewritten triples, unresolved findings,
        applied-repair records).
    """
    catalog_terms = collect_catalog_terms(ontology_graph)
    if not catalog_terms:
        return 0, [], []
    catalog_namespaces = {_namespace_of(term) for term in catalog_terms}

    findings: list[FactsUnitFinding] = []
    applied: list[GraphRepairRecord] = []
    rewritten = 0
    predicates = {
        predicate
        for predicate in graph.predicates()
        if isinstance(predicate, URIRef)
        and str(predicate) not in catalog_terms
        and _namespace_of(str(predicate)) in catalog_namespaces
    }
    for alias in sorted(predicates, key=str):
        candidates = _alias_candidates(alias, graph, catalog_terms)
        strong = [
            candidate
            for candidate in candidates
            if _name_tokens(_local_name(str(alias)))
            and (
                _name_tokens(_local_name(str(alias)))
                <= _name_tokens(_local_name(candidate))
                or _name_tokens(_local_name(candidate))
                <= _name_tokens(_local_name(str(alias)))
                or SequenceMatcher(
                    None,
                    _local_name(str(alias)).lower(),
                    _local_name(candidate).lower(),
                ).ratio()
                >= min_ratio
            )
        ]
        if len(strong) == 1:
            replacement = URIRef(strong[0])
            alias_triples = 0
            for subject, predicate, obj in list(graph.triples((None, alias, None))):
                graph.remove((subject, predicate, obj))
                graph.add((subject, replacement, obj))
                rewritten += 1
                alias_triples += 1
            applied.append(
                GraphRepairRecord(
                    kind=FactsUnitFindingKind.PROPERTY_ALIAS,
                    source=str(alias),
                    target=str(replacement),
                    triple_count=alias_triples,
                )
            )
            logger.info("Repaired property alias %s -> %s", alias, replacement)
            continue
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.PROPERTY_ALIAS,
                message=(
                    f"Predicate <{alias}> is not defined in its ontology; "
                    "replace it with the correct catalog property."
                ),
                predicate=str(alias),
                suggestions=candidates,
            )
        )
    return rewritten, findings, applied


def _collect_linking_evidence(
    graph: RDFGraph,
    ontology_graph: RDFGraph,
    closure: Callable[[URIRef], set[URIRef]],
) -> list[tuple[set[URIRef], set[URIRef], URIRef]]:
    """One scan of the graph's IRI-object links, with type closures resolved.

    Collected once per :func:`resolve_code_literals` call so the per-literal
    fallback in :func:`_observed_linking_predicates` filters in memory instead
    of re-walking the graph and recomputing closures for every code literal.
    """
    evidence: list[tuple[set[URIRef], set[URIRef], URIRef]] = []
    subject_closures: dict[Node, set[URIRef]] = {}
    for subject, predicate, obj in graph:
        if not isinstance(predicate, URIRef) or not isinstance(obj, URIRef):
            continue
        if predicate == RDF.type:
            continue
        obj_types: set[URIRef] = set()
        for type_iri in ontology_graph.objects(obj, RDF.type):
            if isinstance(type_iri, URIRef):
                obj_types |= closure(type_iri)
        if not obj_types:
            continue
        if subject not in subject_closures:
            own_types: set[URIRef] = set()
            for type_iri in graph.objects(subject, RDF.type):
                if isinstance(type_iri, URIRef):
                    own_types |= closure(type_iri)
            subject_closures[subject] = own_types
        evidence.append((subject_closures[subject], obj_types, predicate))
    return evidence


def _observed_linking_predicates(
    evidence: Sequence[tuple[set[URIRef], set[URIRef], URIRef]],
    subject_types: set[URIRef],
    object_types: set[URIRef],
) -> list[URIRef]:
    """Predicates the graph already uses to link these two kinds of node.

    Evidence, not schema: used only where the schema declares no range, and
    only when the evidence is unambiguous (exactly one such predicate).
    """
    observed = {
        predicate
        for own_types, obj_types, predicate in evidence
        if obj_types & object_types and (not subject_types or own_types & subject_types)
    }
    return sorted(observed, key=str)


def resolve_code_literals(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    code_predicates: Sequence[str] = (),
) -> tuple[int, list[GraphRepairRecord]]:
    """Link nodes to the catalog individual whose code they already carry.

    A renderer that reads ``4-15 days`` often annotates the value node with the
    code it saw — ``qudt:ucumCode "d"`` — instead of the object property that
    points at the individual — ``qudt:unit unit:DAY``. The graph is well-formed,
    so no range check fires, but every query reading the object property gets
    an unbound result. The code came from the text and the individual is in the
    catalog, so the link is recoverable without asking the model again.

    Fully schema-driven, no vocabulary compiled in: the connecting property is
    whichever object property the ontology context declares with a range the
    resolved individual is typed as, and a domain the subject satisfies. If the
    schema offers several such properties, or none, nothing is added.

    Args:
        graph: Rendered facts graph, repaired in place.
        ontology_graph: Merged ontology context, read-only.
        code_predicates: Predicates carrying machine-resolvable codes.

    Returns:
        Tuple of (number of added triples, applied-repair records).
    """
    if ontology_graph is None or not code_predicates:
        return 0, []
    code_terms = [URIRef(predicate) for predicate in code_predicates]
    # Only the code predicates themselves resolve here: a label match is a
    # different, much weaker signal and belongs to the shapes-driven pass.
    code_index: dict[str, set[str]] = {}
    for predicate in code_terms:
        for subject, value in ontology_graph.subject_objects(predicate):
            if isinstance(subject, URIRef) and isinstance(value, Literal):
                text = str(value).strip()
                if text:
                    code_index.setdefault(text, set()).add(str(subject))
    if not code_index:
        return 0, []

    domains = _declared_domains(ontology_graph)
    ranges: dict[URIRef, set[URIRef]] = {}
    for predicate, _, range_iri in ontology_graph.triples((None, RDFS.range, None)):
        if isinstance(predicate, URIRef) and isinstance(range_iri, URIRef):
            ranges.setdefault(predicate, set()).add(range_iri)

    # Superclass closures repeat heavily across literals; memoise per call.
    closures: dict[URIRef, set[URIRef]] = {}

    def closure(class_iri: URIRef) -> set[URIRef]:
        if class_iri not in closures:
            closures[class_iri] = _superclass_closure(class_iri, ontology_graph)
        return closures[class_iri]

    # The usage-evidence scan is a full graph walk; build it lazily, once,
    # only if some literal actually needs the no-declared-range fallback.
    linking_evidence: list[tuple[set[URIRef], set[URIRef], URIRef]] | None = None

    added = 0
    records: list[GraphRepairRecord] = []
    for code_predicate in code_terms:
        for subject, value in list(graph.subject_objects(code_predicate)):
            if not isinstance(subject, URIRef) or not isinstance(value, Literal):
                continue
            resolved = resolve_unique_surface(code_index, str(value))
            if resolved is None:
                continue
            resolved_types: set[URIRef] = set()
            for type_iri in ontology_graph.objects(resolved, RDF.type):
                if isinstance(type_iri, URIRef):
                    resolved_types |= closure(type_iri)
            if not resolved_types:
                continue
            subject_types: set[URIRef] = set()
            for type_iri in graph.objects(subject, RDF.type):
                if isinstance(type_iri, URIRef):
                    subject_types |= closure(type_iri)

            candidates = [
                predicate
                for predicate, range_set in ranges.items()
                if range_set & resolved_types
                and (
                    predicate not in domains
                    or not domains[predicate]
                    or domains[predicate] & subject_types
                )
            ]
            if not candidates:
                # Vendored vocabulary projections often declare individuals and
                # their codes but no rdfs:range (the shipped QUDT unit subset is
                # one). Fall back to how the graph already links this kind of
                # subject to this kind of individual -- the same
                # induce-from-usage move the functional-predicate harvest makes.
                if linking_evidence is None:
                    linking_evidence = _collect_linking_evidence(
                        graph, ontology_graph, closure
                    )
                candidates = _observed_linking_predicates(
                    linking_evidence, subject_types, resolved_types
                )
            # Already linked, ambiguous, or unsupported by the schema.
            candidates = [
                predicate
                for predicate in candidates
                if (subject, predicate, None) not in graph
            ]
            if len(candidates) != 1:
                continue
            predicate = candidates[0]
            graph.add((subject, predicate, resolved))
            added += 1
            records.append(
                GraphRepairRecord(
                    kind=FactsGateRepairKind.CODE_RESOLVED,
                    source=f"{code_predicate} {value.n3()}",
                    target=f"{predicate} {resolved}",
                )
            )
            logger.info(
                "Resolved code %s on <%s> to <%s %s>",
                value.n3(),
                subject,
                predicate,
                resolved,
            )
    return added, records


def _superclass_closure(class_iri: URIRef, ontology_graph: RDFGraph) -> set[URIRef]:
    """Asserted supertypes of *class_iri*, including itself.

    Walks ``rdfs:subClassOf`` and steps through ``owl:equivalentClass``
    intersections, because a class defined only as an intersection
    (``LeadHalidePerovskite ≡ Perovskite ⊓ …``) has no asserted subClassOf edge
    to its genus and would otherwise look unrelated to it.
    """
    seen: set[URIRef] = set()
    frontier: list[Node] = [class_iri]
    while frontier:
        current = frontier.pop()
        if not isinstance(current, URIRef) or current in seen:
            continue
        seen.add(current)
        frontier.extend(ontology_graph.objects(current, RDFS.subClassOf))
        for equivalent in ontology_graph.objects(current, OWL.equivalentClass):
            for collection in ontology_graph.objects(equivalent, OWL.intersectionOf):
                frontier.extend(ontology_graph.items(collection))
    return seen


def _described_classes(ontology_graph: RDFGraph) -> set[URIRef]:
    """Classes whose position in the hierarchy the context actually states.

    A class named only as the object of an ``rdfs:domain`` or an ``rdf:type``
    carries no subclass edges here -- typically because it belongs to an
    imported vocabulary (SOSA, QUDT) that the context does not vendor, since
    ``owl:imports`` is not dereferenced. Its ancestors are unknown, so nothing
    can be concluded about whether it relates to another class.
    """
    described: set[URIRef] = set()
    for subject in ontology_graph.subjects(RDF.type, OWL.Class):
        if isinstance(subject, URIRef):
            described.add(subject)
    for subject in ontology_graph.subjects(RDFS.subClassOf, None):
        if isinstance(subject, URIRef):
            described.add(subject)
    return described


def _declared_domains(ontology_graph: RDFGraph) -> dict[URIRef, set[URIRef]]:
    """Predicate -> its declared ``rdfs:domain`` classes, named classes only."""
    domains: dict[URIRef, set[URIRef]] = {}
    for predicate, _, domain in ontology_graph.triples((None, RDFS.domain, None)):
        # Anonymous domains are class expressions (unions, restrictions); they
        # need a reasoner to decide membership, so they are out of scope here.
        if isinstance(predicate, URIRef) and isinstance(domain, URIRef):
            domains.setdefault(predicate, set()).add(domain)
    return domains


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


def collect_unit_findings(
    *,
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    quarantined: list[RejectedLiteralTriple],
    extraction_text: str,
    fact_namespaces: list[str],
    coverage_limit: int = 30,
    additional_standard_namespaces: Sequence[str] = (),
) -> list[FactsUnitFinding]:
    """Assemble all deterministic findings for one rendered unit graph.

    Mandatory: quarantined literals (with closed-range individual
    suggestions), forbidden-namespace terms (``example.org``), doc-namespace
    predicates, unresolved catalog near-misses, and predicates asserted on a
    subject whose type contradicts their ``rdfs:domain``. Advisory-strong:
    numeric mentions of the source text absent from the graph — the renderer
    decides per item whether each is an extractable quantity or an artifact.
    """
    findings: list[FactsUnitFinding] = []
    standard_namespaces = (
        *_STANDARD_NAMESPACES,
        *additional_standard_namespaces,
    )

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
    catalog_namespaces = {_namespace_of(term) for term in catalog_terms}
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
                        suggestions=_alias_candidates(term, graph, catalog_terms)
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
                        suggestions=_alias_candidates(term, graph, catalog_terms)
                        if catalog_terms
                        else [],
                    )
                )
                continue
            namespace = _namespace_of(text)
            if (
                catalog_terms
                and namespace in catalog_namespaces
                and text not in catalog_terms
                and not namespace.startswith(standard_namespaces)
            ):
                flagged_terms.add(text)
                findings.append(
                    FactsUnitFinding(
                        kind=FactsUnitFindingKind.UNKNOWN_TERM,
                        message=(
                            f"<{text}> does not exist in its ontology; use one of "
                            "the suggested catalog terms or the closest correct "
                            "term from the ontology chapter."
                        ),
                        predicate=text,
                        suggestions=_alias_candidates(term, graph, catalog_terms),
                    )
                )

    findings.extend(
        _scalar_as_bounds_findings(graph, ontology_graph, normalized_fact_namespaces)
    )
    findings.extend(domain_violation_findings(graph, ontology_graph))

    missing = missing_numeric_mentions(extraction_text, graph, limit=coverage_limit)
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


def summarize_conformance(
    findings: Sequence[FactsValidationFinding],
    *,
    shacl_evaluated: bool | None = None,
    repairs: Sequence[GraphRepairRecord] = (),
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

    Returns:
        ``conforms`` (None when SHACL did not run), counts by severity, by
        finding kind, by SHACL constraint component and shape, and the applied
        repair counts by kind.
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

    return {
        "shacl_evaluated": shacl_evaluated,
        "conforms": None if not shacl_evaluated else not shacl_findings,
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


def _in_fact_scope(subject: URIRef, fact_namespaces: list[str]) -> bool:
    if not fact_namespaces:
        return True
    text = str(subject)
    return any(text.startswith(namespace) for namespace in fact_namespaces if namespace)


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
    iri_groups: dict[tuple[URIRef, URIRef], set[URIRef]],
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


class ShaclViolation(BaseModel):
    """One SHACL validation result, in the form the repair pass needs.

    ``FactsValidationFinding`` is the reporting shape and deliberately flat;
    this keeps the RDF terms (focus node, path, offending value, constraint
    component) so a repair can act on them.
    """

    model_config = {"arbitrary_types_allowed": True}

    focus: Node | None = None
    path: URIRef | None = None
    value: Node | None = None
    component: URIRef | None = None
    source_shape: URIRef | None = None
    severity: TypingLiteral["error", "warning"] = "error"
    message: str = "SHACL constraint violated."

    def as_finding(self) -> FactsValidationFinding:
        """Project onto the reported finding shape."""
        return FactsValidationFinding(
            kind=FactsValidationFindingKind.SHACL,
            severity=self.severity,
            message=self.message,
            subject=str(self.focus) if self.focus is not None else "",
            predicate=str(self.path) if self.path is not None else "",
            values=[str(self.value)] if self.value is not None else [],
            component=str(self.component) if self.component is not None else "",
            source_shape=(
                str(self.source_shape) if self.source_shape is not None else ""
            ),
        )


# FactsValidationReport is declared above ShaclViolation and forward-references
# it; resolve the reference now that both exist.
FactsValidationReport.model_rebuild()


def run_shacl(
    graph: RDFGraph,
    shapes_graph: RDFGraph,
    *,
    ontology_graph: RDFGraph | None = None,
    inference: str = "rdfs",
    advanced: bool = True,
    max_triples: int = 0,
) -> list[ShaclViolation] | None:
    """Validate ``graph`` against ``shapes_graph``, returning the violations.

    Reaching here means shapes were found, so the caller expects validation to
    happen: a missing extra or a skipped run is reported at warning level, not
    debug. Silently returning "no violations" is indistinguishable from
    "conforms", so those cases return ``None``.

    The ontology context is mixed in (``ont_graph``) rather than left out. A
    facts graph states that a value uses ``unit:DAY``; that the individual *is*
    a ``qudt:Unit`` is stated only in the catalog. Validating the facts alone
    therefore fails every ``sh:class`` constraint pointing at a catalog
    individual — violations that describe the missing schema, not the data.

    RDFS inference is the default for the same reason. SHACL resolves class
    targets through ``rdfs:subClassOf`` on its own, but property paths carry no
    entailment: a shape on ``obs:hasResult`` does not see the
    ``life:hasStorageResult`` the renderer emitted, and reports the more
    specific statement as a missing one. Measured on the three-document matsci
    pilot: 268 violations at ``inference="none"`` against 232 with RDFS.

    Args:
        graph: Data graph to validate.
        shapes_graph: Shapes to validate against.
        ontology_graph: Schema mixed into the data graph for validation.
        inference: pyshacl pre-inference (``none`` / ``rdfs`` / ``owlrl``).
        advanced: Enable SHACL Advanced Features.
        max_triples: Skip validation above this graph size; 0 disables.

    Returns:
        Violations in report order, or ``None`` when validation did not run.
    """
    try:
        import pyshacl
    except ImportError:
        logger.warning(
            "SHACL shapes are configured but pyshacl is not installed; "
            "skipping SHACL validation. Install the extra: uv sync --extra shacl"
        )
        return None

    if max_triples and len(graph) > max_triples:
        logger.warning(
            "Skipping SHACL validation: %d triples exceeds "
            "FACTS_SHACL_MAX_TRIPLES=%d. The graph is unvalidated, not conformant.",
            len(graph),
            max_triples,
        )
        return None

    # pyshacl clones and mixes the data graph through plain rdflib graphs,
    # which cannot hold the RDF 1.2 triple terms an oxigraph-backed aggregated
    # graph carries (rdflib ``Graph.add`` asserts on them). Hand pyshacl a
    # sanitised copy; the dropped reification provenance carries no shape
    # targets, so validation loses nothing.
    data_graph = RDFGraph()
    copy_triples(graph, data_graph, origin="run_shacl")
    for prefix, namespace in graph.namespaces():
        data_graph.bind(prefix, namespace, override=True)

    conforms, results_graph, _ = pyshacl.validate(
        data_graph,
        shacl_graph=shapes_graph,
        ont_graph=(
            ontology_graph
            if ontology_graph is not None and len(ontology_graph)
            else None
        ),
        inference=inference,
        advanced=advanced,
        abort_on_first=False,
    )
    if conforms:
        return []

    violations: list[ShaclViolation] = []
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        severity_iri = results_graph.value(result, SH.resultSeverity)
        message = results_graph.value(result, SH.resultMessage)
        path = results_graph.value(result, SH.resultPath)
        component = results_graph.value(result, SH.sourceConstraintComponent)
        source_shape = results_graph.value(result, SH.sourceShape)
        violations.append(
            ShaclViolation(
                focus=results_graph.value(result, SH.focusNode),
                path=path if isinstance(path, URIRef) else None,
                value=results_graph.value(result, SH.value),
                component=component if isinstance(component, URIRef) else None,
                source_shape=(
                    source_shape if isinstance(source_shape, URIRef) else None
                ),
                severity=("error" if severity_iri == SH.Violation else "warning"),
                message=str(message) if message else "SHACL constraint violated.",
            )
        )
    return violations


def collect_shacl_shapes(
    ontology_graph: RDFGraph | None, shapes_dir: str | None
) -> RDFGraph | None:
    """Assemble the SHACL shapes graph for the validation gate.

    Sources: every ``.ttl`` file under ``shapes_dir`` (when configured), plus
    the ontology context itself when it already carries ``sh:NodeShape``
    declarations inline — the zero-config path for catalogs that ship shapes
    next to their schema.
    """
    shapes = RDFGraph()
    if shapes_dir:
        directory = Path(shapes_dir)
        if not directory.is_dir():
            # Configuring a shapes directory that does not exist must not read
            # as "validated cleanly": glob() on a missing path yields nothing.
            logger.warning(
                "FACTS_SHAPES_DIR points at %s, which is not a directory; "
                "no SHACL shapes loaded",
                shapes_dir,
            )
        else:
            files = sorted(directory.glob("**/*.ttl"))
            if not files:
                logger.warning(
                    "FACTS_SHAPES_DIR %s contains no .ttl shape files", shapes_dir
                )
            for path in files:
                try:
                    shapes.parse(path.as_posix(), format="turtle")
                except Exception as error:
                    logger.warning("Failed to parse shapes file %s: %s", path, error)
    node_shape = SH.NodeShape
    if ontology_graph is not None and (None, RDF.type, node_shape) in ontology_graph:
        shapes += ontology_graph
    return shapes if len(shapes) else None


# --- LLM-free repair of SHACL violations -------------------------------------
#
# The contract for everything below: a repair either rewrites a term the
# catalog already declares, or removes a node that asserts nothing. No repair
# invents a value. A node carrying real data but missing a required property is
# left alone and reported -- filling it in would be fabrication, and dropping it
# would be data loss.

# Predicates that carry no assertion about the world: a node holding only these
# is a placeholder for an extraction that did not happen.
_EMPTY_NODE_PREDICATES = frozenset({RDF.type, RDFS.label, SKOS.prefLabel})

_RETYPABLE_COMPONENTS = frozenset({SH.DatatypeConstraintComponent})
_IRI_RESOLVABLE_COMPONENTS = frozenset(
    {SH.ClassConstraintComponent, SH.NodeKindConstraintComponent}
)
_PRUNABLE_COMPONENTS = frozenset({SH.MinCountConstraintComponent})


class ShaclRepairResult(BaseModel):
    """Outcome of the LLM-free SHACL repair pass."""

    model_config = {"arbitrary_types_allowed": True}

    graph: RDFGraph
    records: list[GraphRepairRecord] = Field(default_factory=list)
    violations_before: int = 0
    violations_after: int = 0
    passes_applied: int = 0
    reverted: bool = False
    ran: bool = False


def build_surface_index(
    ontology_graph: RDFGraph | None,
    code_predicates: Sequence[str] = (),
) -> dict[str, set[str]]:
    """Map exact catalog surface forms to the IRIs declaring them.

    Case-sensitive and exact: these are codes and names a model may have
    transcribed verbatim (``"d"``, ``"meV"``, ``"CsPbBr3"``), not free text to
    be fuzzy-matched. A form claimed by more than one IRI stays in the index and
    is rejected at lookup time — an ambiguous code is not a repairable one.

    Args:
        ontology_graph: Merged ontology context to index.
        code_predicates: Extra code-bearing predicates (UCUM codes, symbols,
            notations) on top of the standard name predicates.

    Returns:
        Surface form -> set of IRIs declaring it.
    """
    index: dict[str, set[str]] = {}
    if ontology_graph is None:
        return index
    predicates: list[URIRef] = [RDFS.label, SKOS.prefLabel, SKOS.notation]
    predicates.extend(URIRef(predicate) for predicate in code_predicates)
    for predicate in predicates:
        for subject, value in ontology_graph.subject_objects(predicate):
            if not isinstance(subject, URIRef) or not isinstance(value, Literal):
                continue
            text = str(value).strip()
            if text:
                index.setdefault(text, set()).add(str(subject))
    return index


def resolve_unique_surface(index: dict[str, set[str]], text: str) -> URIRef | None:
    """The single IRI declaring ``text`` as a surface form, if exactly one does."""
    candidates = index.get(text.strip(), set())
    if len(candidates) != 1:
        return None
    return URIRef(next(iter(candidates)))


def _literal_parses_as(lexical: str, datatype: URIRef) -> bool:
    """True when ``lexical`` is a well-formed literal of ``datatype``."""
    return Literal(lexical, datatype=datatype).value is not None


def _node_in_graph(graph: RDFGraph, node: Node) -> bool:
    """True when ``node`` appears in ``graph`` as a subject or an object.

    With the ontology context mixed into validation, pyshacl reports focus
    nodes that live only in the catalog. Those are not the gate's to repair,
    and "absent from the facts graph" must never read as "asserts nothing".
    """
    return (node, None, None) in graph or (None, None, node) in graph


def _node_asserts_nothing(graph: RDFGraph, node: Node) -> bool:
    """True when ``node`` is in the graph but carries nothing beyond typing/labels."""
    outgoing = list(graph.predicate_objects(node))
    if not outgoing:
        # Only a node the graph actually references is an empty placeholder;
        # a node absent from the graph entirely is simply not ours.
        return (None, None, node) in graph
    return all(predicate in _EMPTY_NODE_PREDICATES for predicate, _ in outgoing)


def _shacl_repairs_for(
    graph: RDFGraph,
    shapes_graph: RDFGraph,
    violations: Sequence[ShaclViolation],
    *,
    mode: str,
    surface_index: dict[str, set[str]],
    fact_namespaces: Sequence[str],
) -> tuple[list[tuple], list[tuple], list[GraphRepairRecord]]:
    """Derive (removals, additions, records) for one round of violations."""
    removals: list[tuple] = []
    additions: list[tuple] = []
    records: list[GraphRepairRecord] = []
    pruned: set[Node] = set()

    for violation in violations:
        if violation.severity != "error" or violation.focus is None:
            continue
        if isinstance(violation.focus, URIRef):
            if not _in_fact_scope(violation.focus, list(fact_namespaces)):
                # Ontology entities are not the gate's business to rewrite.
                continue
        elif not _node_in_graph(graph, violation.focus):
            # Blank nodes carry no namespace to scope by; presence in the
            # facts graph is the boundary. Catalog blank nodes (OWL
            # restrictions, property shapes) reported via the mixed-in
            # ontology stay untouched.
            continue
        component = violation.component
        shape = violation.source_shape

        if (
            component in _RETYPABLE_COMPONENTS
            and violation.path is not None
            and isinstance(violation.value, Literal)
            and shape is not None
        ):
            target = shapes_graph.value(shape, SH.datatype)
            lexical = str(violation.value).strip()
            if (
                isinstance(target, URIRef)
                and violation.value.datatype != target
                and _literal_parses_as(lexical, target)
            ):
                retyped = Literal(lexical, datatype=target)
                removals.append((violation.focus, violation.path, violation.value))
                additions.append((violation.focus, violation.path, retyped))
                records.append(
                    GraphRepairRecord(
                        kind=FactsGateRepairKind.SHACL_RETYPE,
                        source=f"{violation.path} {violation.value.n3()}",
                        target=retyped.n3(),
                    )
                )
            continue

        if (
            component in _IRI_RESOLVABLE_COMPONENTS
            and violation.path is not None
            and isinstance(violation.value, Literal)
        ):
            resolved = resolve_unique_surface(surface_index, str(violation.value))
            if resolved is not None:
                removals.append((violation.focus, violation.path, violation.value))
                additions.append((violation.focus, violation.path, resolved))
                records.append(
                    GraphRepairRecord(
                        kind=FactsGateRepairKind.SHACL_CODE_RESOLVED,
                        source=f"{violation.path} {violation.value.n3()}",
                        target=str(resolved),
                    )
                )
            continue

        if (
            mode == "prune"
            and component in _PRUNABLE_COMPONENTS
            and violation.focus not in pruned
            and _node_asserts_nothing(graph, violation.focus)
        ):
            referrers = {
                subject for subject, _ in graph.subject_predicates(violation.focus)
            }
            if len(referrers) > 1:
                # Shared by several subjects: removing it would silently change
                # statements that were never validated here.
                continue
            pruned.add(violation.focus)
            incoming = list(graph.triples((None, None, violation.focus)))
            outgoing = [
                (violation.focus, predicate, obj)
                for predicate, obj in graph.predicate_objects(violation.focus)
            ]
            removals.extend(incoming)
            removals.extend(outgoing)
            records.append(
                GraphRepairRecord(
                    kind=FactsGateRepairKind.SHACL_PRUNE,
                    source=str(violation.focus),
                    target="",
                    triple_count=len(incoming) + len(outgoing),
                )
            )

    return removals, additions, records


def _fact_scope_violations(
    violations: Sequence[ShaclViolation],
    fact_namespaces: Sequence[str],
) -> list[ShaclViolation]:
    """Violations that would survive the reporting filter.

    Mirrors the filter :func:`validate_aggregated_facts` applies to findings,
    so the ``violations_before``/``violations_after`` metrics count the same
    population as ``conforms`` does — with the ontology mixed in, the raw
    pyshacl count includes catalog nodes the report never shows.
    """
    namespaces = [ns for ns in fact_namespaces if ns]
    if not namespaces:
        return list(violations)
    return [
        violation
        for violation in violations
        if violation.focus is None
        or (
            isinstance(violation.focus, URIRef)
            and _in_fact_scope(violation.focus, namespaces)
        )
    ]


def apply_shacl_repairs(
    graph: RDFGraph,
    shapes_graph: RDFGraph | None,
    ontology_graph: RDFGraph | None,
    *,
    mode: str = "prune",
    passes: int = 1,
    fact_namespaces: Sequence[str] = (),
    code_predicates: Sequence[str] = (),
    inference: str = "rdfs",
    advanced: bool = True,
    max_triples: int = 0,
    initial_violations: Sequence[ShaclViolation] | None = None,
) -> ShaclRepairResult:
    """Repair SHACL violations in code, with no LLM round-trip.

    Bounded ``validate -> repair -> revalidate`` loop. A pass is kept only when
    it strictly reduces the violation count: a repair that trades triples for
    no conformance gain is reverted, the same discipline the un-merge repair
    uses.

    Repairs by constraint component:
        - ``sh:datatype``: retype a literal that parses as the declared
          datatype (``"2019"^^xsd:string`` -> ``"2019"^^xsd:gYear``).
        - ``sh:class`` / ``sh:nodeKind``: replace a string literal with the one
          catalog IRI declaring it as a surface form (``qudt:unit "meV"`` ->
          ``unit:MilliElectronVolt``). Ambiguous forms are left reported.
        - ``sh:minCount`` (mode ``prune`` only): drop a focus node that asserts
          nothing beyond ``rdf:type``/``rdfs:label`` and is referenced by at
          most one subject, together with that reference.

    Everything else -- ``sh:maxCount`` (owned by the functional-violation and
    un-merge machinery), ``sh:not``, ``sh:qualifiedValueShape``, SPARQL
    constraints -- is reported, never repaired.

    Args:
        graph: Aggregated facts graph, repaired **in place**: it may be
            oxigraph-backed and carry RDF 1.2 triple terms, which a copied
            rdflib graph would silently drop. A pass that fails the accept
            test is rolled back triple-for-triple instead.
        shapes_graph: Shapes to validate against; ``None`` disables the pass.
        ontology_graph: Merged ontology context, indexed for surface forms.
        mode: ``off`` | ``rewrite`` (rewrites only) | ``prune`` (also prunes).
        passes: Maximum repair rounds.
        fact_namespaces: Only nodes under these namespaces are repaired.
        code_predicates: Code-bearing predicates for surface resolution.
        inference: pyshacl pre-inference mode.
        advanced: Enable SHACL Advanced Features.
        max_triples: Skip validation above this graph size; 0 disables.
        initial_violations: Violations already computed for ``graph`` with the
            same parameters (e.g. by the reporting pass), reused to skip the
            redundant first validation.

    Returns:
        The repaired graph, the applied repair records, and fact-scoped
        violation counts before and after (the population ``conforms`` is
        judged on; the loop's accept test uses the raw count internally).
    """
    if mode == "off" or shapes_graph is None or not len(shapes_graph) or passes <= 0:
        return ShaclRepairResult(graph=graph)

    def _validate(target: RDFGraph) -> list[ShaclViolation] | None:
        return run_shacl(
            target,
            shapes_graph,
            ontology_graph=ontology_graph,
            inference=inference,
            advanced=advanced,
            max_triples=max_triples,
        )

    violations = (
        list(initial_violations) if initial_violations is not None else _validate(graph)
    )
    if violations is None:
        return ShaclRepairResult(graph=graph)

    def _scoped_count(candidates: Sequence[ShaclViolation]) -> int:
        return len(_fact_scope_violations(candidates, fact_namespaces))

    result = ShaclRepairResult(
        graph=graph,
        violations_before=_scoped_count(violations),
        violations_after=_scoped_count(violations),
        ran=True,
    )
    surface_index = build_surface_index(ontology_graph, code_predicates)

    def _rollback(added: Sequence[tuple], removed: Sequence[tuple]) -> None:
        for triple in added:
            graph.remove(triple)
        for triple in removed:
            graph.add(triple)

    for _ in range(passes):
        if not violations:
            break
        removals, additions, records = _shacl_repairs_for(
            graph,
            shapes_graph,
            violations,
            mode=mode,
            surface_index=surface_index,
            fact_namespaces=fact_namespaces,
        )
        if not records:
            break

        applied_removals: list[tuple] = []
        seen_removals: set[tuple] = set()
        for triple in removals:
            if triple in seen_removals:
                continue
            seen_removals.add(triple)
            if triple in graph:
                graph.remove(triple)
                applied_removals.append(triple)
        applied_additions: list[tuple] = []
        for triple in additions:
            if triple not in graph:
                graph.add(triple)
                applied_additions.append(triple)

        candidate_violations = _validate(graph)
        if candidate_violations is None:
            _rollback(applied_additions, applied_removals)
            break
        if len(candidate_violations) >= len(violations):
            logger.warning(
                "SHACL autofix: pass did not reduce violations (%d -> %d); "
                "keeping the pre-repair graph",
                len(violations),
                len(candidate_violations),
            )
            _rollback(applied_additions, applied_removals)
            result.reverted = True
            break

        logger.info(
            "SHACL autofix: %d repair(s) applied, violations %d -> %d",
            len(records),
            len(violations),
            len(candidate_violations),
        )
        violations = candidate_violations
        result.records.extend(records)
        result.passes_applied += 1
        result.violations_after = _scoped_count(candidate_violations)

    return result


# Scaffolding every facts graph uses regardless of catalog: flagging rdfs:label
# or rdf:type as "not in the ontology context" would bury the signal in noise.
_SCAFFOLDING_NAMESPACES = (str(RDF), str(RDFS), str(OWL), str(XSD))


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
          one (subject, predicate); or >= 2 IRI objects on a predicate that is
          single-valued for a dominant majority of other subjects. Severity is
          configurable — legitimate multi-value modeling exists, bad merges
          are far more common.
        - ``DEGENERATE_COREFERENCE``: one IRI object shared by >= 2 distinct
          functional-ish predicates of one subject (collapsed range bounds).
        - ``SHACL``: optional, when ``pyshacl`` is installed and shapes exist.
        - ``NON_CATALOG_VOCABULARY``: warning-only telemetry for terms the
          ontology context never supplied, which mark a retrieval miss the
          renderer papered over with a documented fallback.

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

    Returns:
        Report with all findings, ordered by subject.
    """
    namespaces = [ns for ns in (fact_namespaces or []) if ns]
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

    dominant_single = _dominant_single_valued_predicates(
        iri_groups, min_single_support=functional_min_single_support
    )
    functional_ish = functional | dominant_single

    findings: list[FactsValidationFinding] = []
    flagged_pairs: set[tuple[URIRef, URIRef]] = set()

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

        iri_objects = iri_groups.get((subject, predicate), set())
        if (
            len(iri_objects) >= 2
            and predicate not in functional
            and predicate in dominant_single
        ):
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
        findings.extend(
            finding
            for finding in (violation.as_finding() for violation in shacl_violations)
            if not finding.subject
            or _in_fact_scope(URIRef(finding.subject), namespaces)
        )

    findings.extend(
        _non_catalog_vocabulary_findings(
            graph, ontology_graph, namespaces, quantity_fallback_vocabulary
        )
    )

    findings.extend(_dangling_reference_findings(graph, namespaces))

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


def format_findings_for_prompt(findings: list[FactsUnitFinding]) -> str:
    """Render findings as MANDATORY-fixes + coverage blocks for the renderer."""
    mandatory = [finding for finding in findings if finding.mandatory]
    coverage = [finding for finding in findings if not finding.mandatory]
    sections: list[str] = []
    if mandatory:
        lines = ["## MANDATORY fixes (deterministic validation — apply every item)"]
        for index, finding in enumerate(mandatory, 1):
            line = f"{index}. {finding.message}"
            if finding.suggestions:
                line += " Candidates: " + ", ".join(
                    f"<{suggestion}>" for suggestion in finding.suggestions
                )
            lines.append(line)
        sections.append("\n".join(lines))
    if coverage:
        lines = ["## Verify numeric coverage"]
        lines.extend(finding.message for finding in coverage)
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
