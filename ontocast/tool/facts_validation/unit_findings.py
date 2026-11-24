"""Deterministic per-unit findings on a rendered facts graph.

Only unresolved, machine-verified issues go back to the LLM, as MANDATORY
fix instructions that must be resolved by rewriting — never by deleting the
statement.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Literal as TypingLiteral

from pydantic import BaseModel, ConfigDict
from rdflib import OWL, RDF, RDFS, SKOS, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.annotation import is_annotation
from ontocast.onto.model import (
    FactsUnitFinding,
    FactsUnitFindingKind,
)
from ontocast.onto.property_sides import sides_by_property
from ontocast.onto.rdfgraph import (
    RDFGraph,
    RejectedLiteralTriple,
)
from ontocast.onto.scaffolding import FactScope
from ontocast.onto.well_known import StatementRoleName, well_known_terms
from ontocast.tool.agg.signatures import harvest_max_one_predicates
from ontocast.tool.agg.statement_conflict import (
    StatementIndex,
    dimension_conflict_message,
    dimension_conflicts,
)
from ontocast.tool.facts_validation.connectivity import (
    isolated_instances,
    states_a_value,
)
from ontocast.tool.facts_validation.grounding import (
    node_key,
    value_grounding_findings,
)
from ontocast.tool.facts_validation.grounding import (
    node_name as _node_name,
)
from ontocast.tool.facts_validation.statements import (
    StatementContract,
    own_terms,
    value_components,
)
from ontocast.tool.facts_validation.term_verdict import (
    TermVerdictKind,
    term_verdict,
    used_terms,
)
from ontocast.tool.facts_validation.terms import (
    _STANDARD_NAMESPACES,
    CatalogRoles,
    ValidationPolicy,
    _alias_candidates,
    _catalog_term_roles,
    _declared_domains,
    _described_classes,
    _folded_local_name,
    _local_name,
    _name_tokens,
    _namespace_of,
    _resolve_type_literal,
    _vocabulary_role_subset,
    collect_catalog_terms,
    collect_declared_namespaces,
    expand_vocabulary_terms,
    types_compatible,
    untyped_instances,
)
from ontocast.util.numeric_inventory import (
    NumericInventory,
    canonical_number,
    missing_numeric_inventory,
    states_quantity,
    unit_surfaces_in_ontology,
)

logger = logging.getLogger(__name__)

CoverageMode = TypingLiteral["off", "measurements", "all"]


def coverage_mode(value: str | bool | None) -> CoverageMode:
    """Normalise the coverage-mandatory knob, boolean form included.

    ``True`` is the setting's old spelling of ``all`` and ``False`` of
    ``off``; anything unrecognised is ``off``, the advisory default.
    """
    if isinstance(value, bool):
        return "all" if value else "off"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "measurements":
            return "measurements"
        if lowered in ("all", "true", "1", "yes", "on"):
            return "all"
    return "off"


def unit_numeric_inventory(
    *,
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    extraction_text: str,
    policy: ValidationPolicy | None = None,
    limit: int = 30,
) -> NumericInventory:
    """Numbers of the unit text absent from its graph, measurements first.

    The unit surfaces come from the unit individuals of the ontology context
    (found through the configured unit-role property), so a catalog-specific
    unit counts as a measurement once the catalog declares it. Measurements
    are judged against structured ``(number, unit)`` pairs in the graph —
    a bare numeric literal does not clear a unit-adjacent mention. Shared by
    the coverage findings and the completion pass, which must agree on what
    is missing.
    """
    policy = policy or ValidationPolicy()
    unit_properties = expand_vocabulary_terms(
        _vocabulary_role_subset(policy.quantity_fallback_vocabulary, "unit"),
        graph,
        ontology_graph,
    )
    return missing_numeric_inventory(
        extraction_text,
        graph,
        unit_surfaces=unit_surfaces_in_ontology(ontology_graph, unit_properties),
        ontology_graph=ontology_graph,
        unit_properties=unit_properties,
        ignore_identifier_fragments=policy.numeric_identifier_guard,
        limit=limit,
    )


class UnitPromptScope(BaseModel):
    """Which conditional rules the facts prompts of one content unit carry.

    One decision, read by the render, critic and gap-pass prompts alike.

    Attributes:
        quantities: The quantity chapter: the unit's text states a number
            (:func:`~ontocast.util.numeric_inventory.states_quantity`) and the
            unit is not a citation-metadata unit.
        statements: The statement rules outside the chapter (a statement's
            subject, the sensor link, the statement checklist line, the
            measurement vocabularies among the well-known ones): ``quantities``,
            or the unit's ontology context holds a statement class of the
            catalog's own (one under a well-known statement class).
        attribution: The attribution rule: every unit but a citation-metadata
            unit, which the citation lane governs.
    """

    model_config = ConfigDict(frozen=True)

    quantities: bool = False
    statements: bool = False
    attribution: bool = True


def unit_prompt_scope(
    *,
    extraction_text: str,
    ontology_graph: RDFGraph | None,
    contract: StatementContract,
    policy: ValidationPolicy | None = None,
    is_citation_metadata: bool = False,
) -> UnitPromptScope:
    """The :class:`UnitPromptScope` of one content unit.

    A number is read as the coverage inventory reads it, with the same unit
    surfaces and identifier guard as :func:`unit_numeric_inventory`, so the
    prompts and the coverage findings agree on which units state numbers.
    """
    if is_citation_metadata:
        quantities = False
    else:
        policy = policy or ValidationPolicy()
        unit_properties = expand_vocabulary_terms(
            _vocabulary_role_subset(policy.quantity_fallback_vocabulary, "unit"),
            ontology_graph,
        )
        quantities = states_quantity(
            extraction_text,
            unit_surfaces=unit_surfaces_in_ontology(ontology_graph, unit_properties),
            ignore_identifier_fragments=policy.numeric_identifier_guard,
        )
    return UnitPromptScope(
        quantities=quantities,
        statements=quantities
        or bool(own_terms(ontology_graph, contract.statement_classes)),
        attribution=not is_citation_metadata,
    )


def domain_violation_findings(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
) -> list[FactsUnitFinding]:
    """Report subjects whose asserted type contradicts a predicate's ``rdfs:domain``.

    Only ``rdfs:domain`` entails, so only it can be contradicted: classes a
    ``schema:domainIncludes`` lists are expected ones, which later uses may
    widen. An untyped subject is never a violation -- the parse-time repair
    types it from the domain, and a subject still untyped is
    ``UNTYPED_INSTANCE``'s to report. It becomes one when the subject carries
    an asserted type unrelated to every domain class: inference would add the
    domain class on top of an incompatible one.

    Conservative by construction, since a false accusation costs a repair.
    A subject is reported only when it has at least one asserted type and every
    asserted type is *unrelated* to every declared domain class -- neither a
    subtype nor a supertype of it, following ``rdfs:subClassOf`` and
    ``owl:equivalentClass`` intersections in both directions. Typing a subject
    with a supertype of the domain (``sosa:Observation`` where the domain is
    ``ex:Measurement``) is consistent and not reported.

    Args:
        graph: Rendered facts graph for one unit.
        ontology_graph: Ontology context the renderer was given.

    Returns:
        list: One mandatory finding per offending (subject, predicate) pair,
        ordered by subject then predicate.
    """
    if ontology_graph is None or not len(ontology_graph):
        return []
    domains = _declared_domains(ontology_graph, source="rdfs")
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
        asserted = {
            value
            for value in graph.objects(subject, RDF.type)
            if isinstance(value, URIRef)
        }
        # Only types and domains the context places in a hierarchy can be
        # argued about: an undecidable pair is not reported.
        fits = types_compatible(
            asserted, declared, ontology_graph, described=described, closures=closures
        )
        if fits is not False:
            continue
        declared = {value for value in declared if value in described}
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
                    "as a class of the domain, or use the property that fits "
                    "the type it has."
                ),
                subject=str(subject),
                predicate=str(predicate),
                suggestions=sorted(str(value) for value in declared),
            )
        )
    return findings


_UNTYPED_LABEL_SUGGESTIONS = 3


def _label_matched_classes(
    graph: RDFGraph, instance: URIRef, known_classes: set[str]
) -> list[str]:
    """Catalog classes whose local name shares a token with the instance's name.

    The instance's name is its ``rdfs:label`` and ``skos:prefLabel`` literals
    plus the local name of its IRI; a class qualifies when the two token sets
    intersect. An advisory list, capped at a few entries.
    """
    tokens: set[str] = _name_tokens(_local_name(str(instance)))
    for predicate in (RDFS.label, SKOS.prefLabel):
        for value in graph.objects(instance, predicate):
            if isinstance(value, Literal):
                tokens |= _name_tokens(str(value))
    if not tokens:
        return []
    matched = [
        candidate
        for candidate in sorted(known_classes)
        if _name_tokens(_local_name(candidate)) & tokens
    ]
    return matched[:_UNTYPED_LABEL_SUGGESTIONS]


def untyped_instance_findings(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    fact_namespaces: Sequence[str],
) -> list[FactsUnitFinding]:
    """Report every instance of the graph that has no concrete type.

    An instance is a fact-namespace IRI in subject or IRI-object position
    (:attr:`~ontocast.onto.scaffolding.FactScope.instances`); it is typed when one of its ``rdf:type`` objects
    is a class outside the fact namespaces that is neither a schema metaclass
    nor vacuous (:func:`concrete_types`). Suggested classes are the declared
    domain of every predicate the instance carries and the declared range of
    every predicate it is the object of (``rdfs`` or ``schema:*Includes``), then catalog
    classes whose name shares a token with the instance's label or local
    name.

    Args:
        graph: Rendered facts graph for one unit.
        ontology_graph: Ontology context the renderer was given.
        fact_namespaces: Namespaces holding minted instances.

    Returns:
        list: One mandatory finding per untyped instance, ordered by IRI.
    """
    untyped = untyped_instances(graph, fact_namespaces)
    if not untyped:
        return []
    domains: dict[URIRef, set[URIRef]] = {}
    ranges: dict[URIRef, set[URIRef]] = {}
    known_classes: set[str] = set()
    if ontology_graph is not None and len(ontology_graph):
        domains = _declared_domains(ontology_graph)
        ranges = sides_by_property(ontology_graph, "range")
        _, known_classes = _catalog_term_roles(ontology_graph)
    findings: list[FactsUnitFinding] = []
    for instance in untyped:
        entailed: set[URIRef] = set()
        for predicate in graph.predicates(instance, None):
            if isinstance(predicate, URIRef):
                entailed |= domains.get(predicate, set())
        for predicate in graph.predicates(None, instance):
            if isinstance(predicate, URIRef):
                entailed |= ranges.get(predicate, set())
        suggestions = sorted(str(value) for value in entailed)
        suggestions.extend(
            candidate
            for candidate in _label_matched_classes(graph, instance, known_classes)
            if candidate not in suggestions
        )
        asserted = sorted(
            f"<{value}>"
            for value in graph.objects(instance, RDF.type)
            if isinstance(value, URIRef)
        )
        situation = (
            f"is typed only {', '.join(asserted)}, which names no concrete class"
            if asserted
            else "has no rdf:type"
        )
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.UNTYPED_INSTANCE,
                message=(
                    f"<{instance}> {situation}. Type it with the class from the "
                    "ONTOLOGY section that fits what the text says it is, or, "
                    "only if none fits, with a specific class from a core "
                    "vocabulary such as schema.org. Not owl:Thing, not a "
                    "metaclass, not a class minted in the facts namespace. "
                    "Keep the instance and its statements."
                ),
                subject=str(instance),
                suggestions=suggestions,
            )
        )
    return findings


def _role_message(node: URIRef, role: str, link: str) -> str:
    if role == "subject":
        return (
            f"<{node}> states a value but names nothing it is about. Link "
            "it to the instance the text says was measured or described "
            "(its feature of interest, or the property the ontology gives "
            "for it), in the direction the property runs; mint that "
            "instance if the text names it and it is missing."
        )
    if role == "property":
        return (
            f"<{node}> states a value but names no property it reports. Link "
            "it to the property the text says was measured or observed, with "
            "a property link its class takes; mint that property instance if "
            "it is missing."
        )
    if role == "link":
        return (
            f"<{node}> states a value without <{link}>, which a restriction "
            "on its class requires. Add it if the text gives it."
        )
    return f"<{node}> states a value but has no {role} link, which its class requires."


def missing_role_findings(
    graph: RDFGraph,
    contract: StatementContract,
    fact_namespaces: Sequence[str],
    *,
    max_candidates: int = 12,
) -> list[FactsUnitFinding]:
    """Report every statement missing a role its class requires.

    A statement is an instance of a statement class; one that states a value
    (:func:`~ontocast.tool.facts_validation.connectivity.states_a_value`)
    must carry each role its classes require
    (:meth:`~ontocast.tool.facts_validation.statements.StatementContract.required_roles`):
    a link playing the role, from the statement or to it as the link runs,
    whose declared side admits one of the statement's classes
    (:meth:`~ontocast.tool.facts_validation.statements.StatementContract.serves`).
    A link a catalog restriction requires on its class (``required_links``)
    must be there too, or a sub-property of it. A value (an instance of a
    value class) outside any statement carries its subject by being linked to
    another instance, in either direction. An instance of another class
    stating its own values is a fact about itself and is not reported.

    Args:
        graph: Rendered facts graph for one unit.
        contract: The catalog's statement contract.
        fact_namespaces: Namespaces holding minted instances.
        max_candidates: Most candidate subjects suggested per finding.

    Returns:
        list: One mandatory finding per missing role, ordered by IRI then
        role; ``facet`` is the role (``link`` for a required link, named in
        ``predicate``), the suggestions the unit's instances that are neither
        statements nor values.
    """
    statements = {URIRef(iri) for iri in contract.statement_classes}
    values = {URIRef(iri) for iri in contract.value_classes}
    scope = FactScope.of(graph, fact_namespaces)
    isolated = isolated_instances(graph, fact_namespaces)
    instances = sorted(scope.instances, key=str)

    def plays(node: URIRef, role: StatementRoleName, types: set[Node]) -> bool:
        links = contract.links(role)
        return any(
            contract.serves(str(prop), types)
            for prop in graph.predicates(node, None)
            if str(prop) in links.forward
        ) or any(
            contract.serves(str(prop), types)
            for prop in graph.predicates(None, node)
            if str(prop) in links.inverse
        )

    def carries(node: URIRef, link: str) -> bool:
        return any(
            link in contract.properties_of(str(prop))
            for prop in graph.predicates(node, None)
        )

    lapsed: list[tuple[URIRef, str, str]] = []
    for node in instances:
        types = set(graph.objects(node, RDF.type))
        if not types & (statements | values) or not states_a_value(scope, node):
            continue
        if types & statements:
            for role in sorted(contract.required_roles(types)):
                if not plays(node, role, types):
                    lapsed.append((node, role, ""))
            required = set().union(
                *(contract.required_links.get(str(cls), frozenset()) for cls in types)
            )
            for link in sorted(required):
                if not carries(node, link):
                    lapsed.append((node, "link", link))
        elif node in isolated:
            lapsed.append((node, "subject", ""))
    if not lapsed:
        return []
    candidates = [
        str(node)
        for node in instances
        if not set(graph.objects(node, RDF.type)) & (statements | values)
    ][:max_candidates]
    return [
        FactsUnitFinding(
            kind=FactsUnitFindingKind.MISSING_ROLE,
            message=_role_message(node, role, link),
            subject=str(node),
            predicate=link,
            facet=role,
            suggestions=candidates,
        )
        for node, role, link in lapsed
    ]


def _role_link_sides(
    contract: StatementContract, ontology_graph: RDFGraph
) -> dict[str, set[URIRef]]:
    """Each role link with the classes its statement end is declared to be.

    The side is the link's domain when it runs from the statement, its range
    when it runs to it (``rdfs`` or ``schema:*Includes``); a link declaring
    none takes those of its catalog super-properties. A link with no side
    anywhere is left out.
    """
    sides = {
        "forward": sides_by_property(ontology_graph, "domain"),
        "inverse": sides_by_property(ontology_graph, "range"),
    }
    found: dict[str, set[URIRef]] = {}
    for role in contract.roles:
        links = contract.links(role)
        for direction, props in (
            ("forward", links.forward),
            ("inverse", links.inverse),
        ):
            for prop in props:
                own = sides[direction].get(URIRef(prop))
                declared = (
                    set(own)
                    if own
                    else set().union(
                        *(
                            sides[direction].get(URIRef(parent), set())
                            for parent in contract.properties_of(prop) - {prop}
                        )
                    )
                )
                if declared:
                    found[prop] = declared
    return found


def _outside_message(node: Node, role: str, link: str, expected: str) -> str:
    if role == "result":
        advice = (
            "A value held this way names no quantity. Restate it as a "
            "statement about the node that names the quantity it measures; "
            "this link is removed when the unit leaves the loop."
        )
    else:
        advice = (
            "Give the link to the statement it belongs to; this link is "
            "removed when the unit leaves the loop."
        )
    return (
        f"<{node}> is not a statement, but carries <{link}>, which links a "
        f"statement ({expected}) to its {role}. {advice}"
    )


def role_outside_statement_findings(
    scope: FactScope,
    contract: StatementContract,
    ontology_graph: RDFGraph | None,
) -> list[FactsUnitFinding]:
    """Report every statement role link whose statement end is no statement.

    A role link (:meth:`StatementContract.links`) runs from a statement or to
    one; its statement end must fit the link's declared side
    (:func:`_role_link_sides`). The end is reported when the catalog places
    its types and the side in a hierarchy and they are unrelated
    (:func:`types_compatible` is ``False``): the link then gives a value,
    subject or property to what is not a statement, and a result held so
    names no quantity. An untyped end, or a pair the catalog cannot decide,
    is not reported.

    Args:
        scope: The unit's fact scope.
        contract: The catalog's statement contract.
        ontology_graph: Ontology context the renderer was given.

    Returns:
        list: One mandatory finding per link, ordered by subject, link and
        object; ``facet`` is the role, ``suggestions`` the declared classes.
    """
    if ontology_graph is None or not len(ontology_graph):
        return []
    sides = _role_link_sides(contract, ontology_graph)
    if not sides:
        return []
    graph = scope.graph
    described = _described_classes(ontology_graph)
    closures: dict[URIRef, set[URIRef]] = {}
    findings: list[FactsUnitFinding] = []
    for role in sorted(contract.roles):
        links = contract.links(role)
        for direction, props in (
            ("forward", links.forward),
            ("inverse", links.inverse),
        ):
            for prop in sorted(props):
                declared = sides.get(prop)
                if not declared:
                    continue
                for subject, obj in graph.subject_objects(URIRef(prop)):
                    triple = (subject, URIRef(prop), obj)
                    node = subject if direction == "forward" else obj
                    if not scope.is_fact(triple) or not scope.holds(node):
                        continue
                    asserted = {
                        value
                        for value in graph.objects(node, RDF.type)
                        if isinstance(value, URIRef)
                    }
                    fits = types_compatible(
                        asserted,
                        declared,
                        ontology_graph,
                        described=described,
                        closures=closures,
                    )
                    if fits is not False:
                        continue
                    classes = sorted(str(value) for value in declared)
                    findings.append(
                        FactsUnitFinding(
                            kind=FactsUnitFindingKind.ROLE_OUTSIDE_STATEMENT,
                            message=_outside_message(
                                node,
                                role,
                                prop,
                                ", ".join(f"<{value}>" for value in classes),
                            ),
                            subject=node_key(subject),
                            predicate=prop,
                            value=node_key(obj),
                            facet=role,
                            suggestions=classes,
                        )
                    )
    return sorted(findings, key=lambda f: (f.subject, f.predicate, f.value))


def empty_value_findings(
    scope: FactScope, contract: StatementContract
) -> list[FactsUnitFinding]:
    """Report every value node that states nothing.

    A value (an instance of a value class, blank or named) states nothing
    when it carries no triple but its types and annotations: no number,
    bound, unit, qualifier or text. Such a node is a label, not a value.

    Returns:
        list: One mandatory finding per value, ordered by node.
    """
    graph = scope.graph
    values = {URIRef(iri) for iri in contract.value_classes}
    if not values:
        return []
    found: list[FactsUnitFinding] = []
    for node in {s for s, _, o in scope.triples((None, RDF.type, None)) if o in values}:
        if not scope.holds(node) or any(
            predicate not in (RDF.type, OWL.sameAs) and not is_annotation(predicate)
            for predicate in graph.predicates(node, None)
        ):
            continue
        found.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.EMPTY_VALUE,
                message=(
                    f"{_node_name(graph, node)} is a value that states nothing: "
                    "no number, bound, unit or qualifier. It is removed when "
                    "the unit leaves the loop."
                ),
                subject=node_key(node),
            )
        )
    return sorted(found, key=lambda f: f.subject)


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
    scope: FactScope,
    contract: StatementContract,
    *,
    unit_properties: set[str],
    numeric_value_properties: set[str],
) -> list[FactsUnitFinding]:
    """Flag value nodes whose only numeric content sits in a label.

    A fact node, blank or named (:meth:`FactScope.holds`), carrying a unit
    property but no number among its value components
    (:func:`~ontocast.tool.facts_validation.statements.value_components`),
    while its label or comment contains a number, is a measurement the
    renderer described instead of structuring — invisible to every query and
    to SHACL alike. Driven entirely by the configured quantity vocabulary
    (the unit-role property identifies value nodes), so no domain terms are
    hardcoded.
    """
    if not unit_properties:
        return []
    graph = scope.graph
    annotation_predicates = {RDFS.label, RDFS.comment, SKOS.prefLabel, SKOS.altLabel}
    findings: list[FactsUnitFinding] = []
    subjects = {
        subject
        for subject, predicate, _ in scope.triples()
        if str(predicate) in unit_properties and scope.holds(subject)
    }
    for subject in sorted(subjects, key=str):
        if value_components(graph, subject, contract).numbers:
            continue
        label_numbers: set[str] = set()
        label_text = ""
        for _, predicate, obj in graph.triples((subject, None, None)):
            if not isinstance(obj, Literal) or predicate not in annotation_predicates:
                continue
            for match in _LABEL_NUMBER_PATTERN.finditer(str(obj)):
                canonical = canonical_number(match.group(1))
                if canonical is not None:
                    label_numbers.add(canonical)
                    if not label_text:
                        label_text = str(obj)
        if not label_numbers:
            continue
        numbers = ", ".join(sorted(label_numbers, key=lambda v: (len(v), v)))
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.LABEL_ONLY_NUMBER,
                message=(
                    f"{_node_name(graph, subject)} carries a unit but no numeric literal on any "
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
    scope: FactScope,
    contract: StatementContract,
    ontology_graph: RDFGraph | None,
) -> list[FactsUnitFinding]:
    """Flag one numeric value duplicated across single-valued numeric predicates.

    An exact scalar written into two distinct schema-constrained (functional /
    max-1) numeric predicates of one node — the classic case being equal lower
    and upper bounds — encodes a single measurement as if it carried epistemic
    structure it does not have. Fully generic: predicates come from the
    ontology's own functional/cardinality declarations, values are the numbers
    of each fact node, blank or named
    (:func:`~ontocast.tool.facts_validation.statements.value_components`).
    """
    constrained = harvest_max_one_predicates(ontology_graph)
    if not constrained:
        return []
    graph = scope.graph
    findings: list[FactsUnitFinding] = []
    by_subject: dict[Node, dict[URIRef, set[str]]] = {}
    for subject, predicate, obj in scope.triples():
        if (
            subject in by_subject
            or not isinstance(obj, Literal)
            or predicate not in constrained
            or not scope.holds(subject)
        ):
            continue
        per_predicate: dict[URIRef, set[str]] = {}
        for number in value_components(graph, subject, contract).numbers:
            if number.number is not None and URIRef(number.predicate) in constrained:
                per_predicate.setdefault(URIRef(number.predicate), set()).add(
                    number.number
                )
        by_subject[subject] = per_predicate
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
                        f"{_node_name(graph, subject)} carries the same numeric value {value} on "
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
    *,
    min_terms: int = 4,
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
    catalog-free deployment is not spammed; when ``min_share`` is 0; and when
    the render uses fewer than ``min_terms`` distinct schema terms. The last
    guard exists because a share has no meaning over a denominator of one or
    two: a front-matter unit that types an identifier and an author with
    generic vocabulary is not a render that abandoned the catalog, and the
    mandatory finding it used to raise drove the critic into retyping the
    identifier as a quantity value.

    Args:
        graph: The rendered unit graph.
        catalog_terms: Terms declared by the unit's ontology context.
        fact_namespaces: Namespaces holding minted instances.
        min_share: Floor on the catalog share, in [0, 1]. 0 disables.
        min_terms: Fewest distinct schema terms before the share is judged.

    Returns:
        list[FactsUnitFinding]: At most one finding.
    """
    if not catalog_terms or not len(graph) or min_share <= 0:
        return []
    from_catalog, total = domain_vocabulary_share(graph, catalog_terms, fact_namespaces)
    if not total or total < min_terms or from_catalog / total >= min_share:
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


def _containment_qualified(term: URIRef, candidates: list[str]) -> list[str]:
    """Keep only replacement candidates that are near-misses of the term.

    The same qualification the LLM-free alias repair applies: a suggestion is
    offered when the local-name tokens of one side are a subset of the
    other's (``value`` in ``numericValue``), or when the two spell the same
    name once case and separators are folded -- never on string similarity
    alone. A pair that swaps one token for another
    (``hasASiteComponent`` / ``hasBSiteComponent``) is a different property,
    and offering it as a candidate would reproduce through the critic the
    substitution the repair no longer makes.
    """
    alias_tokens = _name_tokens(_local_name(str(term)))
    folded = _folded_local_name(str(term))
    kept: list[str] = []
    for candidate in candidates:
        candidate_tokens = _name_tokens(_local_name(candidate))
        if alias_tokens and (
            alias_tokens <= candidate_tokens or candidate_tokens <= alias_tokens
        ):
            kept.append(candidate)
        elif folded and folded == _folded_local_name(candidate):
            kept.append(candidate)
    return kept


def collect_unit_findings(
    *,
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    quarantined: list[RejectedLiteralTriple],
    extraction_text: str,
    fact_namespaces: list[str],
    coverage_limit: int = 30,
    coverage_mandatory: str | bool = "off",
    policy: ValidationPolicy | None = None,
    is_citation_metadata: bool = False,
    is_non_content: bool = False,
    contract: StatementContract = StatementContract(),
    full_catalog_terms: set[str] | None = None,
    document: Node | None = None,
    catalog_roles: CatalogRoles | None = None,
) -> list[FactsUnitFinding]:
    """Assemble all deterministic findings for one rendered unit graph.

    Mandatory: quarantined literals (with closed-range individual
    suggestions), forbidden-namespace terms (``example.org``), doc-namespace
    predicates, terms a closed vocabulary does not have in any position
    (:func:`~ontocast.tool.facts_validation.term_verdict.used_terms`; an
    individual of a value whose property's range is a closed class by role),
    unresolved catalog near-misses, predicates asserted on a subject whose
    type contradicts their ``rdfs:domain``, statement role links held by a
    node that is not a statement (:func:`role_outside_statement_findings`),
    values that state nothing, statements missing a role their
    class requires and values linked to nothing (``contract``), instances
    with no concrete type, value nodes whose only numeric content sits in a
    label, and the numbers and individuals of values the text does not bear
    out (:func:`~ontocast.tool.facts_validation.grounding.value_grounding_findings`),
    and results that disagree in dimension under a property whose quantity
    kind or measurement unit the catalog declares (:func:`dimension_findings`).
    Advisory by default: numeric
    mentions of the source text absent from the graph, as two findings --
    measurements (a number with its unit, in text order with context) and
    bare numbers -- whose mandatory flag ``coverage_mandatory`` sets
    (``off`` | ``measurements`` | ``all``; the booleans are ``off``/``all``).

    The policy's exempt terms (the sanctioned fallback vocabulary the facts
    prompt itself names, plus code predicates) never raise UNKNOWN_TERM:
    flagging the vocabulary the prompt recommends produced mandatory findings
    that repair renders obeyed by deleting correct data.

    A citation-metadata unit is rendered with the bibliographic vocabulary
    and told to keep away from the domain ontology, so its catalog share is
    zero by instruction; ``DOMAIN_ADHERENCE`` is not judged there, nor on a
    non-content unit (front matter, author notes, identifiers), which has no
    domain facts for the catalog to describe.

    ``full_catalog_terms`` is the whole catalog's inventory, as opposed to
    the unit's retrieved snapshot in ``ontology_graph``. A term the snapshot
    did not retrieve is still a term, so it is never reported as unknown --
    and, for a term that is genuinely unknown, replacement candidates are
    offered only under token containment, never on similarity alone.

    ``document`` is the IRI of the document the unit belongs to: a work the
    text cites is a fact instance, never the document node.

    ``catalog_roles`` carries the whole catalog's axioms. The checks that read
    a property's domain or range (``domain_violation``,
    ``role_outside_statement``) read them over the snapshot together with
    those axioms (:meth:`CatalogRoles.with_axioms`): a snapshot may list a
    property without its sides, and what retrieval kept must not decide what
    is a violation. The other checks read the snapshot the unit was given.
    """
    policy = policy or ValidationPolicy()
    known_terms = set(full_catalog_terms or ())
    findings: list[FactsUnitFinding] = []
    standard_namespaces = policy.standard_namespaces()
    # The vocabularies the catalog builds on: a namespace it declares terms
    # in, or borrows a term from (a superclass, a restricted property). A
    # strict vocabulary closes everything outside them; what the catalog
    # borrows it is no authority on, so a borrowed namespace stays open.
    catalog_namespaces = {
        _namespace_of(term)
        for term in (*known_terms, *collect_catalog_terms(ontology_graph))
    }
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
    for subject, obj in graph.subject_objects(RDF.type):
        if not isinstance(obj, Literal):
            continue
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

    scope = FactScope.of(graph, normalized_fact_namespaces, document)
    # One finding per term and position: exit removes the uses in that one.
    flagged_uses: set[tuple[str, str]] = set()
    for used in used_terms(scope, contract):
        text = used.iri
        position = used.position
        if text in flagged_terms or (text, position) in flagged_uses:
            continue
        term = URIRef(text)
        verdict = term_verdict(
            text,
            catalog_terms=catalog_terms,
            known_terms=known_terms,
            fallback_terms=fallback_terms,
            declared_namespaces=declared_namespaces,
            catalog_namespaces=catalog_namespaces,
            standard_namespaces=standard_namespaces,
            vocabulary=policy.vocabulary,
            position=position,
            closed_range=used.closed_range,
        )
        if (
            verdict is not None
            and verdict.kind is TermVerdictKind.PLACEHOLDER_NAMESPACE
        ):
            flagged_uses.add((text, position))
            findings.append(
                FactsUnitFinding(
                    kind=FactsUnitFindingKind.UNKNOWN_TERM,
                    message=(
                        f"<{text}> uses the example.org placeholder namespace; "
                        "replace it with a catalog term or express the "
                        "statement with catalog/standard vocabulary."
                    ),
                    predicate=text,
                    position=position,
                    facet=verdict.kind.value,
                    suggestions=_containment_qualified(
                        term,
                        _alias_candidates(
                            term,
                            graph,
                            catalog_terms,
                            ontology_graph=ontology_graph,
                            position=position,
                        ),
                    )
                    if catalog_terms
                    else [],
                )
            )
            continue
        if position != "object" and any(
            text.startswith(ns) for ns in normalized_fact_namespaces
        ):
            flagged_uses.add((text, position))
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
                    position=position,
                    facet="fact_namespace",
                    suggestions=_containment_qualified(
                        term,
                        _alias_candidates(
                            term,
                            graph,
                            catalog_terms,
                            ontology_graph=ontology_graph,
                            position=position,
                        ),
                    )
                    if catalog_terms
                    else [],
                )
            )
            continue
        if verdict is None:
            continue
        flagged_uses.add((text, position))
        namespace = _namespace_of(text)
        if verdict.kind is TermVerdictKind.WELL_KNOWN_MISSING:
            # Written from memory: the prompt names the vocabulary, not
            # its terms, so a plausible term it does not have is caught
            # here, with near-misses from the vocabulary itself.
            message = (
                f"<{text}> is not a term of {verdict.vocabulary_title}; link "
                "a term that exists in it (or a suggested candidate), or "
                "leave the link out."
                if position == "object"
                else f"<{text}> is not a term of {verdict.vocabulary_title}; "
                "rewrite the statement IN PLACE with a term that "
                "exists in it (or a suggested candidate), or with a "
                "term from the ontology chapter, keeping the "
                "statement and its value."
            )
            pool: set[str] = well_known_terms(namespace)
        elif verdict.kind is TermVerdictKind.CLOSED_RANGE_MISSING:
            message = (
                f"<{text}> is neither an individual of the ontology nor a "
                "term of a listed vocabulary, and its property takes one of "
                "those: link one that exists, or leave the link out."
            )
            pool = catalog_terms
        elif verdict.kind is TermVerdictKind.CLOSED_NAMESPACE_MISSING:
            # A member of a closed namespace: a near-miss of a real term.
            message = (
                f"<{text}> does not exist in its ontology; rewrite "
                "the term IN PLACE to the closest correct term from "
                "the ontology chapter (or a suggested candidate), "
                "keeping the statement and its value. Do NOT delete "
                "the statement."
            )
            pool = catalog_terms
        else:
            # A vocabulary that is neither the catalog's, one it borrows
            # from, nor well-known. The strict vocabulary admits nothing
            # else, so the statement is re-expressed or left out.
            message = (
                f"<{text}> is not a term of the ontology nor of a "
                "well-known vocabulary, and the vocabulary is strict: "
                "rewrite the statement with a term from the ontology "
                "chapter or a well-known vocabulary, or leave the "
                "statement out if none can express it."
            )
            pool = catalog_terms
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.UNKNOWN_TERM,
                message=message,
                predicate=text,
                position=position,
                facet=verdict.kind.value,
                suggestions=_containment_qualified(
                    term,
                    _alias_candidates(
                        term,
                        graph,
                        pool,
                        ontology_graph=ontology_graph,
                        position=position,
                    ),
                ),
            )
        )

    findings.extend(_scalar_as_bounds_findings(scope, contract, ontology_graph))
    findings.extend(
        value_grounding_findings(scope, contract, extraction_text, ontology_graph)
    )
    sides_graph = (
        catalog_roles.with_axioms(ontology_graph or RDFGraph())
        if catalog_roles is not None
        else ontology_graph
    )
    findings.extend(domain_violation_findings(graph, sides_graph))
    findings.extend(role_outside_statement_findings(scope, contract, sides_graph))
    findings.extend(empty_value_findings(scope, contract))
    findings.extend(
        untyped_instance_findings(graph, ontology_graph, normalized_fact_namespaces)
    )
    if not is_citation_metadata and not is_non_content:
        findings.extend(
            missing_role_findings(graph, contract, normalized_fact_namespaces)
        )
        findings.extend(dimension_findings(scope, contract, ontology_graph))
        findings.extend(
            _domain_adherence_findings(
                graph,
                catalog_terms,
                normalized_fact_namespaces,
                policy.domain_adherence_min_share,
                min_terms=policy.domain_adherence_min_terms,
            )
        )
    findings.extend(
        _label_only_number_findings(
            scope,
            contract,
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
        )
    )

    if coverage_limit > 0:
        findings.extend(
            _numeric_coverage_findings(
                unit_numeric_inventory(
                    graph=graph,
                    ontology_graph=ontology_graph,
                    extraction_text=extraction_text,
                    policy=policy,
                    limit=coverage_limit,
                ),
                coverage_mode(coverage_mandatory),
            )
        )

    return findings


def dimension_findings(
    scope: FactScope,
    contract: StatementContract,
    ontology_graph: RDFGraph | None,
) -> list[FactsUnitFinding]:
    """Mandatory findings for results that disagree in dimension under an
    anchored property.

    A conflict (:func:`~ontocast.tool.agg.statement_conflict.dimension_conflicts`)
    is the unit's to resolve only when the catalog declares what the property
    measures (a quantity kind or an applicable measurement unit, through the
    contract's ``anchor_links``); the declaration says which result is the
    odd one, listed in ``suggestions`` when its dimension is known. Any other
    conflict is reported for the document by the facts gate. Inert without
    measurement units, and blind among dimensionless quantities (a ratio, an
    angle, a percentage share one dimension).
    """
    index = StatementIndex.of([scope.facts()], contract)
    return [
        FactsUnitFinding(
            kind=FactsUnitFindingKind.DIMENSION_CONFLICT,
            message=dimension_conflict_message(conflict),
            subject=str(conflict.node),
            value=", ".join(sorted(conflict.dimensions)),
            suggestions=[str(node) for node in conflict.odd],
        )
        for conflict in dimension_conflicts(index, ontology_graph, contract)
        if conflict.anchored
    ]


def _numeric_coverage_findings(
    inventory: NumericInventory, mode: CoverageMode
) -> list[FactsUnitFinding]:
    """The two coverage findings, measurements first.

    Advisory unless the mode says otherwise: the critic judges per mention
    whether it is a quantity or an artifact, and blocking forces that
    judgement into the loop. The measurement list carries each value's unit
    and the phrase it occurs in, which is what lets the completion pass place
    it; the bare-number list is what the text alone cannot classify.
    """
    findings: list[FactsUnitFinding] = []
    if inventory.measurements:
        listed = "; ".join(
            f"{mention.value} {mention.unit} («{mention.context}»)"
            for mention in inventory.measurements
        )
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.NUMERIC_COVERAGE,
                facet="measurements",
                mandatory=mode in ("measurements", "all"),
                message=(
                    "These measurements — a number written with its unit — "
                    "appear in the source text but not in the graph. Each is a "
                    "stated quantity unless it is typography: extract it as a "
                    "typed literal on the quantity it measures, with the "
                    "verbatim value and the source unit (never convert): " + listed
                ),
                value=", ".join(inventory.measurement_values()),
            )
        )
    if inventory.unclassified:
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.NUMERIC_COVERAGE,
                facet="unclassified",
                mandatory=mode == "all",
                message=(
                    "These numbers appear in the source text but not in the "
                    "graph. For each: extract it as a typed literal on an "
                    "appropriate node (verbatim value and source unit — never "
                    "convert units) if it is a factual quantity, or ignore it "
                    "if it is a page/citation/figure artifact: "
                    + ", ".join(inventory.unclassified)
                ),
                value=", ".join(inventory.unclassified),
            )
        )
    return findings
