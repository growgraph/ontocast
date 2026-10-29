"""Validation tools for OntoCast.

This module provides functionality for validating RDF graphs and chunks,
including connectivity validation and graph structure verification.
"""

import logging
from collections import defaultdict, deque

from pydantic import BaseModel, ConfigDict, Field
from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple

logger = logging.getLogger(__name__)

# Ranges compatible with literal objects even though they are not XSD datatypes.
_LITERAL_COMPATIBLE_RANGES: frozenset[URIRef] = frozenset({RDFS.Literal, RDFS.Resource})


def _predicate_expects_iri_object(
    predicate: URIRef, ontology_graph: Graph
) -> tuple[bool, str | None]:
    """Decide from the schema whether *predicate* requires an IRI object.

    True when the predicate is declared ``owl:ObjectProperty``, or its
    ``rdfs:range`` is a class IRI (not XSD, not ``rdfs:Literal``/``Resource``,
    not a declared ``rdfs:Datatype``). Predicates also declared datatype or
    annotation properties, or absent from the schema, never expect an IRI —
    unknown predicates must not be over-quarantined.

    Returns:
        Tuple of (expects IRI, declared range IRI or None).
    """
    types = set(ontology_graph.objects(predicate, RDF.type))
    if OWL.DatatypeProperty in types or OWL.AnnotationProperty in types:
        return False, None
    declared_range = next(
        (
            r
            for r in ontology_graph.objects(predicate, RDFS.range)
            if isinstance(r, URIRef)
        ),
        None,
    )
    if OWL.ObjectProperty in types:
        return True, str(declared_range) if declared_range else None
    if declared_range is None:
        return False, None
    if str(declared_range).startswith(str(XSD)):
        return False, None
    if declared_range in _LITERAL_COMPATIBLE_RANGES:
        return False, None
    if (declared_range, RDF.type, RDFS.Datatype) in ontology_graph:
        return False, None
    return True, str(declared_range)


def partition_object_property_literal_triples(
    facts_graph: RDFGraph, ontology_graph: Graph
) -> tuple[RDFGraph, list[RejectedLiteralTriple]]:
    """Split facts into a clean graph and quarantined literal-on-object-property triples.

    A string literal on a predicate whose schema range is a class (e.g.
    ``qudt:unit`` with range ``qudt:Unit``) silently breaks typing: the value can
    never be linked to the reference individual it names. Such triples are
    quarantined with a reason so the critic loop can ask the renderer to resolve
    the token to an IRI from the ontology context instead.

    Args:
        facts_graph: Freshly rendered facts graph.
        ontology_graph: Ontology context the facts were rendered against.

    Returns:
        Tuple of (clean graph, quarantined triples).
    """
    clean = RDFGraph()
    for prefix, namespace_uri in facts_graph.namespaces():
        if prefix:
            clean.bind(prefix, namespace_uri)

    rejected: list[RejectedLiteralTriple] = []
    expects_cache: dict[URIRef, tuple[bool, str | None]] = {}
    for subject, predicate, obj in facts_graph:
        if isinstance(obj, Literal) and isinstance(predicate, URIRef):
            if predicate not in expects_cache:
                expects_cache[predicate] = _predicate_expects_iri_object(
                    predicate, ontology_graph
                )
            expects_iri, declared_range = expects_cache[predicate]
            if expects_iri:
                rejected.append(
                    RejectedLiteralTriple(
                        subject=str(subject),
                        predicate=str(predicate),
                        object_lexical=str(obj),
                        datatype=str(obj.datatype) if obj.datatype else "",
                        reason=(
                            "object property expects an IRI from the ontology "
                            "context, got a string literal"
                        ),
                        expected_range=declared_range,
                    )
                )
                continue
        clean.add((subject, predicate, obj))

    if rejected:
        logger.warning(
            "Quarantined %d literal triple(s) on object properties", len(rejected)
        )
    return clean, rejected


class PredicateStats(BaseModel):
    """Type definition for predicate statistics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total: int = 0
    with_labels: int = 0
    with_domains: int = 0
    with_ranges: int = 0


class PredicateValidationResult(BaseModel):
    """Type definition for predicate validation results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    has_required_properties: bool = True
    domain_range_consistent: bool = True
    missing_labels: list[str] = Field(default_factory=list)
    domain_range_violations: list[str] = Field(default_factory=list)
    predicate_stats: PredicateStats = Field(default_factory=PredicateStats)


class ConnectivityResult(BaseModel):
    """Type definition for connectivity validation results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_fully_connected: bool = False
    num_components: int = 0
    total_entities: int = 0
    components: list[set[URIRef]] = Field(default_factory=list)
    isolated_entities: list[URIRef] = Field(default_factory=list)
    largest_component_size: int = 0
    has_required_properties: bool = True
    domain_range_consistent: bool = True
    missing_labels: list[str] = Field(default_factory=list)
    domain_range_violations: list[str] = Field(default_factory=list)
    predicate_stats: PredicateStats = Field(default_factory=PredicateStats)


class RDFGraphConnectivityValidator:
    """Validator for RDF graph connectivity.

    This class provides functionality for validating and ensuring connectivity
    in RDF graphs, including finding connected components and adding bridging
    relationships.

    Attributes:
        graph: The RDF graph to validate.
    """

    def __init__(self, graph: RDFGraph):
        """Initialize the validator.

        Args:
            graph: The RDF graph to validate.
        """
        self.graph = graph

    def get_all_entities(self) -> set[URIRef]:
        """Extract all unique entities from the graph.

        Returns:
            set[URIRef]: Set of all unique entity URIs in the graph.
        """
        entities = set()

        for subj, _, obj in self.graph:
            if isinstance(subj, URIRef):
                entities.add(subj)
            if isinstance(obj, URIRef):
                entities.add(obj)

        return entities

    def build_adjacency_graph(self) -> dict[URIRef, set[URIRef]]:
        """Build an adjacency representation of the RDF graph.

        Returns:
            dict[URIRef, set[URIRef]]: Dictionary mapping entities to their neighbors.
        """
        adjacency = defaultdict(set)

        for subj, _, obj in self.graph:
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                adjacency[subj].add(obj)
                adjacency[obj].add(subj)  # Treat as undirected for connectivity

        return adjacency

    def find_connected_components(self) -> list[set[URIRef]]:
        """Find all connected components in the graph using BFS.

        Returns:
            list[set[URIRef]]: List of sets, each containing entities in a component.
        """
        entities = self.get_all_entities()
        adjacency = self.build_adjacency_graph()
        visited = set()
        components = []

        for entity in entities:
            if entity not in visited:
                component = set()
                queue = deque([entity])

                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)

                        # Add neighbors to queue
                        for neighbor in adjacency.get(current, set()):
                            if neighbor not in visited:
                                queue.append(neighbor)

                if component:
                    components.append(component)

        return components

    def validate_predicates(self) -> PredicateValidationResult:
        """Validate predicate consistency and required properties.

        Returns:
            PredicateValidationResult: Pydantic model containing validation results and statistics.
        """
        result = PredicateValidationResult()

        # Track all predicates
        predicates = set()
        for _, pred, _ in self.graph:
            if isinstance(pred, URIRef):
                predicates.add(pred)

        result.predicate_stats.total = len(predicates)

        # Check each predicate
        for pred in predicates:
            has_label = False
            has_domain = False
            has_range = False
            domain = None
            range_ = None

            # Get predicate properties
            for s, p, o in self.graph:
                if s == pred:
                    if p == RDFS.label:
                        has_label = True
                        result.predicate_stats.with_labels += 1
                    elif p == RDFS.domain:
                        has_domain = True
                        domain = o
                        result.predicate_stats.with_domains += 1
                    elif p == RDFS.range:
                        has_range = True
                        range_ = o
                        result.predicate_stats.with_ranges += 1

            # Check required properties
            if not has_label:
                result.has_required_properties = False
                result.missing_labels.append(str(pred))

            # Check domain/range consistency in usage
            if has_domain or has_range:
                for s, p, o in self.graph:
                    if p == pred:
                        if has_domain and isinstance(s, URIRef):
                            # Check if subject is of correct domain type
                            subject_type = None
                            for s2, p2, o2 in self.graph:
                                if s2 == s and p2 == RDF.type:
                                    subject_type = o2
                                    break

                            if subject_type and domain and subject_type != domain:
                                result.domain_range_consistent = False
                                result.domain_range_violations.append(
                                    f"Subject {s} of type {subject_type} "
                                    f"used with predicate {pred} "
                                    f"that requires domain {domain}"
                                )

                        if has_range and isinstance(o, URIRef):
                            # Check if object is of correct range type
                            object_type = None
                            for s2, p2, o2 in self.graph:
                                if s2 == o and p2 == RDF.type:
                                    object_type = o2
                                    break

                            if object_type and range_ and object_type != range_:
                                result.domain_range_consistent = False
                                result.domain_range_violations.append(
                                    f"Object {o} of type {object_type} "
                                    f"used with predicate {pred} "
                                    f"that requires range {range_}"
                                )

                        if has_range and isinstance(o, Literal):
                            # A literal can never satisfy a class range.
                            range_is_class = (
                                isinstance(range_, URIRef)
                                and not str(range_).startswith(str(XSD))
                                and range_ not in _LITERAL_COMPATIBLE_RANGES
                                and (range_, RDF.type, RDFS.Datatype) not in self.graph
                            )
                            if range_is_class:
                                result.domain_range_consistent = False
                                result.domain_range_violations.append(
                                    f"Literal object {str(o)!r} used with "
                                    f"predicate {pred} whose range {range_} "
                                    f"is a class (IRI expected)"
                                )

        return result

    def validate_connectivity(self) -> ConnectivityResult:
        """Validate graph connectivity and return detailed results.

        Returns:
            ConnectivityResult: Pydantic model containing connectivity information and
                validation results.
        """
        components = self.find_connected_components()
        entities = self.get_all_entities()

        result = ConnectivityResult(
            is_fully_connected=len(components) <= 1,
            num_components=len(components),
            total_entities=len(entities),
            components=components,
        )

        if components:
            result.largest_component_size = max(len(comp) for comp in components)

            # Find isolated entities (components of size 1)
            result.isolated_entities = [
                list(comp)[0] for comp in components if len(comp) == 1
            ]

        # Add predicate validation results
        predicate_validation = self.validate_predicates()
        result.has_required_properties = predicate_validation.has_required_properties
        result.domain_range_consistent = predicate_validation.domain_range_consistent
        result.missing_labels = predicate_validation.missing_labels
        result.domain_range_violations = predicate_validation.domain_range_violations
        result.predicate_stats = predicate_validation.predicate_stats

        return result
