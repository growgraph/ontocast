"""Validation tools for OntoCast.

This module provides functionality for validating RDF graphs and chunks,
including connectivity validation and graph structure verification.
"""

import logging
from collections import defaultdict, deque
from typing import Any, Optional, Set

from rdflib import RDF, RDFS, Literal, URIRef

from ontocast.onto.chunk import Chunk
from ontocast.onto.constants import PROV, SCHEMA
from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)


def validate_and_connect_chunk(
    chunk: Chunk,
    auto_connect: bool = True,
) -> Chunk:
    """Validate and optionally connect a chunk graph.

    This function validates the connectivity of a chunk's RDF graph and
    optionally connects any disconnected components.

    Args:
        chunk: The chunk containing the RDF graph to validate.
        auto_connect: Whether to automatically connect disconnected graphs.

    Returns:
        Chunk: The chunk with a validated and optionally connected graph.
    """

    # Ensure an RDFGraph instance
    if not isinstance(chunk.graph, RDFGraph):
        logger.warning("received an redflib.Graph rather than RDFGraph")
        new_graph = RDFGraph()
        for triple in chunk.graph:
            new_graph.add(triple)
        for prefix, namespace in chunk.graph.namespaces():
            new_graph.bind(prefix, namespace)
        chunk.graph = new_graph

    validator = RDFGraphConnectivityValidator(chunk.graph)

    result = validator.validate_connectivity()

    logger.debug(f"\n=== Connectivity Analysis for Chunk {chunk.iri} ===")
    logger.debug(f"Fully connected: {result['is_fully_connected']}")
    logger.debug(f"Number of components: {result['num_components']}")
    logger.debug(f"Total entities: {result['total_entities']}")
    logger.debug(f"Largest component size: {result['largest_component_size']}")

    if result["isolated_entities"]:
        logger.debug(
            f"Isolated entities: {[str(e) for e in result['isolated_entities']]}"
        )

    # Create a new RDFGraph instance instead of using deepcopy
    final_graph = RDFGraph()
    for triple in chunk.graph:
        final_graph.add(triple)
    # Copy namespace bindings
    for prefix, namespace in chunk.graph.namespaces():
        final_graph.bind(prefix, namespace)

    if not result["is_fully_connected"] and auto_connect:
        final_graph = validator.make_graph_connected(chunk.iri)

    chunk.graph = final_graph
    return chunk


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

    def get_all_entities(self) -> Set[URIRef]:
        """Extract all unique entities from the graph.

        Returns:
            Set[URIRef]: Set of all unique entity URIs in the graph.
        """
        entities = set()

        for subj, _, obj in self.graph:
            if isinstance(subj, URIRef):
                entities.add(subj)
            if isinstance(obj, URIRef):
                entities.add(obj)

        return entities

    def build_adjacency_graph(self) -> dict[URIRef, Set[URIRef]]:
        """Build an adjacency representation of the RDF graph.

        Returns:
            dict[URIRef, Set[URIRef]]: Dictionary mapping entities to their neighbors.
        """
        adjacency = defaultdict(set)

        for subj, _, obj in self.graph:
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                adjacency[subj].add(obj)
                adjacency[obj].add(subj)  # Treat as undirected for connectivity

        return adjacency

    def find_connected_components(self) -> list[Set[URIRef]]:
        """Find all connected components in the graph using BFS.

        Returns:
            list[Set[URIRef]]: List of sets, each containing entities in a component.
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

    def validate_predicates(self) -> dict[str, Any]:
        """Validate predicate consistency and required properties.

        Returns:
            dict[str, Any]: Dictionary containing validation results and statistics.
        """
        result = {
            "has_required_properties": True,
            "domain_range_consistent": True,
            "missing_labels": [],
            "domain_range_violations": [],
            "predicate_stats": {
                "total": 0,
                "with_labels": 0,
                "with_domains": 0,
                "with_ranges": 0,
            },
        }

        # Track all predicates
        predicates = set()
        for _, pred, _ in self.graph:
            if isinstance(pred, URIRef):
                predicates.add(pred)

        result["predicate_stats"]["total"] = len(predicates)

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
                        result["predicate_stats"]["with_labels"] += 1
                    elif p == RDFS.domain:
                        has_domain = True
                        domain = o
                        result["predicate_stats"]["with_domains"] += 1
                    elif p == RDFS.range:
                        has_range = True
                        range_ = o
                        result["predicate_stats"]["with_ranges"] += 1

            # Check required properties
            if not has_label:
                result["has_required_properties"] = False
                result["missing_labels"].append(str(pred))

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
                                result["domain_range_consistent"] = False
                                result["domain_range_violations"].append(
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
                                result["domain_range_consistent"] = False
                                result["domain_range_violations"].append(
                                    f"Object {o} of type {object_type} "
                                    f"used with predicate {pred} "
                                    f"that requires range {range_}"
                                )

        return result

    def validate_connectivity(self) -> dict[str, Any]:
        """Validate graph connectivity and return detailed results.

        Returns:
            dict[str, Any]: Dictionary containing connectivity information and
                validation results.
        """
        components = self.find_connected_components()
        entities = self.get_all_entities()

        result = {
            "is_fully_connected": len(components) <= 1,
            "num_components": len(components),
            "total_entities": len(entities),
            "components": components,
            "isolated_entities": [],
            "largest_component_size": 0,
        }

        if components:
            result["largest_component_size"] = max(len(comp) for comp in components)

            # Find isolated entities (components of size 1)
            result["isolated_entities"] = [
                list(comp)[0] for comp in components if len(comp) == 1
            ]

        # Add predicate validation results
        predicate_validation = self.validate_predicates()
        result.update(predicate_validation)

        return result

    def make_graph_connected(self, chunk_iri) -> RDFGraph:
        """Make a disconnected graph connected by adding bridging relationships.

        Args:
            chunk_iri: The IRI of the chunk to use for the hub entity.

        Returns:
            RDFGraph: A new connected graph.
        """
        components = self.find_connected_components()

        if len(components) <= 1:
            logger.info("RDFGraph is already connected")
            return self.graph

        # Create a new graph with all original triples
        connected_graph = RDFGraph()
        for triple in self.graph:
            connected_graph.add(triple)

        # Copy namespace bindings
        for prefix, namespace in self.graph.namespaces():
            connected_graph.bind(prefix, namespace)

        connected_graph = self._connect_via_chunk_hub(
            connected_graph, components, chunk_iri
        )

        logger.info(f"Connected {len(components)} components")
        return connected_graph

    def _connect_via_chunk_hub(
        self, graph: RDFGraph, components: list[Set[URIRef]], chunk_iri
    ) -> RDFGraph:
        """Connect components by creating a chunk hub entity.

        Args:
            graph: The graph to modify.
            components: List of connected components to connect.
            chunk_iri: The IRI to use for the hub entity.

        Returns:
            RDFGraph: The modified graph with connected components.
        """
        # Create or use existing chunk URI
        hub_uri = URIRef(chunk_iri)
        hub_id = hub_uri.split("/")[-1]

        # Add hub entity metadata
        graph.add((hub_uri, RDF.type, SCHEMA.TextDigitalDocument))
        graph.add((hub_uri, RDFS.label, Literal(f"Chunk {hub_id}")))

        # Connect hub to one representative entity from each component
        for i, component in enumerate(components):
            # Choose representative entity (could be improved with better heuristics)
            representative = self._choose_representative_entity(component, graph)

            # Add bidirectional connections
            graph.add((hub_uri, SCHEMA.hasPart, representative))
            graph.add((representative, PROV.wasQuotedFrom, hub_uri))

        return graph

    def _choose_representative_entity(
        self, component: Set[URIRef], graph: RDFGraph
    ) -> Optional[URIRef]:
        """Choose the best representative entity from a component.

        Args:
            component: Set of entities in the component.
            graph: The RDF graph containing the entities.

        Returns:
            Optional[URIRef]: The chosen representative entity, or None if empty.
        """
        if not component:
            return None

        entity_degrees = {}
        entities_with_labels = set()

        for entity in component:
            # Count connections
            degree = sum(1 for s, p, o in graph if s == entity or o == entity)
            entity_degrees[entity] = degree

            # Check if entity has a label
            for s, p, o in graph:
                if s == entity and p in [RDFS.label, RDFS.comment]:
                    entities_with_labels.add(entity)
                    break

        # Prefer entities with labels and high degree
        if entities_with_labels:
            return max(entities_with_labels, key=lambda e: entity_degrees.get(e, 0))
        else:
            return max(component, key=lambda e: entity_degrees.get(e, 0))
