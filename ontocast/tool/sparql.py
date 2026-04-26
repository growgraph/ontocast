"""SPARQL tool for incremental graph updates.

This module provides functionality for executing SPARQL operations on RDF graphs,
enabling incremental updates instead of full graph replacement.
"""

import asyncio
import logging

from rdflib import BNode, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS
from rdflib.plugins.sparql import prepareQuery

from ontocast.onto.enum import SPARQLOperationType
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import SPARQLOperationModel
from ontocast.tool.triple_manager.core import TripleStoreManager

logger = logging.getLogger(__name__)

# Predicates treated as human-facing descriptions for seed entities (always merged in first).
_SEED_DESCRIPTION_PREDICATES: frozenset[URIRef] = frozenset(
    {
        RDFS.label,
        RDFS.comment,
        SKOS.prefLabel,
        SKOS.altLabel,
        SKOS.definition,
        URIRef("http://purl.org/dc/terms/description"),
        URIRef("http://purl.org/dc/elements/1.1/description"),
    }
)

# RDF list expansion (`rdf:first`/`rdf:rest`) tends to introduce many low-value
# blank-node triples that are disconnected from business entities.
_NOISY_EXPANSION_PREDICATES: frozenset[URIRef] = frozenset({RDF.first, RDF.rest})


class SPARQLTool:
    """Tool for executing SPARQL operations on RDF graphs."""

    def __init__(self, triple_store_manager: TripleStoreManager | None = None):
        """Initialize SPARQL tool.

        Args:
            triple_store_manager: Optional triple store manager for persistent storage.
        """
        self.triple_store_manager = triple_store_manager
        self.operation_history = []

    def execute_operations(
        self, graph: RDFGraph, operations: list[SPARQLOperationModel]
    ) -> RDFGraph:
        """Execute a list of SPARQL operations on a graph.

        Args:
            graph: The RDF graph to operate on.
            operations: List of SPARQL operations to execute.

        Returns:
            RDFGraph: Updated graph after applying operations.
        """
        logger.info(f"Executing {len(operations)} SPARQL operations")

        for operation in operations:
            try:
                self._execute_single_operation(graph, operation)
                self.operation_history.append(operation)
                logger.debug(
                    f"Executed {operation.operation_type} operation: {operation.description}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to execute {operation.operation_type} operation: {str(e)}"
                )
                raise

        return graph

    def execute_operation(self, operation: SPARQLOperationModel) -> None:
        """Execute a single SPARQL operation.

        Args:
            operation: The SPARQL operation to execute.
        """
        # For now, we'll use a simple approach - in a real implementation,
        # you might want to track which graph this operation should be applied to
        logger.info(
            f"Executing {operation.operation_type} operation: {operation.description}"
        )
        # This is a placeholder - in practice, you'd need to specify which graph to operate on
        # or maintain a default graph in the tool

    def _execute_single_operation(
        self, graph: RDFGraph, operation: SPARQLOperationModel
    ):
        """Execute a single SPARQL operation.

        Args:
            graph: The RDF graph to operate on.
            operation: The SPARQL operation to execute.
        """
        if operation.operation_type == SPARQLOperationType.INSERT:
            self._execute_insert(graph, operation)
        elif operation.operation_type == SPARQLOperationType.DELETE:
            self._execute_delete(graph, operation)
        elif operation.operation_type == SPARQLOperationType.UPDATE:
            self._execute_update(graph, operation)
        else:
            raise ValueError(f"Unknown operation type: {operation.operation_type}")

    def _execute_insert(self, graph: RDFGraph, operation: SPARQLOperationModel):
        """Execute INSERT operation.

        Args:
            graph: The RDF graph to operate on.
            operation: The INSERT operation to execute.
        """
        # Parse the INSERT query
        query = prepareQuery(operation.query)

        # For INSERT DATA, we need to parse the triples and add them to the graph
        if "INSERT DATA" in operation.query.upper():
            # Extract triples from INSERT DATA query
            triples = self._parse_insert_data_triples(operation.query)
            for triple in triples:
                graph.add(triple)
        else:
            # For other INSERT queries, execute against the graph
            graph.query(query)
            # INSERT queries typically don't return results, but we execute them

    def _execute_delete(self, graph: RDFGraph, operation: SPARQLOperationModel):
        """Execute DELETE operation.

        Args:
            graph: The RDF graph to operate on.
            operation: The DELETE operation to execute.
        """
        # Parse the DELETE query
        query = prepareQuery(operation.query)

        # For DELETE DATA, we need to parse the triples and remove them from the graph
        if "DELETE DATA" in operation.query.upper():
            # Extract triples from DELETE DATA query
            triples = self._parse_delete_data_triples(operation.query)
            for triple in triples:
                graph.remove(triple)
        else:
            # For other DELETE queries, execute against the graph
            graph.query(query)
            # DELETE queries typically don't return results, but we execute them

    def _execute_update(self, graph: RDFGraph, operation: SPARQLOperationModel):
        """Execute UPDATE operation.

        Args:
            graph: The RDF graph to operate on.
            operation: The UPDATE operation to execute.
        """
        # Parse the UPDATE query
        query = prepareQuery(operation.query)

        # Execute the UPDATE query
        graph.query(query)
        # UPDATE queries typically don't return results, but we execute them

    def _parse_insert_data_triples(self, query: str) -> list[tuple]:
        """Parse triples from INSERT DATA query.

        Args:
            query: The INSERT DATA query string.

        Returns:
            List of triples to insert.
        """
        # This is a simplified parser - in practice, you'd want a more robust parser
        triples = []

        # Extract the content between INSERT DATA { ... }
        start = query.upper().find("INSERT DATA {")
        if start == -1:
            return triples

        start += len("INSERT DATA {")
        end = query.rfind("}")

        if end == -1:
            return triples

        data_content = query[start:end].strip()

        # Split by lines and parse each triple
        lines = [line.strip() for line in data_content.split("\n") if line.strip()]

        for line in lines:
            if line.endswith("."):
                line = line[:-1]  # Remove trailing period

            # Parse the triple (simplified - assumes standard N3 format)
            parts = line.split()
            if len(parts) >= 3:
                subject = self._parse_term(parts[0])
                predicate = self._parse_term(parts[1])
                object_part = self._parse_term(" ".join(parts[2:]))

                if subject and predicate and object_part:
                    triples.append((subject, predicate, object_part))

        return triples

    def _parse_delete_data_triples(self, query: str) -> list[tuple]:
        """Parse triples from DELETE DATA query.

        Args:
            query: The DELETE DATA query string.

        Returns:
            List of triples to delete.
        """
        # Similar to INSERT DATA parsing
        return self._parse_insert_data_triples(
            query.replace("DELETE DATA", "INSERT DATA")
        )

    def _parse_term(self, term: str):
        """Parse a SPARQL term (subject, predicate, or object).

        Args:
            term: The term string to parse.

        Returns:
            Parsed RDF term (URIRef, Literal, or BNode).
        """
        term = term.strip()

        if term.startswith("<") and term.endswith(">"):
            # URI
            return URIRef(term[1:-1])
        elif term.startswith('"') and term.endswith('"'):
            # Literal
            return Literal(term[1:-1])
        elif term.startswith("_:"):
            # Blank node
            return BNode(term[2:])
        elif term.startswith('"') and '"^^' in term:
            # Typed literal
            value, datatype = term.split('"^^')
            return Literal(value[1:], datatype=URIRef(datatype))
        else:
            # Assume it's a URI without angle brackets
            return URIRef(term)

    def validate_operation(self, operation: SPARQLOperationModel) -> bool:
        """Validate a SPARQL operation.

        Args:
            operation: The operation to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            prepareQuery(operation.query)
            return True
        except Exception as e:
            logger.error(f"Invalid SPARQL operation: {str(e)}")
            return False

    def get_operation_history(self) -> list[SPARQLOperationModel]:
        """Get the history of executed operations.

        Returns:
            List of executed operations.
        """
        return self.operation_history.copy()

    def clear_history(self):
        """Clear the operation history."""
        self.operation_history.clear()

    @staticmethod
    def _build_induced_subgraph(
        ontologies: list[Ontology],
        entity_uris: list[str],
        entity_relevance: dict[str, float] | None,
        ontology_iris: list[str] | None,
        depth: int,
        max_total_triples: int,
        estimated_triples_per_query: int,
        ontology_version_filters: dict[str, set[str]] | None,
        ontology_hash_filters: dict[str, set[str]] | None,
    ) -> RDFGraph:
        """Merge filtered ontology graphs; return a budgeted relevance-weighted neighborhood."""

        def should_include_expansion_triple(
            subj: object,
            pred: object,
            obj: object,
        ) -> bool:
            if not isinstance(pred, URIRef):
                return False
            if pred in _NOISY_EXPANSION_PREDICATES:
                return False
            if isinstance(subj, BNode) and isinstance(obj, BNode):
                return False
            return True

        ontology_filter = set(ontology_iris or [])
        relevant_graphs: list[RDFGraph] = []
        for ontology in ontologies:
            if ontology_filter and ontology.iri not in ontology_filter:
                continue
            if ontology_version_filters and ontology.iri in ontology_version_filters:
                ontology_version = (
                    str(ontology.version) if ontology.version is not None else None
                )
                if ontology_version not in ontology_version_filters[ontology.iri]:
                    continue
            if ontology_hash_filters and ontology.iri in ontology_hash_filters:
                if ontology.hash not in ontology_hash_filters[ontology.iri]:
                    continue
            relevant_graphs.append(ontology.graph)
        if not relevant_graphs:
            return RDFGraph()

        merged_graph = RDFGraph()
        for graph in relevant_graphs:
            for prefix, namespace in graph.namespaces():
                if prefix:
                    merged_graph.bind(prefix, namespace)
            merged_graph += graph

        if not entity_uris:
            return RDFGraph()
        seed_uris_ranked = list(dict.fromkeys(uri for uri in entity_uris if uri))
        if not seed_uris_ranked:
            return RDFGraph()
        result = RDFGraph()
        for prefix, namespace in merged_graph.namespaces():
            if prefix:
                result.bind(prefix, namespace)

        if max_total_triples <= 0 or estimated_triples_per_query <= 0:
            return result

        relevance = entity_relevance or {}
        sorted_seed_uris = sorted(
            seed_uris_ranked,
            key=lambda uri: (-float(relevance.get(uri, 0.0)), uri),
        )

        score_by_seed: dict[str, float] = {
            uri: float(relevance.get(uri, 0.0)) for uri in sorted_seed_uris
        }
        score_total = sum(max(score, 0.0) for score in score_by_seed.values())
        if score_total <= 0.0:
            score_by_seed = {uri: 1.0 for uri in sorted_seed_uris}
            score_total = float(len(sorted_seed_uris))

        quotas: dict[str, int] = {uri: 0 for uri in sorted_seed_uris}
        per_entity_cap = max(1, estimated_triples_per_query)
        remaining = max_total_triples

        for uri in sorted_seed_uris:
            if remaining <= 0:
                break
            quotas[uri] = 1
            remaining -= 1

        if remaining > 0:
            for uri in sorted_seed_uris:
                if remaining <= 0:
                    break
                weight = max(score_by_seed[uri], 0.0) / score_total
                extra = int(remaining * weight)
                extra = min(extra, per_entity_cap - quotas[uri])
                if extra <= 0:
                    continue
                quotas[uri] += extra
                remaining -= extra

        if remaining > 0:
            for uri in sorted_seed_uris:
                if remaining <= 0:
                    break
                if quotas[uri] >= per_entity_cap:
                    continue
                quotas[uri] += 1
                remaining -= 1

        def _candidate_triples(seed: URIRef) -> list[tuple]:
            candidates: list[tuple] = []
            seen: set[tuple] = set()

            def append_candidate(triple: tuple) -> None:
                subj, pred, obj = triple
                if not should_include_expansion_triple(subj, pred, obj):
                    return
                if triple in seen:
                    return
                seen.add(triple)
                candidates.append(triple)

            for pred in _SEED_DESCRIPTION_PREDICATES:
                outgoing = sorted(
                    merged_graph.triples((seed, pred, None)),
                    key=lambda triple: str(triple),
                )
                incoming = sorted(
                    merged_graph.triples((None, pred, seed)),
                    key=lambda triple: str(triple),
                )
                for triple in outgoing + incoming:
                    append_candidate(triple)

            frontier: set[URIRef] = {seed}
            visited: set[URIRef] = set()
            for _ in range(depth + 1):
                if not frontier:
                    break
                next_frontier: set[URIRef] = set()
                for node in sorted(frontier, key=lambda value: str(value)):
                    if node in visited:
                        continue
                    visited.add(node)

                    outgoing = sorted(
                        merged_graph.triples((node, None, None)),
                        key=lambda triple: str(triple),
                    )
                    incoming = sorted(
                        merged_graph.triples((None, None, node)),
                        key=lambda triple: str(triple),
                    )
                    for triple in outgoing + incoming:
                        subj, pred, obj = triple
                        if not should_include_expansion_triple(subj, pred, obj):
                            continue
                        append_candidate(triple)
                        if isinstance(subj, URIRef) and subj not in visited:
                            next_frontier.add(subj)
                        if isinstance(obj, URIRef) and obj not in visited:
                            next_frontier.add(obj)
                frontier = next_frontier
            return candidates

        candidates_by_seed: dict[str, list[tuple]] = {}
        for seed_uri in sorted_seed_uris:
            candidates_by_seed[seed_uri] = _candidate_triples(URIRef(seed_uri))

        for seed_uri in sorted_seed_uris:
            if len(result) >= max_total_triples:
                break
            quota = quotas.get(seed_uri, 0)
            if quota <= 0:
                continue
            selected = 0
            for triple in candidates_by_seed.get(seed_uri, []):
                if len(result) >= max_total_triples:
                    break
                if triple in result:
                    continue
                result.add(triple)
                selected += 1
                if selected >= quota:
                    break

        if len(result) >= max_total_triples:
            return result

        for seed_uri in sorted_seed_uris:
            for triple in candidates_by_seed.get(seed_uri, []):
                if len(result) >= max_total_triples:
                    return result
                if triple in result:
                    continue
                result.add(triple)
        return result

    def get_induced_subgraph(
        self,
        entity_uris: list[str],
        entity_relevance: dict[str, float] | None = None,
        ontology_iris: list[str] | None = None,
        depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
        ontology_version_filters: dict[str, set[str]] | None = None,
        ontology_hash_filters: dict[str, set[str]] | None = None,
    ) -> RDFGraph:
        """Fetch a deterministic induced subgraph around selected entities."""
        if self.triple_store_manager is None:
            return RDFGraph()
        if depth < 0:
            raise ValueError("depth must be >= 0")
        if max_total_triples <= 0:
            return RDFGraph()
        if estimated_triples_per_query <= 0:
            return RDFGraph()

        ontologies = self.triple_store_manager.fetch_ontologies()
        return self._build_induced_subgraph(
            ontologies,
            entity_uris,
            entity_relevance,
            ontology_iris,
            depth,
            max_total_triples,
            estimated_triples_per_query,
            ontology_version_filters,
            ontology_hash_filters,
        )

    async def aget_induced_subgraph(
        self,
        entity_uris: list[str],
        entity_relevance: dict[str, float] | None = None,
        ontology_iris: list[str] | None = None,
        depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
        ontology_version_filters: dict[str, set[str]] | None = None,
        ontology_hash_filters: dict[str, set[str]] | None = None,
    ) -> RDFGraph:
        """Like ``get_induced_subgraph`` but uses ``afetch_ontologies`` for I/O."""
        if self.triple_store_manager is None:
            return self.get_induced_subgraph(
                entity_uris=entity_uris,
                entity_relevance=entity_relevance,
                ontology_iris=ontology_iris,
                depth=depth,
                max_total_triples=max_total_triples,
                estimated_triples_per_query=estimated_triples_per_query,
                ontology_version_filters=ontology_version_filters,
                ontology_hash_filters=ontology_hash_filters,
            )
        if depth < 0:
            raise ValueError("depth must be >= 0")
        if max_total_triples <= 0:
            return RDFGraph()
        if estimated_triples_per_query <= 0:
            return RDFGraph()

        ontologies = await self.triple_store_manager.afetch_ontologies()
        return await asyncio.to_thread(
            SPARQLTool._build_induced_subgraph,
            ontologies,
            entity_uris,
            entity_relevance,
            ontology_iris,
            depth,
            max_total_triples,
            estimated_triples_per_query,
            ontology_version_filters,
            ontology_hash_filters,
        )

    def create_insert_operation(
        self, query: str, description: str = ""
    ) -> SPARQLOperationModel:
        """Create an INSERT operation.

        Args:
            query: The SPARQL INSERT query.
            description: Optional description of the operation.

        Returns:
            SPARQLOperationModel: The created operation.
        """
        return SPARQLOperationModel(
            operation_type=SPARQLOperationType.INSERT,
            query=query,
            description=description,
        )

    def create_delete_operation(
        self, query: str, description: str = ""
    ) -> SPARQLOperationModel:
        """Create a DELETE operation.

        Args:
            query: The SPARQL DELETE query.
            description: Optional description of the operation.

        Returns:
            SPARQLOperationModel: The created operation.
        """
        return SPARQLOperationModel(
            operation_type=SPARQLOperationType.DELETE,
            query=query,
            description=description,
        )

    def create_update_operation(
        self, query: str, description: str = ""
    ) -> SPARQLOperationModel:
        """Create an UPDATE operation.

        Args:
            query: The SPARQL UPDATE query.
            description: Optional description of the operation.

        Returns:
            SPARQLOperationModel: The created operation.
        """
        return SPARQLOperationModel(
            operation_type=SPARQLOperationType.UPDATE,
            query=query,
            description=description,
        )
