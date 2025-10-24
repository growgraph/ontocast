"""SPARQL tool for incremental graph updates.

This module provides functionality for executing SPARQL operations on RDF graphs,
enabling incremental updates instead of full graph replacement.
"""

import logging

from rdflib import BNode, Literal, URIRef
from rdflib.plugins.sparql import prepareQuery

from ontocast.onto.enum import SPARQLOperationType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import SPARQLOperationModel
from ontocast.tool.triple_manager.core import TripleStoreManager

logger = logging.getLogger(__name__)


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
