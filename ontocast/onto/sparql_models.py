"""Pydantic models for SPARQL graph mutations and tool SPARQL operations.

``GraphUpdate`` / ``TripleOp`` are the canonical LLM pipeline mutation abstraction
(ordered triple patches + optional raw SPARQL strings). ``SPARQLOperationModel`` is
used by tooling (``tool/sparql.py``, ``graph_version_manager.py``) — a separate path.
"""

import logging
from typing import Any
from typing import Literal as TypingLiteral

from pydantic import BaseModel, Field, field_validator
from rdflib import BNode, Literal, Node, URIRef

from ontocast.onto.constants import COMMON_PREFIXES
from ontocast.onto.enum import SPARQLOperationType
from ontocast.onto.llm_graph_payload import LLMGraphWire
from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)

# Convert COMMON_PREFIXES from Turtle format (with angle brackets) to SPARQL format (without)
# Example: "<http://example.org/>" -> "http://example.org/"
STANDARD_PREFIXES = {prefix: uri.strip("<>") for prefix, uri in COMMON_PREFIXES.items()}


class SPARQLOperationModel(BaseModel):
    """Pydantic model for a single SPARQL operation.

    Attributes:
        operation_type: Type of SPARQL operation (INSERT, UPDATE, DELETE)
        query: The SPARQL query string
        description: Optional description of the operation
        metadata: Optional metadata dictionary
    """

    operation_type: SPARQLOperationType = Field(
        description="Type of SPARQL operation: INSERT, UPDATE, or DELETE"
    )
    query: str = Field(
        description="The complete SPARQL query string with proper syntax"
    )
    description: str = Field(
        default="", description="Optional description of the operation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata dictionary for the operation",
    )


class TripleOp(BaseModel):
    """Operation to modify triples in the RDF graph.

    This operation can insert or delete triples. Prefixes are automatically extracted
    from the RDFGraph's namespace bindings (from @prefix declarations in Turtle).
    """

    type: TypingLiteral["insert", "delete"] = Field(
        description="Type of operation: 'insert' to add triples, 'delete' to remove triples"
    )

    @field_validator("type", mode="before")
    @classmethod
    def normalize_op_type(cls, v: object) -> str:
        if isinstance(v, str) and v.lower() == "update":
            return "insert"
        return v  # type: ignore[return-value]

    graph: LLMGraphWire = Field(
        default_factory=RDFGraph,
        description=(
            "RDF graph containing triples to insert or delete. "
            "Encoding is defined by deployment llm_graph_format and OUTPUT INSTRUCTION."
        ),
    )
    prefixes: dict[str, str] = Field(
        default_factory=dict,
        description="Optional: Additional or override prefixes. "
        "Prefixes are automatically extracted from the RDFGraph's namespace bindings. "
        "Standard prefixes from COMMON_PREFIXES in constants.py (rdf, rdfs, owl, xsd, dc, dcterms, skos, foaf, schema, prov, ex) are automatically available. "
        "This field can be used to add or override prefixes if needed. "
        "Mapping format: {'prefix_name': 'namespace_uri'}. Example: {'fca': 'http://example.org/ontologies/fca#'}",
    )


class GenericSparqlQuery(BaseModel):
    """Operation for custom SPARQL queries that go beyond basic insert/delete operations.

    This operation allows for complex SPARQL queries that cannot be expressed
    using the structured operations. Use this when you need custom SPARQL syntax,
    complex WHERE clauses, or operations that don't fit the basic patterns.
    """

    type: TypingLiteral["sparql_query"] = Field(
        default="sparql_query",
        description="Type of operation - always 'sparql_query' for this operation",
    )
    query: str = Field(
        description="The complete SPARQL query string with proper syntax"
    )


class GraphUpdate(BaseModel):
    """Structured representation of RDF graph updates for LLM output.

    This model represents ontology updates as a structured set of operations.
    Each operation in the list is executed in order to modify the graph.
    """

    triple_operations: list[TripleOp] = Field(
        default_factory=list,
        description="List of graph update operations in execution order. "
        "Each operation should be a TripleOp (insert/delete) with graph encoding "
        "per deployment llm_graph_format and OUTPUT INSTRUCTION.",
    )

    sparql_operations: list[GenericSparqlQuery] = Field(
        default_factory=list,
        description="List of graph update operations in execution order. "
        "Each operation should be a GenericSparqlQuery for complex custom queries. ",
    )

    def generate_sparql_queries(self) -> list[str]:
        """Generate a list of SPARQL queries to execute the graph update.

        Returns:
            List of SPARQL query strings that can be executed to perform the update.
            The queries are generated in the exact order of operations in the operations list.
        """
        queries = []

        # Process triple operations first
        for op in self.triple_operations:
            if len(op.graph) > 0:  # Only generate query if there are triples
                # Build prefix block for this operation
                # Start with standard prefixes from COMMON_PREFIXES
                prefixes = STANDARD_PREFIXES.copy()

                # Extract prefixes from RDFGraph's namespace bindings
                for prefix, uri in op.graph.namespaces():
                    if prefix:  # Skip empty prefix
                        prefixes[prefix] = str(uri)

                # Add custom prefixes declared in this operation (may override standard ones)
                prefixes.update(op.prefixes)

                # Generate PREFIX declarations block
                if prefixes:
                    prefix_declarations = []
                    for prefix, uri in prefixes.items():
                        prefix_declarations.append(f"PREFIX {prefix}: <{uri}>")
                    prefix_block = "\n".join(prefix_declarations)
                else:
                    prefix_block = ""

                # Generate query based on operation type
                if op.type == "insert":
                    triple_query = self._generate_insert_query(op.graph, prefix_block)
                else:  # delete
                    triple_query = self._generate_delete_query(op.graph, prefix_block)
                queries.append(triple_query)

        # Process SPARQL operations
        for op in self.sparql_operations:
            if op.query.strip():  # Only generate query if there's content
                # For custom SPARQL queries, use them as-is
                queries.append(op.query)

        return queries

    def count_total_triples(self) -> tuple[int, int]:
        """Count total triples across all operations.

        Returns:
            Tuple of (total_operations, total_triples) where:
            - total_operations: Number of operations
            - total_triples: Total number of triples across all TripleOp operations
        """
        total_triples = sum(len(op.graph) for op in self.triple_operations)
        return (len(self.triple_operations), total_triples)

    def extract_insert_graph(self) -> RDFGraph:
        """Extract RDFGraph of all insert triples from triple_operations.

        Only TripleOps with type='insert' are included. sparql_operations
        are not extractable as triples and are skipped.

        Returns:
            RDFGraph containing the union of all insert triples.
        """
        result = RDFGraph()
        for op in self.triple_operations:
            if op.type == "insert" and len(op.graph) > 0:
                for triple in op.graph:
                    result.add(triple)
                for prefix, uri in op.graph.namespaces():
                    if prefix:
                        result.bind(prefix, uri)
                for prefix, uri in op.prefixes.items():
                    result.bind(prefix, uri)
        return result

    def generate_diff_summary(self) -> str:
        """Generate a human-readable diff summary of all operations for LLM consumption.

        Returns:
            String representation of all operations showing what will be added, removed, and modified.
            Returns empty string if no operations to perform.
        """
        if not self.triple_operations and not self.sparql_operations:
            return ""

        diff_parts = []
        operation_count = 0

        for i, op in enumerate(self.triple_operations, 1):
            if len(op.graph) > 0:
                op_type = op.type.upper()
                diff_parts.append(f"{i}. {op_type} {len(op.graph)} triple(s):")

                # Show prefixes from graph and explicit prefixes
                graph_prefixes = {
                    prefix: str(uri) for prefix, uri in op.graph.namespaces() if prefix
                }
                all_prefixes = {**graph_prefixes, **op.prefixes}
                if all_prefixes:
                    prefix_list = ", ".join(
                        [f"{k}: {v}" for k, v in all_prefixes.items()]
                    )
                    diff_parts.append(f"   Prefixes: {prefix_list}")

                for subject, predicate, obj in op.graph:
                    symbol = "+" if op.type == "insert" else "-"
                    diff_parts.append(
                        f"   {symbol} {self._serialize_rdf_term(subject)} {self._serialize_rdf_term(predicate)} {self._serialize_rdf_term(obj)}"
                    )
                operation_count += 1

        base_index = len(self.triple_operations)
        for j, op in enumerate(self.sparql_operations, 1):
            if op.query.strip():
                i = base_index + j
                query_preview = op.query.strip()
                if len(query_preview) > 100:
                    query_preview = query_preview[:97] + "..."
                diff_parts.append(f"{i}. CUSTOM SPARQL QUERY:")
                diff_parts.append(f"   {query_preview}")
                operation_count += 1

        if operation_count == 0:
            return ""

        summary = f"Ontology Update Summary ({operation_count} operation(s)):\n\n"
        summary += "\n".join(diff_parts)

        return summary

    def _generate_insert_query(self, graph: RDFGraph, prefix_block: str) -> str:
        """Generate a SPARQL INSERT query for the given RDFGraph."""
        if len(graph) == 0:
            return ""

        # Format triples for SPARQL using proper RDF term serialization
        triple_patterns = []
        for subject, predicate, obj in graph:
            triple_patterns.append(
                f"    {self._serialize_rdf_term(subject)} {self._serialize_rdf_term(predicate)} {self._serialize_rdf_term(obj)} ."
            )

        triples_block = "\n".join(triple_patterns)

        query_parts = []
        if prefix_block:
            query_parts.append(prefix_block)
        query_parts.append("INSERT DATA {")
        query_parts.append(triples_block)
        query_parts.append("}")

        return "\n".join(query_parts)

    def _generate_delete_query(self, graph: RDFGraph, prefix_block: str) -> str:
        """Generate a SPARQL DELETE query for the given RDFGraph."""
        if len(graph) == 0:
            return ""

        # Format triples for SPARQL using proper RDF term serialization
        triple_patterns = []
        for subject, predicate, obj in graph:
            triple_patterns.append(
                f"    {self._serialize_rdf_term(subject)} {self._serialize_rdf_term(predicate)} {self._serialize_rdf_term(obj)} ."
            )

        triples_block = "\n".join(triple_patterns)

        query_parts = []
        if prefix_block:
            query_parts.append(prefix_block)
        query_parts.append("DELETE DATA {")
        query_parts.append(triples_block)
        query_parts.append("}")

        return "\n".join(query_parts)

    def _serialize_rdf_term(self, term: Node) -> str:
        """Serialize an RDF term to its SPARQL string representation."""
        if isinstance(term, URIRef):
            # Check if it's already a prefixed name (contains ':')
            if ":" in str(term) and not str(term).startswith("http"):
                return str(term)
            else:
                return f"<{term}>"
        elif isinstance(term, BNode):
            return f"_:{term}"
        elif isinstance(term, Literal):
            # Handle language-tagged literals first
            if term.language:
                return f'"{term}"@{term.language}'
            elif term.datatype:
                return f'"{term}"^^<{term.datatype}>'
            else:
                return f'"{term}"'
        else:
            # Fallback to string representation
            return str(term)
