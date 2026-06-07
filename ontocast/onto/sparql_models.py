"""Pydantic models for graph mutations and tool SPARQL operations.

``GraphUpdate`` / ``TripleOp`` are the canonical LLM pipeline mutation abstraction
(ordered insert/delete triple patches). ``SPARQLOperationModel`` is used by tooling
(``tool/sparql.py``, ``graph_version_manager.py``) — a separate path.
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
        if isinstance(v, str):
            return v
        raise TypeError(f"TripleOp.type must be a string, got {type(v).__name__}")

    graph: LLMGraphWire = Field(
        default_factory=RDFGraph,
        description=(
            "RDF triples for this insert or delete operation. "
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


class GraphUpdate(BaseModel):
    """Structured RDF graph patches for LLM pipeline output.

    Each ``TripleOp`` in ``triple_operations`` is executed in order. SPARQL compilation
    for rdflib apply happens internally via ``generate_sparql_queries()``.
    """

    triple_operations: list[TripleOp] = Field(
        default_factory=list,
        description="List of graph update operations in execution order. "
        "Each operation should be a TripleOp (insert/delete) with graph encoding "
        "per deployment llm_graph_format and OUTPUT INSTRUCTION.",
    )

    def generate_sparql_queries(self) -> list[str]:
        """Compile triple_operations to SPARQL UPDATE strings for rdflib execution.

        Returns:
            List of SPARQL query strings in operation order.
        """
        queries = []

        for op in self.triple_operations:
            if len(op.graph) > 0:
                prefixes = STANDARD_PREFIXES.copy()

                for prefix, uri in op.graph.namespaces():
                    if prefix:
                        prefixes[prefix] = str(uri)

                prefixes.update(op.prefixes)

                if prefixes:
                    prefix_declarations = []
                    for prefix, uri in prefixes.items():
                        prefix_declarations.append(f"PREFIX {prefix}: <{uri}>")
                    prefix_block = "\n".join(prefix_declarations)
                else:
                    prefix_block = ""

                if op.type == "insert":
                    triple_query = self._generate_insert_query(op.graph, prefix_block)
                else:
                    triple_query = self._generate_delete_query(op.graph, prefix_block)
                queries.append(triple_query)

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
        if not self.triple_operations:
            return ""

        diff_parts = []
        operation_count = 0

        for i, op in enumerate(self.triple_operations, 1):
            if len(op.graph) > 0:
                op_type = op.type.upper()
                diff_parts.append(f"{i}. {op_type} {len(op.graph)} triple(s):")

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

        if operation_count == 0:
            return ""

        summary = f"Ontology Update Summary ({operation_count} operation(s)):\n\n"
        summary += "\n".join(diff_parts)

        return summary

    def _generate_insert_query(self, graph: RDFGraph, prefix_block: str) -> str:
        """Generate a SPARQL INSERT query for the given RDFGraph."""
        if len(graph) == 0:
            return ""

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
            if ":" in str(term) and not str(term).startswith("http"):
                return str(term)
            else:
                return f"<{term}>"
        elif isinstance(term, BNode):
            return f"_:{term}"
        elif isinstance(term, Literal):
            if term.language:
                return f'"{term}"@{term.language}'
            elif term.datatype:
                return f'"{term}"^^<{term.datatype}>'
            else:
                return f'"{term}"'
        else:
            return str(term)
