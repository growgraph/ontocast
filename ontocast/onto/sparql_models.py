"""Pydantic models for SPARQL operations.

This module provides Pydantic models for structured SPARQL queries
that can be used with PydanticOutputParser for LLM integration.
"""

import logging
from typing import Any, Union
from typing import Literal as TypingLiteral

from pydantic import BaseModel, Field, model_validator
from rdflib import BNode, Literal, URIRef

from ontocast.onto.constants import COMMON_PREFIXES
from ontocast.onto.enum import SPARQLOperationType
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


class StructuredSPARQLQueryModel(BaseModel):
    """Pydantic model for structured SPARQL queries.

    Attributes:
        operations: List of SPARQL operations (INSERT, UPDATE, DELETE)
        namespaces: Dictionary mapping prefixes to URIs
    """

    operations: list[SPARQLOperationModel] = Field(
        default_factory=list, description="List of SPARQL operations to execute"
    )
    namespaces: dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping namespace prefixes to URIs",
    )

    def get_summary(self) -> str:
        """Get a summary of the structured query."""
        add_count = len(
            [
                op
                for op in self.operations
                if op.operation_type == SPARQLOperationType.INSERT
            ]
        )
        update_count = len(
            [
                op
                for op in self.operations
                if op.operation_type == SPARQLOperationType.UPDATE
            ]
        )
        remove_count = len(
            [
                op
                for op in self.operations
                if op.operation_type == SPARQLOperationType.DELETE
            ]
        )

        return (
            f"Structured SPARQL Query: "
            f"{add_count} ADD operations, "
            f"{update_count} UPDATE operations, "
            f"{remove_count} REMOVE operations"
        )

    def get_all_operations(self) -> list[SPARQLOperationModel]:
        """Get all operations in execution order (INSERT, UPDATE, DELETE)."""
        # Sort operations by type: INSERT first, then UPDATE, then DELETE
        type_order = {
            SPARQLOperationType.INSERT: 0,
            SPARQLOperationType.UPDATE: 1,
            SPARQLOperationType.DELETE: 2,
        }
        return sorted(self.operations, key=lambda op: type_order[op.operation_type])

    def get_add_operations(self) -> list[SPARQLOperationModel]:
        """Get all INSERT operations."""
        return [
            op
            for op in self.operations
            if op.operation_type == SPARQLOperationType.INSERT
        ]

    def get_update_operations(self) -> list[SPARQLOperationModel]:
        """Get all UPDATE operations."""
        return [
            op
            for op in self.operations
            if op.operation_type == SPARQLOperationType.UPDATE
        ]

    def get_remove_operations(self) -> list[SPARQLOperationModel]:
        """Get all DELETE operations."""
        return [
            op
            for op in self.operations
            if op.operation_type == SPARQLOperationType.DELETE
        ]


class OntologyUpdateReport(BaseModel):
    """Report from ontology update process using structured SPARQL.

    Attributes:
        update_success: True if the ontology update was performed successfully
        structured_query: The structured SPARQL query used for the update
        add_count: Number of ADD operations
        update_count: Number of UPDATE operations
        remove_count: Number of REMOVE operations
        critique: Optional critique of the update process
    """

    update_success: bool = Field(
        description="True if the ontology update was performed successfully, False otherwise"
    )
    structured_query: StructuredSPARQLQueryModel = Field(
        description="The structured SPARQL query used for the update"
    )
    add_count: int = Field(
        description="Number of ADD operations in the structured query"
    )
    update_count: int = Field(
        description="Number of UPDATE operations in the structured query"
    )
    remove_count: int = Field(
        description="Number of REMOVE operations in the structured query"
    )
    critique: str | None = Field(
        None, description="Optional critique or explanation of the update process"
    )


class FactsUpdateReport(BaseModel):
    """Report from facts update process using structured SPARQL.

    Attributes:
        update_success: True if the facts update was performed successfully
        structured_query: The structured SPARQL query used for the update
        add_count: Number of ADD operations
        update_count: Number of UPDATE operations
        remove_count: Number of REMOVE operations
        critique: Optional critique of the update process
    """

    update_success: bool = Field(
        description="True if the facts update was performed successfully, False otherwise"
    )
    structured_query: StructuredSPARQLQueryModel = Field(
        description="The structured SPARQL query used for the update"
    )
    add_count: int = Field(
        description="Number of ADD operations in the structured query"
    )
    update_count: int = Field(
        description="Number of UPDATE operations in the structured query"
    )
    remove_count: int = Field(
        description="Number of REMOVE operations in the structured query"
    )
    critique: str | None = Field(
        None, description="Optional critique or explanation of the update process"
    )


class FreshOntologyReport(BaseModel):
    """Report from fresh ontology generation process.

    Attributes:
        generation_success: True if the ontology was generated successfully
        ontology_graph: The generated ontology as an RDFGraph
        ontology_score: Score 0-100 for ontology quality
        critique: Optional critique of the ontology generation
    """

    generation_success: bool = Field(
        description="True if the ontology was generated successfully, False otherwise"
    )
    ontology_graph: RDFGraph = Field(
        default_factory=RDFGraph,
        description="The generated ontology as an RDFGraph in Turtle format",
    )
    ontology_score: float | None = Field(
        None, description="Score 0-100 for ontology quality and completeness"
    )
    critique: str | None = Field(
        None, description="Optional critique or explanation of the ontology generation"
    )


class FreshFactsReport(BaseModel):
    """Report from fresh facts generation process.

    Attributes:
        generation_success: True if the facts were generated successfully
        facts_graph: The generated facts as an RDFGraph
        facts_score: Score 0-100 for facts quality
        critique: Optional critique of the facts generation
    """

    generation_success: bool = Field(
        description="True if the facts were generated successfully, False otherwise"
    )
    facts_graph: RDFGraph = Field(
        default_factory=RDFGraph,
        description="The generated facts as an RDFGraph in Turtle format",
    )
    facts_score: float | None = Field(
        None, description="Score 0-100 for facts quality and completeness"
    )
    critique: str | None = Field(
        None, description="Optional critique or explanation of the facts generation"
    )


class Triple(BaseModel):
    """Represents a single RDF triple (subject, predicate, object).

    This is the basic building block for RDF data and SPARQL operations.
    All components should be valid RDF terms (URIs, literals, or blank nodes).
    Strings are automatically converted to proper RDF terms after initialization.
    """

    subject: str = Field(
        description="The subject of the RDF triple (URI, blank node, or literal)"
    )
    predicate: str = Field(
        description="The predicate of the RDF triple (typically a URI)"
    )
    object: str = Field(
        description="The object of the RDF triple (URI, blank node, or literal)"
    )

    # Internal RDF terms (computed from strings)
    _subject_term: Union[URIRef, BNode] | None = None
    _predicate_term: None | URIRef = None
    _object_term: Union[URIRef, BNode, Literal] | None = None

    @model_validator(mode="after")
    def convert_to_rdf_terms(self):
        """Convert string representations to proper RDF terms."""
        self._subject_term = self._parse_subject(self.subject)
        self._predicate_term = self._parse_predicate(self.predicate)
        self._object_term = self._parse_object(self.object)
        return self

    def _parse_subject(self, subject: str) -> Union[URIRef, BNode]:
        """Parse subject string to RDF term."""
        subject = subject.strip()
        if subject.startswith("_:"):
            return BNode(subject[2:])
        elif subject.startswith("<") and subject.endswith(">"):
            return URIRef(subject[1:-1])
        else:
            # For prefixed names or bare URIs, return as-is for now
            # The SPARQL serialization will handle proper formatting
            return URIRef(subject)

    def _parse_predicate(self, predicate: str) -> URIRef:
        """Parse predicate string to URIRef."""
        predicate = predicate.strip()
        if predicate.startswith("<") and predicate.endswith(">"):
            return URIRef(predicate[1:-1])
        else:
            # For prefixed names or bare URIs, return as-is for now
            return URIRef(predicate)

    def _parse_object(self, object_str: str) -> Union[URIRef, BNode, Literal]:
        """Parse object string to RDF term."""
        object_str = object_str.strip()
        if object_str.startswith("_:"):
            return BNode(object_str[2:])
        elif object_str.startswith("<") and object_str.endswith(">"):
            return URIRef(object_str[1:-1])
        elif object_str.startswith('"') and object_str.endswith('"'):
            # Simple literal
            return Literal(object_str[1:-1])
        elif object_str.startswith('"') and '"^^' in object_str:
            # Typed literal
            value, datatype = object_str.split('"^^', 1)
            # Clean up datatype URI if it has angle brackets
            if datatype.startswith("<") and datatype.endswith(">"):
                datatype = datatype[1:-1]
            return Literal(value[1:], datatype=URIRef(datatype))
        else:
            # For prefixed names or bare URIs, return as-is for now
            return URIRef(object_str)

    def to_rdf_terms(
        self,
    ) -> tuple[Union[URIRef, BNode], URIRef, Union[URIRef, BNode, Literal]]:
        """Get the RDF terms for this triple."""
        if (
            self._subject_term is None
            or self._predicate_term is None
            or self._object_term is None
        ):
            raise ValueError(
                "RDF terms not initialized. This should not happen after model validation."
            )
        return (self._subject_term, self._predicate_term, self._object_term)


class TripleOp(BaseModel):
    """Operation to modify triples in the RDF graph.

    This operation can insert or delete triples. REQUIRES explicit prefix declarations.
    """

    type: TypingLiteral["insert", "delete"] = Field(
        description="Type of operation: 'insert' to add triples, 'delete' to remove triples"
    )
    triples: list[Triple] = Field(
        default_factory=list,
        description="List of RDF triples to insert or delete. "
        "Example: [Triple(subject='fca:CriminalChamber', predicate='rdf:type', object='fca:ChamberType')]",
    )
    prefixes: dict[str, str] = Field(
        default_factory=dict,
        description="REQUIRED: ALL custom prefixes used in the triples must be declared here. "
        "Standard prefixes from COMMON_PREFIXES in constants.py (rdf, rdfs, owl, xsd, dc, dcterms, skos, foaf, schema, prov, ex) are automatically available and do NOT need to be declared. "
        "For example, if you use 'fca:CriminalChamber', you must include 'fca' in this dictionary. "
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


Operation = TripleOp | GenericSparqlQuery


class GraphUpdate(BaseModel):
    """Structured representation of RDF graph updates for LLM output.

    This model represents ontology updates as a structured set of operations.
    Each operation in the list is executed in order to modify the graph.
    """

    operations: list[Operation] = Field(
        default_factory=list,
        description="List of graph update operations in execution order. "
        "Each operation should be a TripleOp (for insert/delete) with explicit prefix declarations, "
        "or a GenericSparqlQuery for complex custom queries. "
        "Example: [TripleOp(type='insert', triples=[...], prefixes={'fca': 'http://example.org/fca#'})]",
    )

    def generate_sparql_queries(self) -> list[str]:
        """Generate a list of SPARQL queries to execute the graph update.

        Returns:
            List of SPARQL query strings that can be executed to perform the update.
            The queries are generated in the exact order of operations in the operations list.
        """
        queries = []

        # Process operations in order, generating a query for each operation
        for op in self.operations:
            if isinstance(op, TripleOp):
                if op.triples:  # Only generate query if there are triples
                    # Build prefix block for this operation
                    # Start with standard prefixes from COMMON_PREFIXES
                    prefixes = STANDARD_PREFIXES.copy()
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
                        triple_query = self._generate_insert_query(
                            op.triples, prefix_block
                        )
                    else:  # delete
                        triple_query = self._generate_delete_query(
                            op.triples, prefix_block
                        )
                    queries.append(triple_query)
            elif isinstance(op, GenericSparqlQuery):
                if op.query.strip():  # Only generate query if there's content
                    # For custom SPARQL queries, use them as-is
                    queries.append(op.query)
        return queries

    def generate_diff_summary(self) -> str:
        """Generate a human-readable diff summary of all operations for LLM consumption.

        Returns:
            String representation of all operations showing what will be added, removed, and modified.
            Returns empty string if no operations to perform.
        """
        if not self.operations:
            return ""

        diff_parts = []
        operation_count = 0

        for i, op in enumerate(self.operations, 1):
            if isinstance(op, TripleOp):
                if op.triples:
                    op_type = op.type.upper()
                    diff_parts.append(f"{i}. {op_type} {len(op.triples)} triple(s):")

                    # Show declared prefixes if any
                    if op.prefixes:
                        prefix_list = ", ".join(
                            [f"{k}: {v}" for k, v in op.prefixes.items()]
                        )
                        diff_parts.append(f"   Prefixes: {prefix_list}")

                    for triple in op.triples:
                        subject_term, predicate_term, object_term = (
                            triple.to_rdf_terms()
                        )
                        symbol = "+" if op.type == "insert" else "-"
                        diff_parts.append(
                            f"   {symbol} {self._serialize_rdf_term(subject_term)} {self._serialize_rdf_term(predicate_term)} {self._serialize_rdf_term(object_term)}"
                        )
                    operation_count += 1

            elif isinstance(op, GenericSparqlQuery):
                if op.query.strip():
                    # Truncate long queries for readability
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

    def _generate_insert_query(self, triples: list[Triple], prefix_block: str) -> str:
        """Generate a SPARQL INSERT query for the given triples."""
        if not triples:
            return ""

        # Format triples for SPARQL using proper RDF term serialization
        triple_patterns = []
        for triple in triples:
            subject_term, predicate_term, object_term = triple.to_rdf_terms()
            triple_patterns.append(
                f"    {self._serialize_rdf_term(subject_term)} {self._serialize_rdf_term(predicate_term)} {self._serialize_rdf_term(object_term)} ."
            )

        triples_block = "\n".join(triple_patterns)

        query_parts = []
        if prefix_block:
            query_parts.append(prefix_block)
        query_parts.append("INSERT DATA {")
        query_parts.append(triples_block)
        query_parts.append("}")

        return "\n".join(query_parts)

    def _generate_delete_query(self, triples: list[Triple], prefix_block: str) -> str:
        """Generate a SPARQL DELETE query for the given triples."""
        if not triples:
            return ""

        # Format triples for SPARQL using proper RDF term serialization
        triple_patterns = []
        for triple in triples:
            subject_term, predicate_term, object_term = triple.to_rdf_terms()
            triple_patterns.append(
                f"    {self._serialize_rdf_term(subject_term)} {self._serialize_rdf_term(predicate_term)} {self._serialize_rdf_term(object_term)} ."
            )

        triples_block = "\n".join(triple_patterns)

        query_parts = []
        if prefix_block:
            query_parts.append(prefix_block)
        query_parts.append("DELETE DATA {")
        query_parts.append(triples_block)
        query_parts.append("}")

        return "\n".join(query_parts)

    def _serialize_rdf_term(self, term: Union[URIRef, BNode, Literal]) -> str:
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
            if term.datatype:
                return f'"{term}"^^<{term.datatype}>'
            else:
                return f'"{term}"'
        else:
            # Fallback to string representation
            return str(term)
