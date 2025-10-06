"""Pydantic models for SPARQL operations.

This module provides Pydantic models for structured SPARQL queries
that can be used with PydanticOutputParser for LLM integration.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from ontocast.onto.enum import SPARQLOperationType
from ontocast.onto.rdfgraph import RDFGraph


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

    operations: List[SPARQLOperationModel] = Field(
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

    def get_all_operations(self) -> List[SPARQLOperationModel]:
        """Get all operations in execution order (INSERT, UPDATE, DELETE)."""
        # Sort operations by type: INSERT first, then UPDATE, then DELETE
        type_order = {
            SPARQLOperationType.INSERT: 0,
            SPARQLOperationType.UPDATE: 1,
            SPARQLOperationType.DELETE: 2,
        }
        return sorted(self.operations, key=lambda op: type_order[op.operation_type])

    def get_add_operations(self) -> List[SPARQLOperationModel]:
        """Get all INSERT operations."""
        return [
            op
            for op in self.operations
            if op.operation_type == SPARQLOperationType.INSERT
        ]

    def get_update_operations(self) -> List[SPARQLOperationModel]:
        """Get all UPDATE operations."""
        return [
            op
            for op in self.operations
            if op.operation_type == SPARQLOperationType.UPDATE
        ]

    def get_remove_operations(self) -> List[SPARQLOperationModel]:
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
    critique: Optional[str] = Field(
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
    critique: Optional[str] = Field(
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
    ontology_score: Optional[float] = Field(
        None, description="Score 0-100 for ontology quality and completeness"
    )
    critique: Optional[str] = Field(
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
    facts_score: Optional[float] = Field(
        None, description="Score 0-100 for facts quality and completeness"
    )
    critique: Optional[str] = Field(
        None, description="Optional critique or explanation of the facts generation"
    )
