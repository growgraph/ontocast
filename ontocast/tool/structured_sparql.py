"""Structured SPARQL parser and executor.

This module provides tools for parsing and executing structured SPARQL queries
with separate ADD, UPDATE, and REMOVE sections.
"""

import logging

from ontocast.onto.enum import SPARQLOperationType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import (
    SPARQLOperationModel,
    StructuredSPARQLQueryModel,
)

logger = logging.getLogger(__name__)


class StructuredSPARQLParser:
    """Parser for structured SPARQL queries with ADD, UPDATE, REMOVE sections."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_structured_query(self, query_text: str) -> StructuredSPARQLQueryModel:
        """Parse a structured SPARQL query from text.

        Args:
            query_text: The structured SPARQL query text

        Returns:
            StructuredSPARQLQueryModel: Parsed structured query
        """
        self.logger.info("Parsing structured SPARQL query")

        # Split into sections
        sections = self._split_into_sections(query_text)

        # Parse each section and combine all operations
        all_operations = []
        all_operations.extend(
            self._parse_section_operations(
                sections.get("ADD", ""), SPARQLOperationType.INSERT
            )
        )
        all_operations.extend(
            self._parse_section_operations(
                sections.get("UPDATE", ""), SPARQLOperationType.UPDATE
            )
        )
        all_operations.extend(
            self._parse_section_operations(
                sections.get("REMOVE", ""), SPARQLOperationType.DELETE
            )
        )

        # Extract namespaces
        namespaces = self._extract_namespaces(query_text)

        structured_query = StructuredSPARQLQueryModel(
            operations=all_operations,
            namespaces=namespaces,
        )

        self.logger.info(f"Parsed structured query: {structured_query.get_summary()}")
        return structured_query

    def _split_into_sections(self, query_text: str) -> dict[str, str]:
        """Split query text into ADD, UPDATE, REMOVE sections.

        Args:
            query_text: The query text to split

        Returns:
            dict[str, str]: Dictionary mapping section names to content
        """
        sections = {}
        current_section = None
        current_content = []

        lines = query_text.split("\n")

        for line in lines:
            line = line.strip()

            # Check for section headers
            if line.upper().startswith("-- ADD SECTION:") or line.upper().startswith(
                "ADD SECTION:"
            ):
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                current_section = "ADD"
                current_content = []
            elif line.upper().startswith(
                "-- UPDATE SECTION:"
            ) or line.upper().startswith("UPDATE SECTION:"):
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                current_section = "UPDATE"
                current_content = []
            elif line.upper().startswith(
                "-- REMOVE SECTION:"
            ) or line.upper().startswith("REMOVE SECTION:"):
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                current_section = "REMOVE"
                current_content = []
            else:
                current_content.append(line)

        # Add the last section
        if current_section:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _parse_section_operations(
        self, section_content: str, default_type: SPARQLOperationType
    ) -> list[SPARQLOperationModel]:
        """Parse operations from a section.

        Args:
            section_content: The section content
            default_type: Default operation type for the section

        Returns:
            list[SPARQLOperationModel]: List of parsed operations
        """
        operations = []

        if not section_content.strip():
            return operations

        # Split by PREFIX blocks and INSERT/UPDATE/DELETE blocks
        blocks = self._split_into_blocks(section_content)

        for block in blocks:
            if block.strip():
                operation = self._create_operation_from_block(block, default_type)
                if operation:
                    operations.append(operation)

        return operations

    def _split_into_blocks(self, content: str) -> list[str]:
        """Split content into SPARQL operation blocks.

        Args:
            content: The content to split

        Returns:
            list[str]: List of operation blocks
        """
        blocks = []
        current_block = []
        in_operation = False

        lines = content.split("\n")

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Check if this is the start of a new operation
            if (
                line.upper().startswith("INSERT")
                or line.upper().startswith("UPDATE")
                or line.upper().startswith("DELETE")
            ):
                # Save previous block if exists
                if current_block:
                    blocks.append("\n".join(current_block))

                # Start new block
                current_block = [line]
                in_operation = True
            elif in_operation:
                current_block.append(line)

                # Check if operation is complete
                if line == "}" and current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                    in_operation = False

        # Add any remaining block
        if current_block:
            blocks.append("\n".join(current_block))

        return blocks

    def _create_operation_from_block(
        self, block: str, default_type: SPARQLOperationType
    ) -> SPARQLOperationModel | None:
        """Create a SPARQL operation from a block.

        Args:
            block: The operation block
            default_type: Default operation type

        Returns:
            SPARQLOperationModel | None: Created operation or None if invalid
        """
        if not block.strip():
            return None

        # Determine operation type
        if block.upper().startswith("INSERT"):
            operation_type = SPARQLOperationType.INSERT
        elif block.upper().startswith("UPDATE"):
            operation_type = SPARQLOperationType.UPDATE
        elif block.upper().startswith("DELETE"):
            operation_type = SPARQLOperationType.DELETE
        else:
            operation_type = default_type

        return SPARQLOperationModel(operation_type=operation_type, query=block.strip())

    def _extract_namespaces(self, query_text: str) -> dict[str, str]:
        """Extract namespace declarations from query text.

        Args:
            query_text: The query text

        Returns:
            dict[str, str]: Dictionary mapping prefixes to URIs
        """
        namespaces = {}

        lines = query_text.split("\n")
        for line in lines:
            line = line.strip()

            if line.upper().startswith("PREFIX "):
                # Parse PREFIX declaration
                parts = line.split()
                if len(parts) >= 3:
                    prefix = parts[1].rstrip(":")
                    uri = parts[2].strip("<>")
                    namespaces[prefix] = uri

        return namespaces


class StructuredSPARQLExecutor:
    """Executor for structured SPARQL queries."""

    def __init__(self, sparql_tool):
        """Initialize the executor.

        Args:
            sparql_tool: The SPARQL tool for executing operations
        """
        self.sparql_tool = sparql_tool
        self.logger = logging.getLogger(__name__)

    def execute_structured_query(
        self, structured_query: StructuredSPARQLQueryModel, graph: RDFGraph
    ) -> bool:
        """Execute a structured SPARQL query.

        Args:
            structured_query: The structured query to execute
            graph: The RDF graph to execute against

        Returns:
            bool: True if execution was successful
        """
        self.logger.info(
            f"Executing structured query: {structured_query.get_summary()}"
        )

        try:
            # Execute operations in order: ADD, UPDATE, REMOVE
            all_operations = structured_query.get_all_operations()

            for operation in all_operations:
                # Execute SPARQLOperationModel directly
                self.sparql_tool.execute_operation(operation)
                self.logger.debug(
                    f"Executed {operation.operation_type.value} operation"
                )

            self.logger.info("Structured query execution completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to execute structured query: {e}")
            return False

    def execute_add_operations(
        self, structured_query: StructuredSPARQLQueryModel, graph: RDFGraph
    ) -> bool:
        """Execute only ADD operations from a structured query.

        Args:
            structured_query: The structured query
            graph: The RDF graph to execute against

        Returns:
            bool: True if execution was successful
        """
        return self._execute_operations(structured_query.get_add_operations(), "ADD")

    def execute_update_operations(
        self, structured_query: StructuredSPARQLQueryModel, graph: RDFGraph
    ) -> bool:
        """Execute only UPDATE operations from a structured query.

        Args:
            structured_query: The structured query
            graph: The RDF graph to execute against

        Returns:
            bool: True if execution was successful
        """
        return self._execute_operations(
            structured_query.get_update_operations(), "UPDATE"
        )

    def execute_remove_operations(
        self, structured_query: StructuredSPARQLQueryModel, graph: RDFGraph
    ) -> bool:
        """Execute only REMOVE operations from a structured query.

        Args:
            structured_query: The structured query
            graph: The RDF graph to execute against

        Returns:
            bool: True if execution was successful
        """
        return self._execute_operations(
            structured_query.get_remove_operations(), "REMOVE"
        )

    def _execute_operations(
        self, operations: list[SPARQLOperationModel], section_name: str
    ) -> bool:
        """Execute a list of operations.

        Args:
            operations: List of operations to execute
            section_name: Name of the section for logging

        Returns:
            bool: True if execution was successful
        """
        if not operations:
            self.logger.info(f"No {section_name} operations to execute")
            return True

        try:
            for operation in operations:
                # Execute SPARQLOperationModel directly
                self.sparql_tool.execute_operation(operation)
                self.logger.debug(
                    f"Executed {section_name} {operation.operation_type.value} operation"
                )

            self.logger.info(
                f"Executed {len(operations)} {section_name} operations successfully"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to execute {section_name} operations: {e}")
            return False
