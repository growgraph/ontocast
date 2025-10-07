"""Graph diff generation and application system.

This module provides functionality for generating and applying diffs between
graph versions, enabling incremental updates and efficient context passing.
"""

import logging
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)


class DiffOperation(StrEnum):
    """Enumeration of diff operations."""

    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"
    UNCHANGED = "unchanged"


class TripleDiff(BaseModel):
    """Represents a diff for a single triple."""

    subject: str = Field(description="Subject of the triple")
    predicate: str = Field(description="Predicate of the triple")
    object: str = Field(description="Object of the triple")
    operation: DiffOperation = Field(description="Operation performed on this triple")
    old_value: str | None = Field(
        default=None, description="Old value for modifications"
    )
    new_value: str | None = Field(
        default=None, description="New value for modifications"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class GraphDiff(BaseModel):
    """Represents differences between two graph versions."""

    # Diff metadata
    diff_id: str = Field(description="Unique identifier for this diff")
    source_version_id: str = Field(description="ID of the source version")
    target_version_id: str = Field(description="ID of the target version")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When this diff was created"
    )

    # Diff content
    triple_diffs: list[TripleDiff] = Field(
        default_factory=list, description="List of triple differences"
    )

    # Summary statistics
    added_triples: int = Field(default=0, description="Number of added triples")
    removed_triples: int = Field(default=0, description="Number of removed triples")
    modified_triples: int = Field(default=0, description="Number of modified triples")
    unchanged_triples: int = Field(default=0, description="Number of unchanged triples")

    # Context information
    context_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Context metadata for this diff"
    )

    def get_summary(self) -> str:
        """Get a summary of this diff.

        Returns:
            str: Human-readable summary of the diff
        """
        return f"""
Graph Diff Summary:
- Diff ID: {self.diff_id}
- Source: {self.source_version_id} → Target: {self.target_version_id}
- Created: {self.created_at.isoformat()}
- Added: {self.added_triples} triples
- Removed: {self.removed_triples} triples
- Modified: {self.modified_triples} triples
- Unchanged: {self.unchanged_triples} triples
- Total changes: {self.added_triples + self.removed_triples + self.modified_triples}
"""

    def get_sparql_operations(self) -> list[str]:
        """Get SPARQL operations for applying this diff.

        Returns:
            list[str]: List of SPARQL queries to apply the diff
        """
        operations = []

        for triple_diff in self.triple_diffs:
            if triple_diff.operation == DiffOperation.ADD:
                # Generate INSERT operation
                sparql = f"INSERT DATA {{ {triple_diff.subject} {triple_diff.predicate} {triple_diff.object} . }}"
                operations.append(sparql)

            elif triple_diff.operation == DiffOperation.REMOVE:
                # Generate DELETE operation
                sparql = f"DELETE DATA {{ {triple_diff.subject} {triple_diff.predicate} {triple_diff.object} . }}"
                operations.append(sparql)

            elif triple_diff.operation == DiffOperation.MODIFY:
                # Generate DELETE + INSERT for modifications
                if triple_diff.old_value:
                    delete_sparql = f"DELETE DATA {{ {triple_diff.subject} {triple_diff.predicate} {triple_diff.old_value} . }}"
                    operations.append(delete_sparql)
                if triple_diff.new_value:
                    insert_sparql = f"INSERT DATA {{ {triple_diff.subject} {triple_diff.predicate} {triple_diff.new_value} . }}"
                    operations.append(insert_sparql)

        return operations

    def is_empty(self) -> bool:
        """Check if this diff is empty (no changes).

        Returns:
            bool: True if no changes, False otherwise
        """
        return (
            self.added_triples == 0
            and self.removed_triples == 0
            and self.modified_triples == 0
        )

    def get_changed_subjects(self) -> set[str]:
        """Get all subjects that have changes.

        Returns:
            set[str]: Set of subject URIs that have changes
        """
        return {
            triple_diff.subject
            for triple_diff in self.triple_diffs
            if triple_diff.operation != DiffOperation.UNCHANGED
        }

    def get_changed_predicates(self) -> set[str]:
        """Get all predicates that have changes.

        Returns:
            set[str]: Set of predicate URIs that have changes
        """
        return {
            triple_diff.predicate
            for triple_diff in self.triple_diffs
            if triple_diff.operation != DiffOperation.UNCHANGED
        }


class DiffTool:
    """Tool for generating and applying graph diffs."""

    def __init__(self):
        """Initialize the diff tool."""
        self.logger = logging.getLogger(__name__)

    def generate_diff(
        self,
        source_graph: RDFGraph,
        target_graph: RDFGraph,
        source_version_id: str,
        target_version_id: str,
        context_metadata: dict[str, Any] | None = None,
    ) -> GraphDiff:
        """Generate a diff between two graphs.

        Args:
            source_graph: The source graph to compare from
            target_graph: The target graph to compare to
            source_version_id: ID of the source version
            target_version_id: ID of the target version
            context_metadata: Optional context metadata

        Returns:
            GraphDiff: The generated diff
        """
        self.logger.info(
            f"Generating diff from {source_version_id} to {target_version_id}"
        )

        # Get triples from both graphs
        source_triples = self._get_triples_set(source_graph)
        target_triples = self._get_triples_set(target_graph)

        # Find differences
        triple_diffs = []
        added_triples = 0
        removed_triples = 0
        modified_triples = 0
        unchanged_triples = 0

        # Find added triples
        for triple in target_triples - source_triples:
            triple_diff = TripleDiff(
                subject=triple[0],
                predicate=triple[1],
                object=triple[2],
                operation=DiffOperation.ADD,
            )
            triple_diffs.append(triple_diff)
            added_triples += 1

        # Find removed triples
        for triple in source_triples - target_triples:
            triple_diff = TripleDiff(
                subject=triple[0],
                predicate=triple[1],
                object=triple[2],
                operation=DiffOperation.REMOVE,
            )
            triple_diffs.append(triple_diff)
            removed_triples += 1

        # Find unchanged triples
        for triple in source_triples & target_triples:
            triple_diff = TripleDiff(
                subject=triple[0],
                predicate=triple[1],
                object=triple[2],
                operation=DiffOperation.UNCHANGED,
            )
            triple_diffs.append(triple_diff)
            unchanged_triples += 1

        # Create diff
        diff = GraphDiff(
            diff_id=f"diff_{source_version_id}_{target_version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_version_id=source_version_id,
            target_version_id=target_version_id,
            triple_diffs=triple_diffs,
            added_triples=added_triples,
            removed_triples=removed_triples,
            modified_triples=modified_triples,
            unchanged_triples=unchanged_triples,
            context_metadata=context_metadata or {},
        )

        self.logger.info(
            f"Generated diff with {added_triples} additions, {removed_triples} removals, {modified_triples} modifications"
        )
        return diff

    def apply_diff(self, graph: RDFGraph, diff: GraphDiff) -> RDFGraph:
        """Apply a diff to a graph.

        Args:
            graph: The graph to apply the diff to
            diff: The diff to apply

        Returns:
            RDFGraph: The updated graph
        """
        self.logger.info(f"Applying diff {diff.diff_id} to graph")

        # Create a copy of the graph
        updated_graph = RDFGraph()
        updated_graph += graph

        # Apply each triple diff
        for triple_diff in diff.triple_diffs:
            if triple_diff.operation == DiffOperation.ADD:
                # Add the triple
                updated_graph.add_triple(
                    triple_diff.subject,
                    triple_diff.predicate,
                    triple_diff.object,
                )

            elif triple_diff.operation == DiffOperation.REMOVE:
                # Remove the triple
                updated_graph.remove_triple(
                    triple_diff.subject,
                    triple_diff.predicate,
                    triple_diff.object,
                )

            elif triple_diff.operation == DiffOperation.MODIFY:
                # Remove old, add new
                if triple_diff.old_value:
                    updated_graph.remove_triple(
                        triple_diff.subject,
                        triple_diff.predicate,
                        triple_diff.old_value,
                    )
                if triple_diff.new_value:
                    updated_graph.add_triple(
                        triple_diff.subject,
                        triple_diff.predicate,
                        triple_diff.new_value,
                    )

        self.logger.info(
            f"Applied diff to graph, new triple count: {len(updated_graph)}"
        )
        return updated_graph

    def _get_triples_set(self, graph: RDFGraph) -> set[tuple[str, str, str]]:
        """Get a set of triples from a graph.

        Args:
            graph: The graph to extract triples from

        Returns:
            set[tuple[str, str, str]]: Set of (subject, predicate, object) tuples
        """
        triples = set()
        for triple in graph:
            triples.add((str(triple[0]), str(triple[1]), str(triple[2])))
        return triples

    def get_diff_summary(self, diff: GraphDiff) -> str:
        """Get a human-readable summary of a diff.

        Args:
            diff: The diff to summarize

        Returns:
            str: Human-readable summary
        """
        return f"""
Diff Summary:
- ID: {diff.diff_id}
- Source: {diff.source_version_id} → Target: {diff.target_version_id}
- Created: {diff.created_at.isoformat()}
- Changes: {diff.added_triples} added, {diff.removed_triples} removed, {diff.modified_triples} modified
- Total triples: {len(diff.triple_diffs)}
- Changed subjects: {len(diff.get_changed_subjects())}
- Changed predicates: {len(diff.get_changed_predicates())}
"""

    def merge_diffs(self, diffs: list[GraphDiff]) -> GraphDiff:
        """Merge multiple diffs into a single diff.

        Args:
            diffs: List of diffs to merge

        Returns:
            GraphDiff: Merged diff
        """
        if not diffs:
            raise ValueError("Cannot merge empty list of diffs")

        if len(diffs) == 1:
            return diffs[0]

        # Start with the first diff
        merged_diff = diffs[0]

        # Merge each subsequent diff
        for diff in diffs[1:]:
            # Combine triple diffs
            merged_diff.triple_diffs.extend(diff.triple_diffs)

            # Update statistics
            merged_diff.added_triples += diff.added_triples
            merged_diff.removed_triples += diff.removed_triples
            merged_diff.modified_triples += diff.modified_triples
            merged_diff.unchanged_triples += diff.unchanged_triples

            # Update target version
            merged_diff.target_version_id = diff.target_version_id

        # Update diff ID
        merged_diff.diff_id = f"merged_{merged_diff.source_version_id}_{merged_diff.target_version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Merged {len(diffs)} diffs into single diff")
        return merged_diff
