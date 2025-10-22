"""Graph version manager for tracking ontology and facts graph changes.

This module provides functionality for managing versions of RDF graphs,
enabling incremental updates and change tracking.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import SPARQLOperationModel

logger = logging.getLogger(__name__)


class VersionDetails(BaseModel):
    """Pydantic model for version details in statistics."""

    version_count: int = Field(description="Number of versions")
    latest_size: int = Field(description="Size of the latest version")
    latest_timestamp: str | None = Field(description="Timestamp of the latest version")


class VersionStatistics(BaseModel):
    """Pydantic model for version statistics."""

    total_ontologies: int = Field(description="Total number of ontologies")
    total_chunks: int = Field(description="Total number of chunks")
    total_ontology_versions: int = Field(
        description="Total number of ontology versions"
    )
    total_facts_versions: int = Field(description="Total number of facts versions")
    ontology_details: dict[str, VersionDetails] = Field(
        description="Details for each ontology"
    )
    chunk_details: dict[str, VersionDetails] = Field(
        description="Details for each chunk"
    )


class GraphVersion(BaseModel):
    """Represents a version of a graph."""

    id: str = Field(description="Unique identifier for this graph version")
    graph: RDFGraph = Field(description="The RDF graph for this version")
    timestamp: datetime = Field(description="When this version was created")
    operations: list[SPARQLOperationModel] = Field(
        default_factory=list,
        description="List of SPARQL operations that created this version",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata for this version"
    )
    parent_version_id: str | None = Field(
        default=None, description="ID of the parent version this was derived from"
    )

    def get_size(self) -> int:
        """Get the number of triples in this version."""
        return len(self.graph)

    def get_namespaces(self) -> dict[str, str]:
        """Get the namespaces bound in this version."""
        return dict(self.graph.namespaces())


class GraphDiff(BaseModel):
    """Represents differences between two graph versions."""

    added_triples: list[tuple] = Field(
        default_factory=list, description="List of triples that were added"
    )
    removed_triples: list[tuple] = Field(
        default_factory=list, description="List of triples that were removed"
    )
    modified_triples: list[tuple[tuple, tuple]] = Field(
        default_factory=list,
        description="List of (old, new) triple pairs that were modified",
    )
    added_namespaces: dict[str, str] = Field(
        default_factory=dict, description="Namespaces that were added"
    )
    removed_namespaces: dict[str, str] = Field(
        default_factory=dict, description="Namespaces that were removed"
    )

    def is_empty(self) -> bool:
        """Check if the diff is empty (no changes)."""
        return (
            len(self.added_triples) == 0
            and len(self.removed_triples) == 0
            and len(self.modified_triples) == 0
            and len(self.added_namespaces) == 0
            and len(self.removed_namespaces) == 0
        )


class GraphVersionManager:
    """Manages versions of ontology and facts graphs."""

    def __init__(self):
        """Initialize the graph version manager."""
        self.ontology_versions: dict[str, list[GraphVersion]] = defaultdict(list)
        self.facts_versions: dict[str, list[GraphVersion]] = defaultdict(list)
        self.version_metadata: dict[str, dict[str, Any]] = {}

    def create_ontology_version(
        self,
        ontology_id: str,
        graph: RDFGraph,
        operations: list[SPARQLOperationModel] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GraphVersion:
        """Create a new version of an ontology.

        Args:
            ontology_id: Unique identifier for the ontology.
            graph: The RDF graph for this version.
            operations: SPARQL operations that created this version.
            metadata: Additional metadata for this version.

        Returns:
            GraphVersion: The created version.
        """
        version_number = len(self.ontology_versions[ontology_id]) + 1
        version_id = f"{ontology_id}_v{version_number}"

        # Get parent version if it exists
        parent_version_id = None
        if self.ontology_versions[ontology_id]:
            parent_version_id = self.ontology_versions[ontology_id][-1].id

        version = GraphVersion(
            id=version_id,
            graph=graph,
            timestamp=datetime.now(),
            operations=operations or [],
            metadata=metadata or {},
            parent_version_id=parent_version_id,
        )

        self.ontology_versions[ontology_id].append(version)
        logger.info(f"Created ontology version {version_id} with {len(graph)} triples")

        return version

    def create_facts_version(
        self,
        chunk_id: str,
        graph: RDFGraph,
        operations: list[SPARQLOperationModel] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GraphVersion:
        """Create a new version of facts for a chunk.

        Args:
            chunk_id: Unique identifier for the chunk.
            graph: The RDF graph for this version.
            operations: SPARQL operations that created this version.
            metadata: Additional metadata for this version.

        Returns:
            GraphVersion: The created version.
        """
        version_number = len(self.facts_versions[chunk_id]) + 1
        version_id = f"{chunk_id}_v{version_number}"

        # Get parent version if it exists
        parent_version_id = None
        if self.facts_versions[chunk_id]:
            parent_version_id = self.facts_versions[chunk_id][-1].id

        version = GraphVersion(
            id=version_id,
            graph=graph,
            timestamp=datetime.now(),
            operations=operations or [],
            metadata=metadata or {},
            parent_version_id=parent_version_id,
        )

        self.facts_versions[chunk_id].append(version)
        logger.info(f"Created facts version {version_id} with {len(graph)} triples")

        return version

    def get_latest_ontology_version(self, ontology_id: str) -> GraphVersion | None:
        """Get the latest version of an ontology.

        Args:
            ontology_id: The ontology identifier.

        Returns:
            GraphVersion: The latest version, or None if not found.
        """
        versions = self.ontology_versions.get(ontology_id, [])
        return versions[-1] if versions else None

    def get_latest_facts_version(self, chunk_id: str) -> GraphVersion | None:
        """Get the latest version of facts for a chunk.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            GraphVersion: The latest version, or None if not found.
        """
        versions = self.facts_versions.get(chunk_id, [])
        return versions[-1] if versions else None

    def get_ontology_version(
        self, ontology_id: str, version_index: int
    ) -> GraphVersion | None:
        """Get a specific version of an ontology.

        Args:
            ontology_id: The ontology identifier.
            version_index: The version index (0-based).

        Returns:
            GraphVersion: The requested version, or None if not found.
        """
        versions = self.ontology_versions.get(ontology_id, [])
        if 0 <= version_index < len(versions):
            return versions[version_index]
        return None

    def get_facts_version(
        self, chunk_id: str, version_index: int
    ) -> GraphVersion | None:
        """Get a specific version of facts for a chunk.

        Args:
            chunk_id: The chunk identifier.
            version_index: The version index (0-based).

        Returns:
            GraphVersion: The requested version, or None if not found.
        """
        versions = self.facts_versions.get(chunk_id, [])
        if 0 <= version_index < len(versions):
            return versions[version_index]
        return None

    def calculate_ontology_diff(
        self, ontology_id: str, from_version: int, to_version: int
    ) -> GraphDiff:
        """Calculate differences between two ontology versions.

        Args:
            ontology_id: The ontology identifier.
            from_version: The source version index.
            to_version: The target version index.

        Returns:
            GraphDiff: The differences between versions.
        """
        from_ver = self.get_ontology_version(ontology_id, from_version)
        to_ver = self.get_ontology_version(ontology_id, to_version)

        if not from_ver or not to_ver:
            raise ValueError(f"Invalid version indices for ontology {ontology_id}")

        return self._calculate_graph_diff(from_ver.graph, to_ver.graph)

    def calculate_facts_diff(
        self, chunk_id: str, from_version: int, to_version: int
    ) -> GraphDiff:
        """Calculate differences between two facts versions.

        Args:
            chunk_id: The chunk identifier.
            from_version: The source version index.
            to_version: The target version index.

        Returns:
            GraphDiff: The differences between versions.
        """
        from_ver = self.get_facts_version(chunk_id, from_version)
        to_ver = self.get_facts_version(chunk_id, to_version)

        if not from_ver or not to_ver:
            raise ValueError(f"Invalid version indices for chunk {chunk_id}")

        return self._calculate_graph_diff(from_ver.graph, to_ver.graph)

    def _calculate_graph_diff(
        self, from_graph: RDFGraph, to_graph: RDFGraph
    ) -> GraphDiff:
        """Calculate differences between two graphs.

        Args:
            from_graph: The source graph.
            to_graph: The target graph.

        Returns:
            GraphDiff: The differences between graphs.
        """
        from_triples = set(from_graph)
        to_triples = set(to_graph)

        added_triples = list(to_triples - from_triples)
        removed_triples = list(from_triples - to_triples)

        # For modified triples, we need to identify triples that changed
        # This is a simplified approach - in practice, you might want more sophisticated matching
        modified_triples = []

        # Get namespace differences
        from_namespaces = {k: str(v) for k, v in from_graph.namespaces()}
        to_namespaces = {k: str(v) for k, v in to_graph.namespaces()}

        added_namespaces = {
            k: v for k, v in to_namespaces.items() if k not in from_namespaces
        }
        removed_namespaces = {
            k: v for k, v in from_namespaces.items() if k not in to_namespaces
        }

        return GraphDiff(
            added_triples=added_triples,
            removed_triples=removed_triples,
            modified_triples=modified_triples,
            added_namespaces=added_namespaces,
            removed_namespaces=removed_namespaces,
        )

    def get_ontology_version_count(self, ontology_id: str) -> int:
        """Get the number of versions for an ontology.

        Args:
            ontology_id: The ontology identifier.

        Returns:
            int: The number of versions.
        """
        return len(self.ontology_versions.get(ontology_id, []))

    def get_facts_version_count(self, chunk_id: str) -> int:
        """Get the number of versions for facts in a chunk.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            int: The number of versions.
        """
        return len(self.facts_versions.get(chunk_id, []))

    def get_all_ontology_ids(self) -> list[str]:
        """Get all ontology identifiers.

        Returns:
            list[str]: All ontology identifiers.
        """
        return list(self.ontology_versions.keys())

    def get_all_chunk_ids(self) -> list[str]:
        """Get all chunk identifiers.

        Returns:
            list[str]: All chunk identifiers.
        """
        return list(self.facts_versions.keys())

    def delete_ontology_versions(self, ontology_id: str, keep_latest: bool = True):
        """Delete all versions of an ontology.

        Args:
            ontology_id: The ontology identifier.
            keep_latest: If True, keep only the latest version.
        """
        if ontology_id in self.ontology_versions:
            if keep_latest and len(self.ontology_versions[ontology_id]) > 1:
                # Keep only the latest version
                latest_version = self.ontology_versions[ontology_id][-1]
                self.ontology_versions[ontology_id] = [latest_version]
                logger.info(f"Deleted all but latest version of ontology {ontology_id}")
            else:
                del self.ontology_versions[ontology_id]
                logger.info(f"Deleted all versions of ontology {ontology_id}")

    def delete_facts_versions(self, chunk_id: str, keep_latest: bool = True):
        """Delete all versions of facts for a chunk.

        Args:
            chunk_id: The chunk identifier.
            keep_latest: If True, keep only the latest version.
        """
        if chunk_id in self.facts_versions:
            if keep_latest and len(self.facts_versions[chunk_id]) > 1:
                # Keep only the latest version
                latest_version = self.facts_versions[chunk_id][-1]
                self.facts_versions[chunk_id] = [latest_version]
                logger.info(
                    f"Deleted all but latest version of facts for chunk {chunk_id}"
                )
            else:
                del self.facts_versions[chunk_id]
                logger.info(f"Deleted all versions of facts for chunk {chunk_id}")

    def get_version_statistics(self) -> VersionStatistics:
        """Get statistics about all versions.

        Returns:
            VersionStatistics: Version statistics.
        """
        ontology_details = {}
        for ontology_id, versions in self.ontology_versions.items():
            ontology_details[ontology_id] = VersionDetails(
                version_count=len(versions),
                latest_size=versions[-1].get_size() if versions else 0,
                latest_timestamp=versions[-1].timestamp.isoformat()
                if versions
                else None,
            )

        chunk_details = {}
        for chunk_id, versions in self.facts_versions.items():
            chunk_details[chunk_id] = VersionDetails(
                version_count=len(versions),
                latest_size=versions[-1].get_size() if versions else 0,
                latest_timestamp=versions[-1].timestamp.isoformat()
                if versions
                else None,
            )

        return VersionStatistics(
            total_ontologies=len(self.ontology_versions),
            total_chunks=len(self.facts_versions),
            total_ontology_versions=sum(
                len(versions) for versions in self.ontology_versions.values()
            ),
            total_facts_versions=sum(
                len(versions) for versions in self.facts_versions.values()
            ),
            ontology_details=ontology_details,
            chunk_details=chunk_details,
        )
