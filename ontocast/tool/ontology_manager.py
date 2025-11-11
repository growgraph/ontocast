"""Ontology management tool for OntoCast.

This module provides functionality for managing multiple ontologies, including
loading, updating, and retrieving ontologies by name or IRI. Tracks version
lineage using hash-based identifiers.
"""

import logging

from pydantic import Field

from ..onto.extras import NULL_ONTOLOGY
from ..onto.ontology import Ontology
from ..onto.rdfgraph import RDFGraph
from ..onto.util import derive_ontology_id
from .onto import Tool

logger = logging.getLogger(__name__)


class OntologyManager(Tool):
    """Manager for handling multiple ontologies with version tracking.

    This class provides functionality for managing a collection of ontologies,
    tracking version lineage using hash-based identifiers. For each ontology_id,
    it maintains a tree/graph of all versions identified by their hashes.

    Attributes:
        ontology_versions: Dictionary mapping ontology_id to list of all
            ontology versions (identified by hash). Each ontology_id can have
            multiple versions forming a lineage tree.
    """

    ontology_versions: dict[str, list[Ontology]] = Field(default_factory=dict)

    def __init__(self, **kwargs):
        """Initialize the ontology manager.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        # Cache dictionary mapping ontology_id to hash of freshest terminal ontology.
        # Updated incrementally when ontologies are added.
        self._cached_ontologies: dict[str, str] = {}

    def __contains__(self, item):
        """Check if an item (ontology_id or IRI) is in the ontology manager.

        Args:
            item: The ontology_id or IRI to check.

        Returns:
            bool: True if the item exists in any version of any ontology.
        """
        # Check by ontology_id
        if item in self.ontology_versions:
            return True
        # Check by IRI in all versions
        for versions in self.ontology_versions.values():
            for o in versions:
                if o.iri == item:
                    return True
        return False

    def add_ontology(self, ontology: Ontology) -> None:
        """Add an ontology to the version tree for its ontology_id.

        If an ontology with the same hash already exists, it is not added again.
        The ontology is added to the version tree for its ontology_id.
        Ensures that created_at is set if not already present.

        Args:
            ontology: The ontology to add.
        """
        if not ontology.ontology_id:
            logger.warning(
                f"Cannot add ontology without ontology_id (IRI: {ontology.iri})"
            )
            return

        if not ontology.hash:
            logger.warning(
                f"Cannot add ontology without hash (ontology_id: {ontology.ontology_id})"
            )
            return

        # Ensure created_at is set
        if not ontology.created_at:
            from datetime import datetime, timezone

            ontology.created_at = datetime.now(timezone.utc)
            logger.debug(
                f"Set created_at for ontology {ontology.ontology_id} with hash {ontology.hash[:8]}..."
            )

        if ontology.ontology_id not in self.ontology_versions:
            self.ontology_versions[ontology.ontology_id] = []

        # Check if this hash already exists
        existing_hashes = {o.hash for o in self.ontology_versions[ontology.ontology_id]}
        if ontology.hash not in existing_hashes:
            self.ontology_versions[ontology.ontology_id].append(ontology)
            # Update cache for this specific ontology_id (store hash only)
            freshest = self.get_freshest_terminal_ontology(ontology.ontology_id)
            if freshest and freshest.hash:
                self._cached_ontologies[ontology.ontology_id] = freshest.hash
            logger.debug(
                f"Added ontology {ontology.ontology_id} with hash {ontology.hash[:8]}..."
            )
        else:
            logger.debug(
                f"Ontology {ontology.ontology_id} with hash {ontology.hash[:8]}... already exists"
            )

    def get_terminal_ontologies(self, ontology_id: str | None = None) -> list[Ontology]:
        """Get terminal (leaf) ontologies in the version graph.

        Terminal ontologies are those that are not parents of any other ontology
        in the version tree. If ontology_id is provided, returns terminals for
        that ontology only; otherwise returns terminals for all ontologies.

        Args:
            ontology_id: Optional ontology_id to filter by.

        Returns:
            list[Ontology]: List of terminal ontologies.
        """
        if ontology_id:
            if ontology_id not in self.ontology_versions:
                return []
            ontologies = self.ontology_versions[ontology_id]
        else:
            ontologies = [
                o for versions in self.ontology_versions.values() for o in versions
            ]

        if not ontologies:
            return []

        # Build a set of all parent hashes
        all_parent_hashes = set()
        for o in ontologies:
            all_parent_hashes.update(o.parent_hashes)

        # Terminal nodes are those whose hash is not in any parent_hashes
        terminal_hashes = {o.hash for o in ontologies} - all_parent_hashes

        return [o for o in ontologies if o.hash in terminal_hashes]

    def get_freshest_terminal_ontology(
        self, ontology_id: str | None = None
    ) -> Ontology | None:
        """Get the freshest terminal ontology based on created_at timestamp.

        Returns the terminal ontology with the most recent `created_at` timestamp.
        If multiple terminal ontologies exist, returns the one that was most recently
        created. If no created_at is set, falls back to the first terminal ontology.

        Args:
            ontology_id: Optional ontology_id to filter by. If None, searches across
                all ontologies.

        Returns:
            Ontology: The freshest terminal ontology, or None if no terminal
                ontologies exist.
        """
        terminals = self.get_terminal_ontologies(ontology_id)

        if not terminals:
            return None

        # Filter out ontologies without created_at and sort by created_at
        with_timestamp = [o for o in terminals if o.created_at is not None]
        without_timestamp = [o for o in terminals if o.created_at is None]

        if with_timestamp:
            # Sort by created_at descending (most recent first)
            # Type assertion: we know created_at is not None due to filter above
            from datetime import datetime
            from typing import cast

            freshest = max(
                with_timestamp,
                key=lambda o: cast(datetime, o.created_at),
            )
            return freshest
        elif without_timestamp:
            # Fallback to first terminal if no timestamps available
            return without_timestamp[0]

        return None

    def get_ontology_versions(self, ontology_id: str) -> list[Ontology]:
        """Get all versions of an ontology by ontology_id.

        Args:
            ontology_id: The ontology_id to retrieve versions for.

        Returns:
            list[Ontology]: List of all versions of the ontology.
        """
        return self.ontology_versions.get(ontology_id, [])

    def get_lineage_graph(self, ontology_id: str):
        """Get the lineage graph for a specific ontology_id.

        Args:
            ontology_id: The ontology_id to get the lineage graph for.

        Returns:
            networkx.DiGraph: The lineage graph for the ontology, or None if not found.
        """
        if ontology_id not in self.ontology_versions:
            return None

        return Ontology.build_lineage_graph(self.ontology_versions[ontology_id])

    def get_ontology(
        self,
        ontology_id: str | None = None,
        ontology_iri: str | None = None,
        hash: str | None = None,
    ) -> Ontology:
        """Get an ontology by its short name, IRI, or hash.

        If hash is provided, returns the specific version. Otherwise, returns
        a terminal (most recent) version if multiple versions exist.

        Args:
            ontology_id: The short name of the ontology to retrieve (optional).
            ontology_iri: The IRI of the ontology to retrieve (optional).
            hash: The hash of a specific version to retrieve (optional).

        Returns:
            Ontology: The matching ontology if found, NULL_ONTOLOGY otherwise.
        """
        # If hash is provided, search by hash first
        if hash:
            for versions in self.ontology_versions.values():
                for o in versions:
                    if o.hash == hash:
                        return o

        # Try by ontology_id if provided
        if ontology_id is not None:
            if ontology_id in self.ontology_versions:
                versions = self.ontology_versions[ontology_id]
                if hash:
                    # Find specific version by hash
                    for o in versions:
                        if o.hash == hash:
                            return o
                else:
                    # Return terminal version (most recent)
                    terminals = self.get_terminal_ontologies(ontology_id)
                    if terminals:
                        return terminals[0]
                    # Fallback to first version if no terminals
                    if versions:
                        return versions[0]

                # If IRI is also provided, check consistency
                if ontology_iri:
                    derived_id = derive_ontology_id(ontology_iri)
                    if ontology_id != derived_id:
                        logger.warning(
                            f"Ontology id '{ontology_id}' does not match id derived from IRI '{ontology_iri}': '{derived_id}'"
                        )

        # Try by IRI if provided
        if ontology_iri is not None:
            for versions in self.ontology_versions.values():
                for o in versions:
                    if o.iri == ontology_iri:
                        return o

        # Not found
        return NULL_ONTOLOGY

    def get_ontology_names(self) -> list[str]:
        """Get a list of all ontology short names.

        Returns:
            list[str]: List of ontology short names.
        """
        return list(self.ontology_versions.keys())

    @property
    def has_ontologies(self) -> bool:
        """Check if there are any ontologies available.

        Returns:
            bool: True if there are any ontologies, False otherwise.
        """
        return len(self._cached_ontologies) > 0 or len(self.ontology_versions) > 0

    @property
    def ontologies(self) -> list[Ontology]:
        """Get freshest terminal ontology for each ontology_id.

        This property provides backward compatibility with code that expects
        a list of ontologies. Returns the freshest (most recently created)
        terminal version for each ontology_id.

        The result is cached per ontology_id (as hashes) and updated incrementally
        when ontologies are added.

        Returns:
            list[Ontology]: List of freshest terminal ontologies, one per ontology_id.
        """
        result = []

        # Ensure cache is up to date for all ontology_ids
        for ontology_id in self.ontology_versions.keys():
            if ontology_id not in self._cached_ontologies:
                freshest = self.get_freshest_terminal_ontology(ontology_id)
                if freshest and freshest.hash:
                    self._cached_ontologies[ontology_id] = freshest.hash

        # Remove entries for ontology_ids that no longer exist
        cached_ids = set(self._cached_ontologies.keys())
        current_ids = set(self.ontology_versions.keys())
        for removed_id in cached_ids - current_ids:
            del self._cached_ontologies[removed_id]

        # Look up actual ontology objects by hash
        for ontology_id, cached_hash in self._cached_ontologies.items():
            if ontology_id in self.ontology_versions:
                # Find ontology with matching hash
                for ontology in self.ontology_versions[ontology_id]:
                    if ontology.hash == cached_hash:
                        result.append(ontology)
                        break

        return result

    def update_ontology(self, ontology_id: str, ontology_addendum: RDFGraph):
        """Update an existing ontology with additional triples.

        Note: This method is deprecated. Use add_ontology() with a new version
        that has the current hash in parent_hashes instead.

        Args:
            ontology_id: The short name of the ontology to update.
            ontology_addendum: The RDF graph containing additional triples to add.
        """
        logger.warning(
            "update_ontology() is deprecated. Use add_ontology() with version tracking instead."
        )
        terminals = self.get_terminal_ontologies(ontology_id)
        if terminals:
            terminals[0] += ontology_addendum
            # Update cache for this ontology_id (though this method is deprecated)
            freshest = self.get_freshest_terminal_ontology(ontology_id)
            if freshest and freshest.hash:
                self._cached_ontologies[ontology_id] = freshest.hash
