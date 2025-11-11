"""Ontology management tool for OntoCast.

This module provides functionality for managing multiple ontologies, including
loading, updating, and retrieving ontologies by name or IRI. Tracks version
lineage using hash-based identifiers.
"""

import logging

from pydantic import Field

from ..onto.null import NULL_ONTOLOGY
from ..onto.ontology import Ontology
from ..onto.rdfgraph import RDFGraph
from .onto import Tool

logger = logging.getLogger(__name__)


class OntologyManager(Tool):
    """Manager for handling multiple ontologies with version tracking.

    This class provides functionality for managing a collection of ontologies,
    tracking version lineage using hash-based identifiers. For each IRI,
    it maintains a tree/graph of all versions identified by their hashes.

    Attributes:
        ontology_versions: Dictionary mapping IRI to list of all
            ontology versions (identified by hash). Each IRI can have
            multiple versions forming a lineage tree.
    """

    ontology_versions: dict[str, list[Ontology]] = Field(default_factory=dict)

    def __init__(self, **kwargs):
        """Initialize the ontology manager.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        # Cache dictionary mapping IRI to hash of freshest terminal ontology.
        # Updated incrementally when ontologies are added.
        self._cached_ontologies: dict[str, str] = {}

    def __contains__(self, item):
        """Check if an item (IRI or ontology_id) is in the ontology manager.

        Args:
            item: The IRI or ontology_id to check.

        Returns:
            bool: True if the item exists in any version of any ontology.
        """
        # Check by IRI (primary key)
        if item in self.ontology_versions:
            return True
        # Check by ontology_id (fallback for backward compatibility)
        for versions in self.ontology_versions.values():
            for o in versions:
                if o.ontology_id == item:
                    return True
        return False

    def add_ontology(self, ontology: Ontology) -> None:
        """Add an ontology to the version tree for its IRI.

        If an ontology with the same hash already exists, it is not added again.
        The ontology is added to the version tree for its IRI.
        Ensures that created_at is set if not already present.

        Args:
            ontology: The ontology to add.
        """
        if not ontology.iri or ontology.iri == NULL_ONTOLOGY.iri:
            logger.warning(
                f"Cannot add ontology without valid IRI (ontology_id: {ontology.ontology_id})"
            )
            return

        if not ontology.hash:
            logger.warning(f"Cannot add ontology without hash (IRI: {ontology.iri})")
            return

        # Ensure created_at is set
        if not ontology.created_at:
            from datetime import datetime, timezone

            ontology.created_at = datetime.now(timezone.utc)
            logger.debug(
                f"Set created_at for ontology {ontology.iri} with hash {ontology.hash[:8]}..."
            )

        if ontology.iri not in self.ontology_versions:
            self.ontology_versions[ontology.iri] = []

        # Check if this hash already exists
        existing_hashes = {o.hash for o in self.ontology_versions[ontology.iri]}
        if ontology.hash not in existing_hashes:
            self.ontology_versions[ontology.iri].append(ontology)
            # Update cache for this specific IRI (store hash only)
            freshest = self.get_freshest_terminal_ontology_by_iri(ontology.iri)
            if freshest and freshest.hash:
                self._cached_ontologies[ontology.iri] = freshest.hash
            logger.debug(
                f"Added ontology {ontology.iri} with hash {ontology.hash[:8]}..."
            )
        else:
            logger.debug(
                f"Ontology {ontology.iri} with hash {ontology.hash[:8]}... already exists"
            )

    def get_terminal_ontologies_by_iri(self, iri: str | None = None) -> list[Ontology]:
        """Get terminal (leaf) ontologies in the version graph.

        Terminal ontologies are those that are not parents of any other ontology
        in the version tree. If iri is provided, returns terminals for
        that ontology only; otherwise returns terminals for all ontologies.

        Args:
            iri: Optional IRI to filter by.

        Returns:
            list[Ontology]: List of terminal ontologies.
        """
        if iri:
            if iri not in self.ontology_versions:
                return []
            ontologies = self.ontology_versions[iri]
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

    def get_terminal_ontologies(self, ontology_id: str | None = None) -> list[Ontology]:
        """Get terminal (leaf) ontologies by ontology_id (backward compatibility wrapper).

        Args:
            ontology_id: Optional ontology_id to filter by.

        Returns:
            list[Ontology]: List of terminal ontologies.
        """
        if ontology_id:
            # Find IRI(s) matching this ontology_id
            matching_iris = [
                iri
                for iri, versions in self.ontology_versions.items()
                if any(o.ontology_id == ontology_id for o in versions)
            ]
            if not matching_iris:
                return []
            # Get terminals for all matching IRIs
            all_terminals = []
            for iri in matching_iris:
                all_terminals.extend(self.get_terminal_ontologies_by_iri(iri))
            return all_terminals
        else:
            return self.get_terminal_ontologies_by_iri(None)

    def get_freshest_terminal_ontology_by_iri(
        self, iri: str | None = None
    ) -> Ontology | None:
        """Get the freshest terminal ontology based on created_at timestamp.

        Returns the terminal ontology with the most recent `created_at` timestamp.
        If multiple terminal ontologies exist, returns the one that was most recently
        created. If no created_at is set, falls back to the first terminal ontology.

        Args:
            iri: Optional IRI to filter by. If None, searches across
                all ontologies.

        Returns:
            Ontology: The freshest terminal ontology, or None if no terminal
                ontologies exist.
        """
        terminals = self.get_terminal_ontologies_by_iri(iri)

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

    def get_freshest_terminal_ontology(
        self, ontology_id: str | None = None
    ) -> Ontology | None:
        """Get the freshest terminal ontology by ontology_id (backward compatibility wrapper).

        Args:
            ontology_id: Optional ontology_id to filter by.

        Returns:
            Ontology: The freshest terminal ontology, or None if no terminal
                ontologies exist.
        """
        if ontology_id:
            # Find IRI(s) matching this ontology_id
            matching_iris = [
                iri
                for iri, versions in self.ontology_versions.items()
                if any(o.ontology_id == ontology_id for o in versions)
            ]
            if not matching_iris:
                return None
            # Get freshest for all matching IRIs and return the most recent
            candidates = []
            for iri in matching_iris:
                freshest = self.get_freshest_terminal_ontology_by_iri(iri)
                if freshest:
                    candidates.append(freshest)
            if not candidates:
                return None
            # Return the most recent among all candidates
            from datetime import datetime
            from typing import cast

            with_timestamp = [o for o in candidates if o.created_at is not None]
            if with_timestamp:
                return max(with_timestamp, key=lambda o: cast(datetime, o.created_at))
            return candidates[0]
        else:
            return self.get_freshest_terminal_ontology_by_iri(None)

    def get_ontology_versions_by_iri(self, iri: str) -> list[Ontology]:
        """Get all versions of an ontology by IRI.

        Args:
            iri: The IRI to retrieve versions for.

        Returns:
            list[Ontology]: List of all versions of the ontology.
        """
        return self.ontology_versions.get(iri, [])

    def get_ontology_versions(self, ontology_id: str) -> list[Ontology]:
        """Get all versions of an ontology by ontology_id (backward compatibility wrapper).

        Args:
            ontology_id: The ontology_id to retrieve versions for.

        Returns:
            list[Ontology]: List of all versions of the ontology.
        """
        # Find all IRIs matching this ontology_id
        all_versions = []
        for iri, versions in self.ontology_versions.items():
            if any(o.ontology_id == ontology_id for o in versions):
                all_versions.extend(versions)
        return all_versions

    def get_lineage_graph_by_iri(self, iri: str):
        """Get the lineage graph for a specific IRI.

        Args:
            iri: The IRI to get the lineage graph for.

        Returns:
            networkx.DiGraph: The lineage graph for the ontology, or None if not found.
        """
        if iri not in self.ontology_versions:
            return None

        return Ontology.build_lineage_graph(self.ontology_versions[iri])

    def get_lineage_graph(self, ontology_id: str):
        """Get the lineage graph for a specific ontology_id (backward compatibility wrapper).

        Args:
            ontology_id: The ontology_id to get the lineage graph for.

        Returns:
            networkx.DiGraph: The lineage graph for the ontology, or None if not found.
        """
        # Find first IRI matching this ontology_id
        for iri, versions in self.ontology_versions.items():
            if any(o.ontology_id == ontology_id for o in versions):
                return Ontology.build_lineage_graph(versions)
        return None

    def get_ontology(
        self,
        ontology_id: str | None = None,
        ontology_iri: str | None = None,
        hash: str | None = None,
    ) -> Ontology:
        """Get an ontology by its IRI, ontology_id, or hash.

        If hash is provided, returns the specific version. Otherwise, returns
        a terminal (most recent) version if multiple versions exist.
        IRI is preferred over ontology_id for lookup.

        Args:
            ontology_id: The short name of the ontology to retrieve (optional, for backward compatibility).
            ontology_iri: The IRI of the ontology to retrieve (preferred).
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

        # Try by IRI first (preferred method)
        if ontology_iri is not None:
            if ontology_iri in self.ontology_versions:
                versions = self.ontology_versions[ontology_iri]
                if hash:
                    # Find specific version by hash
                    for o in versions:
                        if o.hash == hash:
                            return o
                else:
                    # Return terminal version (most recent)
                    terminals = self.get_terminal_ontologies_by_iri(ontology_iri)
                    if terminals:
                        return terminals[0]
                    # Fallback to first version if no terminals
                    if versions:
                        return versions[0]

        # Try by ontology_id if provided (backward compatibility)
        if ontology_id is not None:
            # Find IRI(s) matching this ontology_id
            matching_iris = [
                iri
                for iri, versions in self.ontology_versions.items()
                if any(o.ontology_id == ontology_id for o in versions)
            ]
            if matching_iris:
                # Use first matching IRI
                iri = matching_iris[0]
                versions = self.ontology_versions[iri]
                if hash:
                    # Find specific version by hash
                    for o in versions:
                        if o.hash == hash:
                            return o
                else:
                    # Return terminal version (most recent)
                    terminals = self.get_terminal_ontologies_by_iri(iri)
                    if terminals:
                        return terminals[0]
                    # Fallback to first version if no terminals
                    if versions:
                        return versions[0]

                # If IRI is also provided, check consistency
                if ontology_iri and ontology_iri != iri:
                    logger.warning(
                        f"Ontology id '{ontology_id}' matches IRI '{iri}' but different IRI '{ontology_iri}' was provided"
                    )

        # Not found
        return NULL_ONTOLOGY

    def get_ontology_iris(self) -> list[str]:
        """Get a list of all ontology IRIs.

        Returns:
            list[str]: List of ontology IRIs.
        """
        return list(self.ontology_versions.keys())

    def get_ontology_names(self) -> list[str]:
        """Get a list of all ontology short names (backward compatibility wrapper).

        Returns:
            list[str]: List of unique ontology short names.
        """
        names = set()
        for versions in self.ontology_versions.values():
            for o in versions:
                if o.ontology_id:
                    names.add(o.ontology_id)
        return sorted(list(names))

    @property
    def has_ontologies(self) -> bool:
        """Check if there are any ontologies available.

        Returns:
            bool: True if there are any ontologies, False otherwise.
        """
        return len(self._cached_ontologies) > 0 or len(self.ontology_versions) > 0

    @property
    def ontologies(self) -> list[Ontology]:
        """Get freshest terminal ontology for each IRI.

        This property provides backward compatibility with code that expects
        a list of ontologies. Returns the freshest (most recently created)
        terminal version for each IRI.

        The result is cached per IRI (as hashes) and updated incrementally
        when ontologies are added.

        Returns:
            list[Ontology]: List of freshest terminal ontologies, one per IRI.
        """
        result = []

        # Ensure cache is up to date for all IRIs
        for iri in self.ontology_versions.keys():
            if iri not in self._cached_ontologies:
                freshest = self.get_freshest_terminal_ontology_by_iri(iri)
                if freshest and freshest.hash:
                    self._cached_ontologies[iri] = freshest.hash

        # Remove entries for IRIs that no longer exist
        cached_iris = set(self._cached_ontologies.keys())
        current_iris = set(self.ontology_versions.keys())
        for removed_iri in cached_iris - current_iris:
            del self._cached_ontologies[removed_iri]

        # Look up actual ontology objects by hash
        for iri, cached_hash in self._cached_ontologies.items():
            if iri in self.ontology_versions:
                # Find ontology with matching hash
                for ontology in self.ontology_versions[iri]:
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
            # Update cache for the IRI (though this method is deprecated)
            iri = terminals[0].iri
            freshest = self.get_freshest_terminal_ontology_by_iri(iri)
            if freshest and freshest.hash:
                self._cached_ontologies[iri] = freshest.hash
