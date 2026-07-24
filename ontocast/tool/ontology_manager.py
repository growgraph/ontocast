"""Ontology management tool for OntoCast.

This module provides functionality for managing multiple ontologies, including
loading, updating, and retrieving ontologies by name or IRI. Tracks version
lineage using hash-based identifiers.
"""

import logging
from copy import deepcopy
from typing import TYPE_CHECKING

from pydantic import Field

from ..onto.null import NULL_ONTOLOGY
from ..onto.ontology import Ontology
from ..onto.rdfgraph import RDFGraph
from ..onto.util import normalize_ontology_iri
from .onto import Tool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever


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
        self._patch_retriever: OntologyPatchRetriever | None = None
        # Canonical short handle per IRI (ontology_id); prefix may differ.
        self._iri_to_ontology_id: dict[str, str] = {}
        # Lowercased alias (ontology_id, author prefix, …) → IRI.
        self._alias_to_iri: dict[str, str] = {}
        # Preferred author prefix per namespace URI (for sanitize preference).
        self._namespace_to_author_prefix: dict[str, str] = {}

    @staticmethod
    def _primary_ontology_id(ontology: Ontology) -> str:
        identity = (ontology.ontology_id or "").strip().lower()
        if not identity:
            raise ValueError(
                "Ontology identity is missing: ontology_id is required for catalog registration"
            )
        return identity

    def _collect_aliases(self, ontology: Ontology) -> list[str]:
        aliases: list[str] = []
        for candidate in (ontology.ontology_id, ontology.prefix):
            if not candidate:
                continue
            cleaned = candidate.strip().lower()
            if cleaned and cleaned not in aliases:
                aliases.append(cleaned)
        return aliases

    def validate_identity_uniqueness(self, ontology: Ontology) -> None:
        """Validate catalog IRI and alias uniqueness across the manager.

        Same IRI may not change its primary ``ontology_id``. The same alias
        may not point at two different IRIs. Author ``prefix`` may differ from
        ``ontology_id`` (both register as aliases of the same IRI).
        """
        iri = (ontology.iri or "").strip()
        if not iri:
            raise ValueError("Ontology IRI is missing")
        if iri == NULL_ONTOLOGY.iri:
            raise ValueError("Null ontology IRI cannot be registered")

        primary = self._primary_ontology_id(ontology)

        existing_primary = self._iri_to_ontology_id.get(iri)
        if existing_primary is not None and existing_primary != primary:
            raise ValueError(
                "Ontology identity conflict: IRI "
                f"'{iri}' is already bound to identity '{existing_primary}', "
                f"received '{primary}'"
            )

        for alias in self._collect_aliases(ontology):
            existing_iri = self._alias_to_iri.get(alias)
            if existing_iri is not None and existing_iri != iri:
                raise ValueError(
                    "Ontology identity conflict: identity "
                    f"'{alias}' is already bound to IRI '{existing_iri}', "
                    f"received '{iri}'"
                )

    def _register_identity(self, ontology: Ontology) -> None:
        iri = ontology.iri.strip()
        primary = self._primary_ontology_id(ontology)
        self._iri_to_ontology_id[iri] = primary
        for alias in self._collect_aliases(ontology):
            self._alias_to_iri[alias] = iri
        # Also allow looking up by the raw IRI string and its normalized form.
        self._alias_to_iri[iri.lower()] = iri
        normalized = normalize_ontology_iri(iri).lower()
        if normalized:
            self._alias_to_iri[normalized] = iri
        prefix = ontology.prefix
        if prefix and ontology.namespace:
            self._namespace_to_author_prefix[str(ontology.namespace)] = prefix

    def resolve_ontology_ref(self, ref: str) -> str | None:
        """Resolve an absolute IRI or registered alias to a catalog ontology IRI."""
        if not ref or not str(ref).strip():
            return None
        cleaned = str(ref).strip()
        if cleaned in self.ontology_versions:
            return cleaned
        normalized = normalize_ontology_iri(cleaned)
        if normalized in self.ontology_versions:
            return normalized
        for key in (cleaned.lower(), normalized.lower()):
            iri = self._alias_to_iri.get(key)
            if iri is not None:
                return iri
        return None

    def author_prefix_for_namespace(self, namespace: str) -> str | None:
        """Return the catalog-registered author prefix for a namespace, if any."""
        direct = self._namespace_to_author_prefix.get(namespace)
        if direct is not None:
            return direct
        stripped = namespace.rstrip("/#")
        for key, value in self._namespace_to_author_prefix.items():
            if key.rstrip("/#") == stripped:
                return value
        return None

    @property
    def preferred_namespace_prefixes(self) -> dict[str, str]:
        """Namespace URI → author prefix for sanitize preference."""
        return dict(self._namespace_to_author_prefix)

    def __contains__(self, item):
        """Check if an item (IRI or alias) is in the ontology manager.

        Args:
            item: The IRI, ontology_id, or author prefix to check.

        Returns:
            bool: True if the item resolves to a tracked ontology IRI.
        """
        return self.resolve_ontology_ref(str(item)) is not None

    def add_ontology(
        self, ontology: Ontology, *, skip_vector_index: bool = False
    ) -> None:
        """Add an ontology to the version tree for its IRI.

        If an ontology with the same hash already exists, it is not added again.
        The ontology is added to the version tree for its IRI.
        Ensures that created_at is set if not already present.

        Args:
            ontology: The ontology to add.
            skip_vector_index: If True, do not call the vector store (caller
                already materialized embeddings, e.g. during ToolBox.initialize).
        """
        if not ontology.iri or ontology.iri == NULL_ONTOLOGY.iri:
            logger.warning(
                f"Cannot add ontology without valid IRI (ontology_id: {ontology.ontology_id})"
            )
            return

        if not ontology.hash:
            logger.warning(f"Cannot add ontology without hash (IRI: {ontology.iri})")
            return

        self.validate_identity_uniqueness(ontology)
        self._register_identity(ontology)

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
            if self._patch_retriever is not None and not skip_vector_index:
                self._patch_retriever.vector_store.reindex_ontology(ontology)
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

    def remove_ontology_by_iri(self, iri: str) -> None:
        """Drop all tracked versions for an ontology IRI and clear caches."""
        self.ontology_versions.pop(iri, None)
        self._cached_ontologies.pop(iri, None)
        self._iri_to_ontology_id.pop(iri, None)
        # Drop all aliases pointing at this IRI.
        stale = [alias for alias, bound in self._alias_to_iri.items() if bound == iri]
        for alias in stale:
            del self._alias_to_iri[alias]
        # Drop author-prefix entries whose IRI matches (by scanning versions was already removed).
        # Namespace map is best-effort; rebuild from remaining ontologies.
        self._namespace_to_author_prefix = {}
        for versions in self.ontology_versions.values():
            if not versions:
                continue
            onto = versions[-1]
            if onto.prefix and onto.namespace:
                self._namespace_to_author_prefix[str(onto.namespace)] = onto.prefix

    def register_vector_store(self, retriever: "OntologyPatchRetriever") -> None:
        """Register a patch retriever for vector context lookups."""
        self._patch_retriever = retriever

    def _effective_patch_top_k(self, top_k: int | None) -> int:
        if top_k is not None:
            return top_k
        if self._patch_retriever is not None:
            return self._patch_retriever.vector_store.store_config.top_k
        return 10

    def get_patch_context(
        self,
        query: str,
        top_k: int | None = None,
        subgraph_depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
    ) -> RDFGraph | None:
        """Retrieve multi-ontology patch context for a query.

        Falls back to the freshest available ontology graph if vector retrieval
        is not configured or yields no atoms.
        """
        graph, _ = self.get_patch_context_with_sources(
            query=query,
            top_k=top_k,
            subgraph_depth=subgraph_depth,
            max_total_triples=max_total_triples,
            estimated_triples_per_query=estimated_triples_per_query,
        )
        return graph

    async def aget_patch_context(
        self,
        query: str,
        top_k: int | None = None,
        subgraph_depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
    ) -> RDFGraph | None:
        """Async variant of :meth:`get_patch_context`."""
        graph, _ = await self.aget_patch_context_with_sources(
            query=query,
            top_k=top_k,
            subgraph_depth=subgraph_depth,
            max_total_triples=max_total_triples,
            estimated_triples_per_query=estimated_triples_per_query,
        )
        return graph

    def get_patch_context_with_sources(
        self,
        query: str,
        top_k: int | None = None,
        subgraph_depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
    ) -> tuple[RDFGraph | None, list[str]]:
        """Retrieve patch context and contributing ontology IRIs."""
        results = self.get_patch_contexts_with_sources(
            queries=[query],
            top_k=top_k,
            subgraph_depth=subgraph_depth,
            max_total_triples=max_total_triples,
            estimated_triples_per_query=estimated_triples_per_query,
        )
        if not results:
            return None, []
        return results[0]

    async def aget_patch_context_with_sources(
        self,
        query: str,
        top_k: int | None = None,
        subgraph_depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
    ) -> tuple[RDFGraph | None, list[str]]:
        """Async variant of :meth:`get_patch_context_with_sources`."""
        results = await self.aget_patch_contexts_with_sources(
            queries=[query],
            top_k=top_k,
            subgraph_depth=subgraph_depth,
            max_total_triples=max_total_triples,
            estimated_triples_per_query=estimated_triples_per_query,
        )
        if not results:
            return None, []
        return results[0]

    def get_patch_contexts_with_sources(
        self,
        queries: list[str],
        top_k: int | None = None,
        subgraph_depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
    ) -> list[tuple[RDFGraph | None, list[str]]]:
        """Retrieve patch contexts for many queries in a batched pass.

        With a patch retriever, the list has length 1 (ensemble graph + sources).
        Without it, length matches ``queries`` (fallback ontology per query).
        """
        if not queries:
            return []
        if self._patch_retriever is not None:
            graph, sources = self._patch_retriever.retrieve_ensemble(
                queries=queries,
                top_k=self._effective_patch_top_k(top_k),
                subgraph_depth=subgraph_depth,
                max_total_triples=max_total_triples,
                estimated_triples_per_query=estimated_triples_per_query,
            )
            return [(graph, sources) if len(graph) > 0 else (RDFGraph(), sources)]

        fallback = self.get_freshest_terminal_ontology_by_iri(None)
        if fallback is None:
            return [(None, []) for _ in queries]
        fallback_graph = deepcopy(fallback.graph)
        return [(deepcopy(fallback_graph), [fallback.iri]) for _ in queries]

    async def aget_patch_contexts_with_sources(
        self,
        queries: list[str],
        top_k: int | None = None,
        subgraph_depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
    ) -> list[tuple[RDFGraph | None, list[str]]]:
        """Async patch retrieval (vector + induced subgraph) for many queries.

        With a patch retriever, returns a one-element list: a single induced graph for
        the union of hits over ``queries``, plus contributing ontology IRIs.
        """
        if not queries:
            return []
        if self._patch_retriever is not None:
            graph, sources = await self._patch_retriever.aretrieve_ensemble(
                queries=queries,
                top_k=self._effective_patch_top_k(top_k),
                subgraph_depth=subgraph_depth,
                max_total_triples=max_total_triples,
                estimated_triples_per_query=estimated_triples_per_query,
            )
            return [(graph, sources) if len(graph) > 0 else (RDFGraph(), sources)]

        fallback = self.get_freshest_terminal_ontology_by_iri(None)
        if fallback is None:
            return [(None, []) for _ in queries]
        fallback_graph = deepcopy(fallback.graph)
        return [(deepcopy(fallback_graph), [fallback.iri]) for _ in queries]

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
        """Get terminal (leaf) ontologies by ontology_id or alias.

        Args:
            ontology_id: Optional ontology_id / alias / IRI to filter by.

        Returns:
            list[Ontology]: List of terminal ontologies.
        """
        if ontology_id:
            iri = self.resolve_ontology_ref(ontology_id)
            if iri is None:
                return []
            return self.get_terminal_ontologies_by_iri(iri)
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
        """Get the freshest terminal ontology by ontology_id, alias, or IRI.

        Args:
            ontology_id: Optional ontology_id / alias / IRI to filter by.

        Returns:
            Ontology: The freshest terminal ontology, or None if no terminal
                ontologies exist.
        """
        if ontology_id:
            iri = self.resolve_ontology_ref(ontology_id)
            if iri is None:
                return None
            return self.get_freshest_terminal_ontology_by_iri(iri)
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
        """Get all versions of an ontology by ontology_id, alias, or IRI.

        Args:
            ontology_id: The ontology_id / alias / IRI to retrieve versions for.

        Returns:
            list[Ontology]: List of all versions of the ontology.
        """
        iri = self.resolve_ontology_ref(ontology_id)
        if iri is None:
            return []
        return self.get_ontology_versions_by_iri(iri)

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
        """Get the lineage graph for a specific ontology_id, alias, or IRI.

        Args:
            ontology_id: The ontology_id / alias / IRI to get the lineage graph for.

        Returns:
            networkx.DiGraph: The lineage graph for the ontology, or None if not found.
        """
        iri = self.resolve_ontology_ref(ontology_id)
        if iri is None:
            return None
        return self.get_lineage_graph_by_iri(iri)

    def get_ontology(
        self,
        ontology_id: str | None = None,
        ontology_iri: str | None = None,
        hash: str | None = None,
    ) -> Ontology:
        """Get an ontology by its IRI, ontology_id/alias, or hash.

        If hash is provided, returns the specific version. Otherwise, returns
        a terminal (most recent) version if multiple versions exist.
        IRI is preferred over ontology_id for lookup.

        Args:
            ontology_id: Short name, author prefix, or IRI (optional).
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

        resolved_iri: str | None = None
        if ontology_iri is not None:
            resolved_iri = self.resolve_ontology_ref(ontology_iri)
        if resolved_iri is None and ontology_id is not None:
            resolved_iri = self.resolve_ontology_ref(ontology_id)

        if resolved_iri is not None and resolved_iri in self.ontology_versions:
            versions = self.ontology_versions[resolved_iri]
            if hash:
                for o in versions:
                    if o.hash == hash:
                        return o
            else:
                terminals = self.get_terminal_ontologies_by_iri(resolved_iri)
                if terminals:
                    return terminals[0]
                if versions:
                    return versions[0]

            if (
                ontology_iri
                and ontology_id
                and self.resolve_ontology_ref(ontology_id) not in (None, resolved_iri)
            ):
                logger.warning(
                    "Ontology id '%s' resolves differently from IRI '%s'",
                    ontology_id,
                    ontology_iri,
                )

        return NULL_ONTOLOGY

    def get_ontology_iris(self) -> list[str]:
        """Get a list of all ontology IRIs.

        Returns:
            list[str]: List of ontology IRIs.
        """
        return list(self.ontology_versions.keys())

    def get_ontology_names(self) -> list[str]:
        """Return unique catalog ``ontology_id`` values currently tracked.

        Returns:
            list[str]: Sorted unique ontology short names.
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
        """Return the freshest terminal ontology for each catalog IRI.

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
