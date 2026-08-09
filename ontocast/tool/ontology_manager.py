"""Ontology management tool for OntoCast.

This module provides functionality for managing multiple ontologies, including
loading, updating, and retrieving ontologies by name or IRI. Tracks version
lineage using hash-based identifiers.
"""

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

from pydantic import Field
from rdflib import URIRef

from ..onto.null import NULL_ONTOLOGY
from ..onto.ontology import Ontology
from ..onto.ontology_header import OntologyHeader
from ..onto.rdfgraph import RDFGraph
from ..onto.util import normalize_ontology_iri
from .onto import Tool
from .triple_manager.core import TripleStoreManager
from .triple_manager.util import dedupe_terminal_ontologies

logger = logging.getLogger(__name__)

#: Distinct ontology selections whose merged graphs stay resident. Content units
#: within a document overwhelmingly repeat the same selection, so a handful of
#: entries absorbs the whole fan-out; the bound only exists to stop a pathological
#: document from pinning one merged graph per unit.
_MERGED_CACHE_MAX_ENTRIES = 8

# Materialized ontology graphs are far smaller than a merged union, but they are
# still whole rdflib graphs; a long-lived server needs an upper bound.
_GRAPH_CACHE_MAX_ENTRIES = 64

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
        self._triple_store_manager: TripleStoreManager | None = None
        # Canonical short handle per IRI (ontology_id); prefix may differ.
        self._iri_to_ontology_id: dict[str, str] = {}
        # Lowercased alias (ontology_id, author prefix, …) → IRI.
        self._alias_to_iri: dict[str, str] = {}
        # Preferred author prefix per namespace URI (for sanitize preference).
        self._namespace_to_author_prefix: dict[str, str] = {}
        # Content-addressed caches. An entry can never go stale on read: a
        # concurrent writer produces a *new* key, which is a miss, never an
        # incorrect hit. Both are bounded -- they hold whole rdflib graphs, and
        # a long-lived server would otherwise grow without limit.
        #
        # _graph_cache is keyed by the header's ``graph_uri`` (see
        # :meth:`_cache_graph`), *not* by ``versioned_iri``: the two coincide
        # only while content hashing is round-trip stable. Eviction must use the
        # same key, so the graph URI each IRI was cached under is tracked here.
        self._graph_cache: OrderedDict[str, Ontology] = OrderedDict()
        self._graph_uris_by_iri: dict[str, set[str]] = {}
        self._merged_cache: OrderedDict[
            frozenset[str], tuple[RDFGraph, dict[str, str]]
        ] = OrderedDict()
        self._graph_cache_hits = 0
        self._graph_cache_misses = 0
        self._merged_cache_hits = 0
        self._merged_cache_misses = 0

    @staticmethod
    def _primary_ontology_id(ontology: Ontology) -> str:
        identity = (ontology.ontology_id or "").strip().lower()
        if not identity:
            raise ValueError(
                "Ontology identity is missing: ontology_id is required for catalog registration"
            )
        return identity

    def _collect_aliases(self, ontology: Ontology) -> list[tuple[str, str]]:
        """Collect ``(alias, kind)`` pairs; kind is ``ontology_id`` or ``prefix``.

        When ``ontology_id`` and author prefix coincide, the alias keeps the
        stricter ``ontology_id`` kind.
        """
        aliases: list[tuple[str, str]] = []
        seen: set[str] = set()
        for candidate, kind in (
            (ontology.ontology_id, "ontology_id"),
            (ontology.prefix, "prefix"),
        ):
            if not candidate:
                continue
            cleaned = candidate.strip().lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                aliases.append((cleaned, kind))
        return aliases

    def validate_identity_uniqueness(self, ontology: Ontology) -> None:
        """Validate catalog IRI and alias uniqueness across the manager.

        Same IRI may not change its primary ``ontology_id``. The same
        ``ontology_id`` alias may not point at two different IRIs. Author
        ``prefix`` may differ from ``ontology_id`` (both register as aliases of
        the same IRI); a *prefix* collision across IRIs does not block ingest —
        the colliding prefix alias is simply skipped at registration and the
        ontology stays addressable by IRI and ``ontology_id``.
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

        for alias, kind in self._collect_aliases(ontology):
            existing_iri = self._alias_to_iri.get(alias)
            if existing_iri is None or existing_iri == iri:
                continue
            if kind == "prefix":
                # Convenience alias only; degrades to IRI-only addressing.
                continue
            raise ValueError(
                "Ontology identity conflict: identity "
                f"'{alias}' is already bound to IRI '{existing_iri}', "
                f"received '{iri}'"
            )

    def _register_identity(self, ontology: Ontology) -> None:
        iri = ontology.iri.strip()
        primary = self._primary_ontology_id(ontology)
        self._iri_to_ontology_id[iri] = primary
        for alias, _kind in self._collect_aliases(ontology):
            existing_iri = self._alias_to_iri.get(alias)
            if existing_iri is not None and existing_iri != iri:
                # validate_identity_uniqueness raises on ontology_id conflicts,
                # so only author-prefix aliases can reach this branch.
                logger.warning(
                    "Author prefix alias '%s' is already bound to IRI %s; "
                    "skipping alias registration for %s (addressable by IRI "
                    "and ontology_id only).",
                    alias,
                    existing_iri,
                    iri,
                )
                continue
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

    def _prepare_ontology_for_catalog(self, ontology: Ontology) -> bool:
        """Validate and register ``ontology``; return True if a new hash was appended."""
        if not ontology.iri or ontology.iri == NULL_ONTOLOGY.iri:
            logger.warning(
                f"Cannot add ontology without valid IRI (ontology_id: {ontology.ontology_id})"
            )
            return False

        if not ontology.hash:
            logger.warning(f"Cannot add ontology without hash (IRI: {ontology.iri})")
            return False

        # Author @prefix names die at the triple-store boundary; persisting them
        # as sh:declare triples here (hash-neutral, idempotent) lets any later
        # export rebind them instead of inventing synthetic stem-derived names.
        ontology.graph.materialize_prefix_declarations(URIRef(ontology.iri))

        self.validate_identity_uniqueness(ontology)
        self._register_identity(ontology)

        if not ontology.created_at:
            ontology.created_at = datetime.now(timezone.utc)
            logger.debug(
                f"Set created_at for ontology {ontology.iri} with hash {ontology.hash[:8]}..."
            )

        if ontology.iri not in self.ontology_versions:
            self.ontology_versions[ontology.iri] = []

        existing_hashes = {o.hash for o in self.ontology_versions[ontology.iri]}
        if ontology.hash in existing_hashes:
            logger.debug(
                f"Ontology {ontology.iri} with hash {ontology.hash[:8]}... already exists"
            )
            return False

        self.ontology_versions[ontology.iri].append(ontology)
        freshest = self.get_freshest_terminal_ontology_by_iri(ontology.iri)
        if freshest and freshest.hash:
            self._cached_ontologies[ontology.iri] = freshest.hash
        logger.debug(f"Added ontology {ontology.iri} with hash {ontology.hash[:8]}...")
        return True

    def _reindex_ontology_sync(self, ontology: Ontology) -> None:
        """Sync vector reindex (caller must ensure no running event loop)."""
        if self._patch_retriever is None:
            return
        self._patch_retriever.vector_store.reindex_ontology(ontology)

    def _ensure_sync_reindex_allowed(self, *, skip_vector_index: bool) -> None:
        """Raise if sync reindex would block a running event loop."""
        if skip_vector_index or self._patch_retriever is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            "add_ontology() cannot reindex inside async code; use await aadd_ontology()"
        )

    async def _reindex_ontology_async(self, ontology: Ontology) -> None:
        if self._patch_retriever is None:
            return
        await asyncio.to_thread(
            self._patch_retriever.vector_store.reindex_ontology, ontology
        )

    def add_ontology(
        self, ontology: Ontology, *, skip_vector_index: bool = False
    ) -> None:
        """Add an ontology to the version tree for its IRI.

        If an ontology with the same hash already exists, it is not added again.
        Ensures that created_at is set if not already present.

        Args:
            ontology: The ontology to add.
            skip_vector_index: If True, do not call the vector store (caller
                already materialized embeddings, e.g. during ToolBox.initialize).

        Raises:
            RuntimeError: If vector reindex would run while an event loop is
                already active. Use :meth:`aadd_ontology` from async code.
        """
        self._ensure_sync_reindex_allowed(skip_vector_index=skip_vector_index)
        if not self._prepare_ontology_for_catalog(ontology):
            return
        if not skip_vector_index:
            self._reindex_ontology_sync(ontology)

    async def aadd_ontology(
        self, ontology: Ontology, *, skip_vector_index: bool = False
    ) -> None:
        """Async variant of :meth:`add_ontology` (reindex off the event loop)."""
        if not self._prepare_ontology_for_catalog(ontology):
            return
        if not skip_vector_index:
            await self._reindex_ontology_async(ontology)

    def remove_ontology_by_iri(self, iri: str) -> None:
        """Drop all tracked versions for an ontology IRI and clear caches."""
        # Evict under the key entries were *inserted* with. Popping
        # ``versioned_iri`` here -- as this did once -- silently missed every
        # entry whenever the recomputed hash differed from the stored graph URI,
        # leaving a removed ontology still resolvable from cache.
        for graph_uri in self._graph_uris_by_iri.pop(iri, set()):
            self._graph_cache.pop(graph_uri, None)
        for ontology in self.ontology_versions.get(iri, []):
            self._graph_cache.pop(ontology.versioned_iri, None)
        stale_merges = [
            key
            for key in self._merged_cache
            # An ontology with no hash falls back to the bare IRI as its
            # versioned IRI, so match that exactly as well as the `#hash` form.
            if any(
                versioned == iri or versioned.startswith(f"{iri}#") for versioned in key
            )
        ]
        for key in stale_merges:
            del self._merged_cache[key]
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

    def register_triple_store(self, manager: TripleStoreManager | None) -> None:
        """Register the triple store this catalog reads through on a cache miss."""
        self._triple_store_manager = manager

    def reset_catalog(self) -> None:
        """Drop every tracked ontology, identity binding, and cached graph.

        Called when the active tenant/project changes: the catalog, the alias
        collision ledger, and the graph caches are all partition-scoped, and
        carrying them across a switch leaks one tenant's ontologies into another's
        requests.
        """
        self.ontology_versions.clear()
        self._cached_ontologies.clear()
        self._iri_to_ontology_id.clear()
        self._alias_to_iri.clear()
        self._namespace_to_author_prefix.clear()
        self._graph_cache.clear()
        self._graph_uris_by_iri.clear()
        self._merged_cache.clear()

    def _require_triple_store(self) -> TripleStoreManager:
        if self._triple_store_manager is None:
            raise RuntimeError(
                "OntologyManager has no triple store registered; "
                "call register_triple_store() before reading the catalog"
            )
        return self._triple_store_manager

    async def aget_catalog_headers(self) -> list[OntologyHeader]:
        """Read ontology header metadata for every stored version.

        Deliberately **not** cached. Headers are what terminal-version selection
        runs on, so caching them would let this process miss another worker's
        writes to a shared store -- the one thing the graph cache cannot go wrong
        about, and the one thing this would.

        Returns:
            list[OntologyHeader]: One header per stored ontology version.
        """
        return await self._require_triple_store().afetch_ontology_catalog()

    async def aget_ontologies_by_iri(self, iris: Sequence[str]) -> list[Ontology]:
        """Return terminal ontologies for ``iris``, fetching only cache misses.

        Terminal selection always runs against freshly read headers; only the
        graph bytes come from cache, keyed by the content-addressed
        ``versioned_iri``.

        Args:
            iris: Ontology IRIs to resolve. Empty means "no restriction", matching
                :meth:`~ontocast.tool.triple_manager.core.TripleStoreManager.afetch_ontologies_by_iri`.

        Returns:
            list[Ontology]: Terminal ontologies with graphs. Callers must treat
            these as shared read-only references.
        """
        store = self._require_triple_store()
        headers = dedupe_terminal_ontologies(await self.aget_catalog_headers())
        if iris:
            wanted = set(iris)
            headers = [header for header in headers if header.iri in wanted]

        resolved: list[Ontology] = []
        missing_iris: list[str] = []
        graph_uri_by_iri: dict[str, str] = {}
        for header in headers:
            cached = self._graph_cache.get(header.graph_uri)
            if cached is not None:
                self._graph_cache_hits += 1
                self._graph_cache.move_to_end(header.graph_uri)
                resolved.append(cached)
            else:
                self._graph_cache_misses += 1
                missing_iris.append(header.iri)
                graph_uri_by_iri[header.iri] = header.graph_uri

        if missing_iris:
            fetched = await store.afetch_ontologies_by_iri(missing_iris)
            for ontology in fetched:
                self._cache_graph(ontology, graph_uri_by_iri.get(ontology.iri))
            resolved.extend(fetched)
        return resolved

    async def aget_merged_graph(
        self, ontologies: Sequence[Ontology]
    ) -> tuple[RDFGraph, dict[str, str]]:
        """Return the prefix-bound union of ``ontologies``, cached by version set.

        The induced-subgraph builder reads this union without mutating it, so one
        merge can be shared by every content unit that selects the same ontology
        versions -- which is the common case inside a document.

        Args:
            ontologies: Ontology versions to merge.

        Returns:
            tuple: ``(merged_graph, prefix_map)``. The graph **must not be mutated
            by callers**; it is shared.
        """
        from .sparql import merge_ontology_graphs

        key = frozenset(onto.versioned_iri for onto in ontologies)
        cached = self._merged_cache.get(key)
        if cached is not None:
            self._merged_cache_hits += 1
            self._merged_cache.move_to_end(key)
            return cached

        self._merged_cache_misses += 1
        merged = await asyncio.to_thread(merge_ontology_graphs, list(ontologies))
        self._merged_cache[key] = merged
        while len(self._merged_cache) > _MERGED_CACHE_MAX_ENTRIES:
            self._merged_cache.popitem(last=False)
        return merged

    def catalog_cache_stats(self) -> dict[str, int]:
        """Cache hit/miss counters, for tests and retrieval diagnostics."""
        return {
            "catalog_graph_cache_hits": self._graph_cache_hits,
            "catalog_graph_cache_misses": self._graph_cache_misses,
            "catalog_merge_cache_hits": self._merged_cache_hits,
            "catalog_merge_cache_misses": self._merged_cache_misses,
        }

    def _cache_graph(self, ontology: Ontology, graph_uri: str | None = None) -> None:
        """Register a store-read ``ontology`` under the graph URI it was read from.

        Only ever called with graphs that came *from* the triple store. Seeding the
        cache from :meth:`add_ontology` instead would be tempting -- those graphs are
        already in memory -- but a registered ontology and its persisted form are not
        byte-identical: writing round-trips through deterministic Turtle, which
        relabels blank nodes. Snapshot expansion tie-breaks on ``str(triple)``, so
        mixing the two makes retrieval depend on whether a graph happened to be
        written by this process.

        The key must be the *header's* ``graph_uri``, since that is what
        :meth:`aget_ontologies_by_iri` looks up. Keying on the recomputed
        ``versioned_iri`` instead is only equivalent while content hashing is
        round-trip stable; when it is not, the two never coincide and every
        lookup misses forever.

        Args:
            ontology: Ontology materialized from the triple store.
            graph_uri: Named graph it was read from. Falls back to the
                content-addressed ``versioned_iri`` when the caller has no header.
        """
        key = graph_uri or (ontology.versioned_iri if ontology.hash else None)
        if not key:
            return
        self._graph_cache.setdefault(key, ontology)
        self._graph_cache.move_to_end(key)
        self._graph_uris_by_iri.setdefault(ontology.iri, set()).add(key)
        while len(self._graph_cache) > _GRAPH_CACHE_MAX_ENTRIES:
            evicted_key, evicted = self._graph_cache.popitem(last=False)
            uris = self._graph_uris_by_iri.get(evicted.iri)
            if uris is not None:
                uris.discard(evicted_key)
                if not uris:
                    self._graph_uris_by_iri.pop(evicted.iri, None)

    def _effective_patch_top_k(self, top_k: int | None) -> int:
        if top_k is not None:
            return top_k
        if self._patch_retriever is not None:
            return self._patch_retriever.vector_store.store_config.top_k
        return 10

    def _fallback_patch_results(
        self, queries: list[str]
    ) -> list[tuple[RDFGraph | None, list[str]]]:
        """Per-query independent copies of the freshest terminal ontology graph."""
        fallback = self.get_freshest_terminal_ontology_by_iri(None)
        if fallback is None:
            return [(None, []) for _ in queries]
        sources = [fallback.iri]
        return [(fallback.graph.copy(), sources) for _ in queries]

    @staticmethod
    def _normalize_patch_graph(
        graph: RDFGraph, sources: list[str]
    ) -> tuple[RDFGraph, list[str]]:
        return (graph, sources) if len(graph) > 0 else (RDFGraph(), sources)

    def get_patch_context(
        self,
        query: str,
        top_k: int | None = None,
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
        estimated_triples_per_query: int | None = None,
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
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
        estimated_triples_per_query: int | None = None,
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
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
        estimated_triples_per_query: int | None = None,
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
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
        estimated_triples_per_query: int | None = None,
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
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
        estimated_triples_per_query: int | None = None,
    ) -> list[tuple[RDFGraph | None, list[str]]]:
        """Retrieve patch contexts for many queries in a batched pass.

        With a patch retriever, the list has length 1 (ensemble graph + sources).
        Without it, length matches ``queries`` (fallback ontology per query).

        Raises:
            RuntimeError: If called while an event loop is running. Use
                :meth:`aget_patch_contexts_with_sources` from async code.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aget_patch_contexts_with_sources(
                    queries=queries,
                    top_k=top_k,
                    subgraph_depth=subgraph_depth,
                    max_total_triples=max_total_triples,
                    estimated_triples_per_query=estimated_triples_per_query,
                )
            )
        raise RuntimeError(
            "get_patch_contexts_with_sources() cannot be called from async code; "
            "use await aget_patch_contexts_with_sources()"
        )

    async def aget_patch_contexts_with_sources(
        self,
        queries: list[str],
        top_k: int | None = None,
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
        estimated_triples_per_query: int | None = None,
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
            return [self._normalize_patch_graph(graph, sources)]

        return self._fallback_patch_results(queries)

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
