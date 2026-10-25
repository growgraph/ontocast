"""Ontology management tool for OntoCast.

This module provides functionality for managing multiple ontologies, including
loading, updating, and retrieving ontologies by name or IRI. Tracks version
lineage using hash-based identifiers.
"""

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ontocast.onto.ontology_header import DivergedOntologyError, heads_by_iri

from ..onto.null import NULL_ONTOLOGY
from ..onto.ontology import Ontology
from ..onto.ontology_header import OntologyHeader
from ..onto.rdfgraph import RDFGraph
from ..onto.util import normalize_ontology_iri
from .facts_validation.terms import collect_catalog_terms
from .onto import Tool
from .triple_manager.core import TripleStoreManager
from .triple_manager.util import head_headers

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
    import networkx as nx

    from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever


class OntologyManager(Tool):
    """The catalog: every ontology this process holds, with its versions.

    For each IRI it holds the stamped versions it has seen, identified by
    their version IRIs and linked by ``prior_version_iris``. The *head* of an
    ontology is the version nothing names as prior; an ontology with several
    heads is diverged and cannot be extended until merged.

    Attributes:
        ontology_versions: Stamped versions per ontology IRI.
    """

    ontology_versions: dict[str, list[Ontology]] = Field(default_factory=dict)

    def __init__(self, **kwargs: Any):
        """Initialize the ontology manager.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        # IRI -> version IRI of its head. Updated when a version is added.
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
        # _graph_cache is keyed by version IRI (the named graph a version was
        # read from); the version IRIs cached per ontology are tracked for
        # eviction.
        self._graph_cache: OrderedDict[str, Ontology] = OrderedDict()
        self._graph_uris_by_iri: dict[str, set[str]] = {}
        self._merged_cache: OrderedDict[
            frozenset[str], tuple[RDFGraph, dict[str, str]]
        ] = OrderedDict()
        self._graph_cache_hits = 0
        self._graph_cache_misses = 0
        self._merged_cache_hits = 0
        self._merged_cache_misses = 0
        # Union of every served ontology's IRIs, keyed by the served version
        # set so it is rebuilt exactly when the catalog changes. Read per unit
        # by the deterministic repairs; see :meth:`catalog_terms`.
        self._catalog_terms_memo: tuple[frozenset[str], set[str]] | None = None

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

    def __contains__(self, item: object) -> bool:
        """Check if an item (IRI or alias) is in the ontology manager.

        Args:
            item: The IRI, ontology_id, or author prefix to check.

        Returns:
            bool: True if the item resolves to a tracked ontology IRI.
        """
        return self.resolve_ontology_ref(str(item)) is not None

    def _prepare_ontology_for_catalog(self, ontology: Ontology) -> bool:
        """Validate and register a stamped version; True if it is new here.

        Raises:
            ValueError: the ontology is null, unstamped, or its identity
                collides with a registered one.
        """
        if not ontology.iri or ontology.iri == NULL_ONTOLOGY.iri:
            raise ValueError("cannot register the null ontology")
        if not ontology.version_iri:
            raise ValueError(f"ontology {ontology.iri} is not stamped as a version")

        self.validate_identity_uniqueness(ontology)
        self._register_identity(ontology)

        # A registered version is a stored one: what derives from it next
        # derives from this version.
        if not (
            len(ontology.bases) == 1
            and ontology.bases[0].version_iri == ontology.version_iri
        ):
            ontology.bases = [OntologyHeader.from_ontology(ontology)]

        versions = self.ontology_versions.setdefault(ontology.iri, [])
        if any(o.version_iri == ontology.version_iri for o in versions):
            logger.debug("Version %s already registered", ontology.version_iri)
            return False
        versions.append(ontology)
        self._cached_ontologies.pop(ontology.iri, None)
        logger.debug("Registered version %s", ontology.version_iri)
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

    def replace_version(self, ontology: Ontology) -> None:
        """Replace a tracked version in place (e.g. after writing a tombstone).

        Raises:
            ValueError: ``ontology`` is unstamped, or no version with its
                ``version_iri`` is held.
        """
        if not ontology.version_iri:
            raise ValueError(f"ontology {ontology.iri} is not stamped as a version")
        versions = self.ontology_versions.get(ontology.iri, [])
        for index, held in enumerate(versions):
            if held.version_iri == ontology.version_iri:
                versions[index] = ontology
                self._cached_ontologies.pop(ontology.iri, None)
                self._graph_cache[ontology.version_iri] = ontology
                self._graph_cache.move_to_end(ontology.version_iri)
                return
        raise ValueError(f"no tracked version {ontology.version_iri}")

    def remove_ontology_by_iri(self, iri: str) -> None:
        """Drop all tracked versions for an ontology IRI and clear caches."""
        for graph_uri in self._graph_uris_by_iri.pop(iri, set()):
            self._graph_cache.pop(graph_uri, None)
        for ontology in self.ontology_versions.get(iri, []):
            if ontology.version_iri:
                self._graph_cache.pop(ontology.version_iri, None)
        stale_merges = [
            key
            for key in self._merged_cache
            if any(versioned.startswith(f"{iri}#") for versioned in key)
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
        """Return the heads of ``iris``, fetching only cache misses.

        Head selection always runs against freshly read headers; only the
        graph bytes come from cache, keyed by version IRI.

        Args:
            iris: Ontology IRIs to resolve. Empty means "no restriction", matching
                :meth:`~ontocast.tool.triple_manager.core.TripleStoreManager.afetch_ontologies_by_iri`.

        Returns:
            list[Ontology]: Terminal ontologies with graphs. Callers must treat
            these as shared read-only references.
        """
        store = self._require_triple_store()
        headers = head_headers(await self.aget_catalog_headers())
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

        key = frozenset(onto.version_iri or onto.iri for onto in ontologies)
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

    def catalog_terms(self) -> set[str]:
        """Every IRI the held ontologies declare or reference, as one set.

        The union of :func:`collect_catalog_terms` over every ontology this
        manager holds -- the terminal versions registered in memory and the
        graphs read through the triple store and resident in the graph cache.
        Built lazily and memoised on the content-addressed identifiers of that
        set, so it is recomputed only when an ontology is added, superseded,
        removed, or newly read from the store.

        The per-unit repairs read it for membership: a term present here is a
        real catalog term even when a unit's retrieved snapshot omits it, and
        must never be rewritten toward a look-alike the snapshot does carry.

        Returns:
            set[str]: Shared across units -- treat as read-only.
        """
        held: dict[str, Ontology] = {
            ontology.version_iri or ontology.iri: ontology
            for ontology in self.get_heads_by_iri(None)
        }
        for graph_uri, ontology in self._graph_cache.items():
            held.setdefault(graph_uri, ontology)
        key = frozenset(held)
        memo = self._catalog_terms_memo
        if memo is not None and memo[0] == key:
            return memo[1]
        terms: set[str] = set()
        for ontology in held.values():
            terms |= collect_catalog_terms(ontology.graph)
        self._catalog_terms_memo = (key, terms)
        return terms

    def _cache_graph(self, ontology: Ontology, graph_uri: str | None = None) -> None:
        """Register a store-read ``ontology`` under the graph URI it was read from.

        Only ever called with graphs that came *from* the triple store. Seeding the
        cache from :meth:`add_ontology` instead would be tempting -- those graphs are
        already in memory -- but a registered ontology and its persisted form are not
        byte-identical: writing round-trips through deterministic Turtle, which
        relabels blank nodes. Snapshot expansion tie-breaks on ``str(triple)``, so
        mixing the two makes retrieval depend on whether a graph happened to be
        written by this process.

        Args:
            ontology: Ontology materialized from the triple store.
            graph_uri: Named graph it was read from; defaults to its version IRI.
        """
        key = graph_uri or ontology.version_iri
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
        """Per-query independent copies of the catalog's one head graph."""
        fallback = self.get_head_by_iri(None)
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

    def get_heads_by_iri(self, iri: str | None = None) -> list[Ontology]:
        """The heads held for ``iri``, or for every ontology when ``iri`` is None.

        A head is a live version that no live version descends from; descent
        walks through tombstoned versions.
        """
        if iri:
            versions = list(self.ontology_versions.get(iri, []))
        else:
            versions = [o for group in self.ontology_versions.values() for o in group]
        if not versions:
            return []
        by_version_iri = {o.version_iri: o for o in versions if o.version_iri}
        heads = heads_by_iri([OntologyHeader.from_ontology(o) for o in versions])
        return [
            by_version_iri[header.version_iri]
            for group in heads.values()
            for header in group
            if header.version_iri in by_version_iri
        ]

    def get_heads(self, ontology_id: str | None = None) -> list[Ontology]:
        """The heads of the ontology ``ontology_id`` names (id, alias or IRI)."""
        if ontology_id:
            iri = self.resolve_ontology_ref(ontology_id)
            if iri is None:
                return []
            return self.get_heads_by_iri(iri)
        return self.get_heads_by_iri(None)

    def get_head_by_iri(self, iri: str | None = None) -> Ontology | None:
        """The one head of ``iri``, or None when nothing is held for it.

        With ``iri`` None, the head of the single held ontology (None when
        several are held: there is no "the" head across ontologies).

        Raises:
            DivergedOntologyError: the ontology has several heads.
        """
        heads = self.get_heads_by_iri(iri)
        if not heads:
            return None
        if iri is None:
            iris = {o.iri for o in heads}
            if len(iris) != 1:
                return None
            iri = heads[0].iri
            heads = [o for o in heads if o.iri == iri]
        if len(heads) > 1:
            raise DivergedOntologyError(iri, [o.version_iri or "" for o in heads])
        return heads[0]

    def get_head(self, ontology_id: str | None = None) -> Ontology | None:
        """The one head of the ontology ``ontology_id`` names (id, alias or IRI).

        Raises:
            DivergedOntologyError: the ontology has several heads.
        """
        if ontology_id:
            iri = self.resolve_ontology_ref(ontology_id)
            if iri is None:
                return None
            return self.get_head_by_iri(iri)
        return self.get_head_by_iri(None)

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

    def get_lineage_graph_by_iri(self, iri: str) -> "nx.DiGraph | None":
        """The version DAG held for ``iri`` (nodes are version IRIs), or None."""
        if iri not in self.ontology_versions:
            return None
        return Ontology.build_lineage_graph(self.ontology_versions[iri])

    def get_lineage_graph(self, ontology_id: str) -> "nx.DiGraph | None":
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
        version_iri: str | None = None,
    ) -> Ontology:
        """Get an ontology by IRI, id/alias, or a specific version IRI.

        Without ``version_iri`` the head is returned (the newest head of a
        diverged ontology). IRI is preferred over ``ontology_id``.

        Returns:
            Ontology: The matching ontology, NULL_ONTOLOGY when none.
        """
        if version_iri:
            for versions in self.ontology_versions.values():
                for o in versions:
                    if o.version_iri == version_iri:
                        return o

        resolved_iri: str | None = None
        if ontology_iri is not None:
            resolved_iri = self.resolve_ontology_ref(ontology_iri)
        if resolved_iri is None and ontology_id is not None:
            resolved_iri = self.resolve_ontology_ref(ontology_id)

        if resolved_iri is not None and resolved_iri in self.ontology_versions:
            versions = self.ontology_versions[resolved_iri]
            if not version_iri:
                heads = self.get_heads_by_iri(resolved_iri)
                if heads:
                    return heads[0]
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
        """The head of each held ontology, one per IRI.

        A diverged ontology (several heads) is left out: nothing extracts
        against it until it is merged.

        Returns:
            list[Ontology]: One head per IRI, in IRI order.
        """
        result: list[Ontology] = []
        for iri in sorted(self.ontology_versions):
            cached = self._cached_ontologies.get(iri)
            if cached is not None:
                head = next(
                    (o for o in self.ontology_versions[iri] if o.version_iri == cached),
                    None,
                )
                if head is not None:
                    result.append(head)
                    continue
            try:
                head = self.get_head_by_iri(iri)
            except DivergedOntologyError as exc:
                logger.error("%s", exc)
                continue
            if head is not None and head.version_iri:
                self._cached_ontologies[iri] = head.version_iri
                result.append(head)
        for removed_iri in set(self._cached_ontologies) - set(self.ontology_versions):
            del self._cached_ontologies[removed_iri]
        return result
