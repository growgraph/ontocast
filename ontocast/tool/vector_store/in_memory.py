"""Process-local vector store for ontology atoms.

This is the vector-store counterpart to
:class:`~ontocast.tool.triple_manager.in_memory.InMemoryTripleStoreManager`: it
completes the zero-external-services path, so a bare ``pip install ontocast``
can run every ontology-context mode, including
:attr:`~ontocast.onto.enum.OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY`.

Two properties make an exact implementation the right choice at this scale.
Ontology atom counts are in the thousands, where a full numpy matrix product
beats any approximate index plus its build cost. And BM25 over the same corpus
is a few dozen lines, which removes the ``fastembed`` dependency (and with it an
ONNX runtime) that the Qdrant and LanceDB lanes require.

State lives in the instance, so it is lost when the process exits and is not
shared across workers. That is the intended trade: use Qdrant or LanceDB when
the index must outlive the process.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from typing import Any

import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from ontocast.config import EmbeddingConfig, VectorStoreConfig
from ontocast.onto.ontology import Ontology
from ontocast.onto.tenancy import TENANCY_SEP, TenancyScope
from ontocast.tool.vector_store.atomizer import GraphAtomizer
from ontocast.tool.vector_store.core import (
    GraphAtom,
    OntologySearchHit,
    OntologySearchHitsByChannel,
    VectorStoreManager,
)
from ontocast.tool.vector_store.embedding import (
    EmbeddingTool,
    FastembedBm25SparseTool,
)
from ontocast.tool.vector_store.lexical_trigger import LexicalTriggerIndex
from ontocast.tool.vector_store.util import (
    dedupe_hits_by_identity,
    effective_top_k,
    iter_batches,
    normalized_fusion_weights,
    point_id_for_atom,
    rank_fuse_channel_hits,
    require_embedding_vector_length,
    sync_atomizer_from_store_config,
)

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Okapi BM25 term-frequency saturation and length-normalisation constants. The
# standard defaults; the corpus here is short label text where neither is
# sensitive enough to warrant exposing as configuration.
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Split text into case-folded alphanumeric tokens for BM25."""
    return _TOKEN_RE.findall(text.lower())


class _Record:
    """One indexed atom with its dense vectors and BM25 term counts."""

    __slots__ = ("atom", "core", "neighborhood", "terms", "length")

    def __init__(
        self,
        atom: GraphAtom,
        core: list[float],
        neighborhood: list[float],
    ) -> None:
        self.atom = atom
        self.core = core
        self.neighborhood = neighborhood
        self.terms = Counter(_tokenize(atom.minimal_representation))
        self.length = sum(self.terms.values())


class InMemoryVectorStoreManager(VectorStoreManager):
    """Keeps ontology atoms in process memory with exact dense and BM25 search."""

    store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingTool = Field(..., exclude=True)
    sparse_embedding: FastembedBm25SparseTool | None = Field(default=None, exclude=True)
    atomizer: GraphAtomizer = Field(default_factory=GraphAtomizer, exclude=True)

    _records: dict[str, _Record] = PrivateAttr(default_factory=dict)
    _core_matrix: Any = PrivateAttr(default=None)
    _neighborhood_matrix: Any = PrivateAttr(default=None)
    _matrix_keys: list[str] = PrivateAttr(default_factory=list)
    _lexical_trigger_index: LexicalTriggerIndex | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _sync_atomizer_with_store_config(self) -> "InMemoryVectorStoreManager":
        """Mirror representation settings from store config onto the atomizer."""
        sync_atomizer_from_store_config(self.atomizer, self.store_config)
        return self

    @property
    def embedding_config(self) -> EmbeddingConfig:
        """Embedding configuration of the wired dense provider."""
        return self.embedding.config

    def _dense_dimension(self) -> int:
        return self.embedding.config.dimension

    # -- lifecycle ---------------------------------------------------------

    async def initialize(self) -> None:
        """No-op: the store is ready as soon as it is constructed."""
        return None

    async def wipe_store(self) -> None:
        """Drop every indexed atom."""
        self._records.clear()
        self._invalidate_matrices()
        self._lexical_trigger_index = None

    def close(self) -> None:
        """Release indexed state; there is no external connection to close."""
        self._records.clear()
        self._invalidate_matrices()

    # -- tenancy -----------------------------------------------------------

    def supports_tenancy_partition(self) -> bool:
        """True: each partition gets its own index.

        With a ToolBox per scope that is automatic -- the stores are separate
        objects. When a single store is retargeted instead, :meth:`apply_tenancy`
        drops the index rather than carrying it across.
        """
        return True

    def apply_tenancy(
        self, tenant: str, project: str, *, sep: str = TENANCY_SEP
    ) -> None:
        """Bind this store to ``tenant`` / ``project``, dropping any prior index.

        The index is partition-scoped, so carrying it across a switch would leak
        one tenant's ontology terms into another's retrieval. Nothing is
        migrated: re-index after switching.
        """
        scope = TenancyScope.build(tenant, project, sep=sep)
        changed = self.store_config.ontology_table != scope.ontologies_name
        self.store_config.ontology_table = scope.ontologies_name
        self.store_config.facts_table = scope.facts_name
        if changed and self._records:
            logger.info(
                "Dropping %d in-memory vector(s) on tenancy switch to %s/%s",
                len(self._records),
                scope.tenant,
                scope.project,
            )
            self._records.clear()
            self._invalidate_matrices()
            self._lexical_trigger_index = None

    async def clean_tenancy(self, tenant: str, project: str) -> None:
        """Drop the index when it belongs to the named partition."""
        scope = TenancyScope.build(tenant, project)
        if self.store_config.ontology_table == scope.ontologies_name:
            await self.wipe_store()

    # -- matrix cache ------------------------------------------------------

    def _invalidate_matrices(self) -> None:
        self._core_matrix = None
        self._neighborhood_matrix = None
        self._matrix_keys = []

    def _ensure_matrices(self) -> None:
        """Rebuild the L2-normalised vector matrices if the index changed.

        Normalising once at build time turns every later cosine similarity into
        a plain dot product, which is what makes the exact search cheap.
        """
        if self._core_matrix is not None and len(self._matrix_keys) == len(
            self._records
        ):
            return
        keys = sorted(self._records)
        if not keys:
            self._matrix_keys = []
            self._core_matrix = np.zeros((0, 0), dtype=np.float32)
            self._neighborhood_matrix = np.zeros((0, 0), dtype=np.float32)
            return
        core = np.asarray([self._records[k].core for k in keys], dtype=np.float32)
        neighborhood = np.asarray(
            [self._records[k].neighborhood for k in keys], dtype=np.float32
        )
        self._matrix_keys = keys
        self._core_matrix = _l2_normalize(core)
        self._neighborhood_matrix = _l2_normalize(neighborhood)

    # -- indexing ----------------------------------------------------------

    def index_ontology(self, ontology: Ontology) -> int:
        """Atomize and index an ontology, returning the number of atoms stored."""
        atoms = self.atomizer.atomize(source=ontology, depth=1)
        if not atoms:
            return 0

        n = len(atoms)
        dense_texts = [atom.core_representation for atom in atoms] + [
            atom.neighborhood_representation for atom in atoms
        ]
        dense_vectors = self._embed_texts_batched(dense_texts)
        if len(dense_vectors) != 2 * n:
            raise ValueError(
                "Embedding provider returned mismatched vector counts for atoms"
            )

        for i, atom in enumerate(atoms):
            key = point_id_for_atom(atom, store_config=self.store_config)
            self._records[key] = _Record(atom, dense_vectors[i], dense_vectors[n + i])

        self._invalidate_matrices()
        self._register_lexical_triggers(atoms)
        return n

    def delete_ontology(
        self,
        iri: str,
        version: str | None = None,
        ontology_hash: str | None = None,
    ) -> None:
        """Delete indexed atoms matching the given ontology identity."""
        doomed = [
            key
            for key, record in self._records.items()
            if _matches(record.atom, iri, version, ontology_hash)
        ]
        for key in doomed:
            del self._records[key]
        if doomed:
            self._invalidate_matrices()
        self._get_lexical_trigger_index().unregister_ontology(iri)

    def list_indexed_ontology_iris(self) -> set[str]:
        """Return distinct ``ontology_iri`` values currently indexed."""
        return {record.atom.ontology_iri for record in self._records.values()}

    def _embed_texts_batched(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for batch in iter_batches(texts, self.store_config.embedding_batch_size):
            batch_vectors = self.embedding.embed(batch)
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    "Embedding provider returned mismatched vectors for batch"
                )
            for j, vec in enumerate(batch_vectors):
                require_embedding_vector_length(
                    vec,
                    role=f"Index embedding batch offset {len(vectors) + j}",
                    expected=self._dense_dimension(),
                )
            vectors.extend(batch_vectors)
        return vectors

    # -- search ------------------------------------------------------------

    def _candidate_keys(
        self,
        *,
        filter_iri: str | None,
        filter_version: str | None,
        filter_hash: str | None,
    ) -> set[str] | None:
        """Return the keys passing the identity filters, or None for "all"."""
        if filter_iri is None and filter_version is None and filter_hash is None:
            return None
        return {
            key
            for key, record in self._records.items()
            if _matches(record.atom, filter_iri, filter_version, filter_hash)
        }

    def _search_dense_channel(
        self,
        vector: list[float],
        *,
        matrix: Any,
        limit: int,
        allowed: set[str] | None,
        apply_neighborhood_empty_penalty: bool = False,
    ) -> list[OntologySearchHit]:
        if not self._matrix_keys or matrix is None or matrix.size == 0:
            return []
        query = _l2_normalize(np.asarray([vector], dtype=np.float32))[0]
        scores = matrix @ query

        order = np.argsort(-scores)
        hits: list[OntologySearchHit] = []
        for idx in order:
            key = self._matrix_keys[int(idx)]
            if allowed is not None and key not in allowed:
                continue
            score = float(scores[int(idx)])
            atom = self._records[key].atom
            if apply_neighborhood_empty_penalty and (
                atom.neighborhood_representation.strip().lower()
                == "no neighborhood facts available"
            ):
                score = 0.0
            hits.append(
                OntologySearchHit(
                    atom=atom.model_copy(update={"score": score}), score=score
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _search_bm25_channel(
        self,
        query: str,
        *,
        limit: int,
        allowed: set[str] | None,
    ) -> list[OntologySearchHit]:
        """Score the corpus with Okapi BM25 over ``minimal_representation``.

        Implemented locally rather than through ``fastembed`` so that the
        in-memory backend needs no ONNX runtime. Term statistics are recomputed
        per query; at ontology scale that is cheaper than maintaining an
        inverted index across edits.
        """
        terms = _tokenize(query)
        if not terms or not self._records:
            return []

        keys = [k for k in self._records if allowed is None or k in allowed]
        if not keys:
            return []
        records = [self._records[k] for k in keys]

        total = len(records)
        avg_len = sum(r.length for r in records) / total if total else 0.0
        if avg_len <= 0.0:
            return []

        doc_freq = Counter()
        for record in records:
            for term in set(terms):
                if term in record.terms:
                    doc_freq[term] += 1

        scored: list[tuple[float, GraphAtom]] = []
        for record in records:
            score = 0.0
            for term in terms:
                freq = record.terms.get(term, 0)
                if not freq:
                    continue
                n_q = doc_freq[term]
                idf = math.log(1.0 + (total - n_q + 0.5) / (n_q + 0.5))
                denom = freq + _BM25_K1 * (
                    1.0 - _BM25_B + _BM25_B * record.length / avg_len
                )
                score += idf * (freq * (_BM25_K1 + 1.0)) / denom
            if score > 0.0:
                scored.append((score, record.atom))

        scored.sort(key=lambda pair: -pair[0])
        return [
            OntologySearchHit(
                atom=atom.model_copy(update={"score": score}), score=score
            )
            for score, atom in scored[:limit]
        ]

    def search_hits_by_vector(
        self,
        core_vector: list[float],
        neighborhood_vector: list[float],
        bm25_query: str | None = None,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> OntologySearchHitsByChannel:
        """Search all channels for one pre-encoded query."""
        eff_top_k = effective_top_k(self.store_config, top_k)
        self._ensure_matrices()
        allowed = self._candidate_keys(
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )
        core_hits = self._search_dense_channel(
            core_vector,
            matrix=self._core_matrix,
            limit=eff_top_k,
            allowed=allowed,
        )
        neighborhood_hits = self._search_dense_channel(
            neighborhood_vector,
            matrix=self._neighborhood_matrix,
            limit=eff_top_k,
            allowed=allowed,
            apply_neighborhood_empty_penalty=True,
        )
        bm25_hits: list[OntologySearchHit] = []
        if bm25_query is not None:
            bm25_hits = self._search_bm25_channel(
                bm25_query, limit=eff_top_k, allowed=allowed
            )
        if self.store_config.dedup_query_hits_by_iri:
            core_hits = dedupe_hits_by_identity(
                core_hits, store_config=self.store_config
            )
            neighborhood_hits = dedupe_hits_by_identity(
                neighborhood_hits, store_config=self.store_config
            )
            bm25_hits = dedupe_hits_by_identity(
                bm25_hits, store_config=self.store_config
            )
        return OntologySearchHitsByChannel(
            core_hits=core_hits,
            neighborhood_hits=neighborhood_hits,
            bm25_hits=bm25_hits,
        )

    def _encode_query_vectors_batch(
        self, queries: list[str]
    ) -> list[tuple[list[float], list[float], str]]:
        n = len(queries)
        if n == 0:
            return []
        dense_vecs = self.embedding.embed_query(queries)
        if len(dense_vecs) != n:
            raise ValueError(
                "Embedding provider returned mismatched vectors for queries"
            )
        for i, vec in enumerate(dense_vecs):
            require_embedding_vector_length(
                vec,
                role=f"Query embedding[{i}]",
                expected=self._dense_dimension(),
            )
        return [(dense_vecs[i], dense_vecs[i], queries[i]) for i in range(n)]

    def search_patches(
        self,
        query: str,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[GraphAtom]:
        """Search ontology atoms and return the fused atoms."""
        return [
            hit.atom
            for hit in self.search_patch_hits(
                query=query,
                top_k=top_k,
                filter_iri=filter_iri,
                filter_version=filter_version,
                filter_hash=filter_hash,
            )
        ]

    def search_patch_hits(
        self,
        query: str,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHit]:
        """Search ontology atoms and return rank-fused scored hits."""
        results = self.search_patch_hits_many(
            [query],
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )
        if not results:
            return []
        eff_top_k = effective_top_k(self.store_config, top_k)
        cw, nw, bw = normalized_fusion_weights(self.store_config)
        channel_hits = results[0]
        return rank_fuse_channel_hits(
            channel_hits.core_hits,
            channel_hits.neighborhood_hits,
            channel_hits.bm25_hits,
            core_weight=cw,
            neighborhood_weight=nw,
            bm25_weight=bw,
            limit=eff_top_k,
        )

    def search_patch_hits_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHitsByChannel]:
        """Search many queries, returning split-channel hits for each."""
        if not queries:
            return []
        eff_top_k = effective_top_k(self.store_config, top_k)
        triples = self._encode_query_vectors_batch(queries)
        return [
            self.search_hits_by_vector(
                core_vector=core_v,
                neighborhood_vector=neigh_v,
                bm25_query=bm25_q,
                top_k=eff_top_k,
                filter_iri=filter_iri,
                filter_version=filter_version,
                filter_hash=filter_hash,
            )
            for core_v, neigh_v, bm25_q in triples
        ]

    async def asearch_patch_hits_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHitsByChannel]:
        """Async variant; the search itself is CPU-bound, so it runs in a thread."""
        if not queries:
            return []
        return await asyncio.to_thread(
            self.search_patch_hits_many,
            queries,
            top_k,
            filter_iri,
            filter_version,
            filter_hash,
        )

    # -- vector and atom fetch --------------------------------------------

    def fetch_vectors(
        self,
        atom_ids: list[str],
    ) -> dict[str, tuple[list[float], list[float]]]:
        """Batch-fetch dense core/neighborhood vectors by ``atom_id``."""
        if not atom_ids:
            return {}
        wanted = set(atom_ids)
        out: dict[str, tuple[list[float], list[float]]] = {}
        for record in self._records.values():
            if record.atom.atom_id in wanted:
                out[record.atom.atom_id] = (record.core, record.neighborhood)
        return out

    def fetch_atoms_by_ids(self, atom_ids: list[str]) -> list[GraphAtom]:
        """Batch-fetch atom payloads by ``atom_id``, preserving input order."""
        if not atom_ids:
            return []
        by_id = {record.atom.atom_id: record.atom for record in self._records.values()}
        return [by_id[aid] for aid in atom_ids if aid in by_id]

    # -- lexical triggers --------------------------------------------------

    def _get_lexical_trigger_index(self) -> LexicalTriggerIndex:
        if self._lexical_trigger_index is None:
            self._lexical_trigger_index = LexicalTriggerIndex(
                max_match_atoms=self.store_config.lexical_trigger_max_atoms
            )
        return self._lexical_trigger_index

    def _register_lexical_triggers(self, atoms: list[GraphAtom]) -> None:
        if not self.store_config.lexical_trigger_enabled:
            return
        self._get_lexical_trigger_index().register_atoms(atoms)

    def match_lexical_triggers(
        self, text: str, *, max_atoms: int | None = None
    ) -> list[GraphAtom]:
        """Match raw text against the lexical-trigger index and return atoms."""
        if not self.store_config.lexical_trigger_enabled or not text.strip():
            return []
        limit = (
            self.store_config.lexical_trigger_max_atoms
            if max_atoms is None
            else max_atoms
        )
        if limit <= 0:
            return []
        matched_ids = self._get_lexical_trigger_index().match(text, max_atoms=limit)
        atoms = self.fetch_atoms_by_ids(matched_ids)
        trigger_score = self.store_config.lexical_trigger_score
        return [atom.model_copy(update={"score": trigger_score}) for atom in atoms]


def _l2_normalize(matrix: Any) -> Any:
    """Return ``matrix`` with each row scaled to unit length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _matches(
    atom: GraphAtom,
    iri: str | None,
    version: str | None,
    ontology_hash: str | None,
) -> bool:
    """Return whether an atom satisfies the given identity filters."""
    if iri is not None and atom.ontology_iri != iri:
        return False
    if version is not None and atom.ontology_version != version:
        return False
    if ontology_hash is not None and atom.ontology_hash != ontology_hash:
        return False
    return True
