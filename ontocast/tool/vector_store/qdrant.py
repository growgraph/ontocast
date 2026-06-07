"""Qdrant-backed vector store for ontology atoms."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeAlias, cast

from pydantic import Field, PrivateAttr
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from ontocast.config import EmbeddingConfig, QdrantConfig, VectorStoreConfig
from ontocast.onto.ontology import Ontology
from ontocast.onto.tenancy import (
    TENANCY_SEP,
    tenant_project_facts_name,
    tenant_project_ontologies_name,
)
from ontocast.tool.vector_store.atomizer import GraphAtomizer
from ontocast.tool.vector_store.core import (
    BM25_VECTOR_NAME,
    CORE_VECTOR_NAME,
    NEIGHBORHOOD_VECTOR_NAME,
    GraphAtom,
    OntologySearchHit,
    OntologySearchHitsByChannel,
    VectorStoreManager,
)
from ontocast.tool.vector_store.embedding import (
    EmbeddingTool,
    FastembedBm25SparseTool,
)
from ontocast.tool.vector_store.util import (
    META_EMBEDDING_MODEL,
    EmbeddingContractMismatchError,
    atom_from_payload,
    atom_payload,
    collection_embedding_metadata,
    dedupe_hits_by_identity,
    effective_top_k,
    embedding_contract_help,
    identity_key_for_atom,
    iter_batches,
    normalized_fusion_weights,
    point_id,
    point_id_for_atom,
    rank_fuse_channel_hits,
    require_embedding_vector_length,
    validate_embedding_contract_metadata,
)

logger = logging.getLogger(__name__)

ChannelVector: TypeAlias = list[float] | qdrant_models.SparseVector


class QdrantVectorStoreManager(VectorStoreManager):
    """Stores ontology atoms in Qdrant and supports similarity lookup."""

    store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    qdrant_config: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingTool = Field(..., exclude=True)
    sparse_embedding: FastembedBm25SparseTool | None = Field(default=None, exclude=True)
    atomizer: GraphAtomizer = Field(default_factory=GraphAtomizer, exclude=True)
    _client: QdrantClient | None = PrivateAttr(default=None)

    @property
    def embedding_config(self) -> EmbeddingConfig:
        return self.embedding.config

    def _require_sparse_embedding_tool(self) -> FastembedBm25SparseTool:
        if self.sparse_embedding is None:
            raise ValueError(
                "BM25 sparse embedding is required for vector search but "
                "sparse_embedding was not wired"
            )
        return self.sparse_embedding

    def _encode_single_query_vectors(
        self, query: str
    ) -> tuple[list[float], list[float], qdrant_models.SparseVector]:
        triples = self._encode_query_vectors_batch([query])
        return triples[0]

    def _encode_query_vectors_batch(
        self, queries: list[str]
    ) -> list[tuple[list[float], list[float], qdrant_models.SparseVector]]:
        n = len(queries)
        if n == 0:
            return []
        dense_vecs = self.embedding.embed(queries)
        if len(dense_vecs) != n:
            raise ValueError(
                "Embedding provider returned mismatched vectors for queries"
            )
        for i, vec in enumerate(dense_vecs):
            self._require_embedding_vector_length(vec, role=f"Query embedding[{i}]")
        sparse_vecs = self._require_sparse_embedding_tool().embed_sparse(queries)
        if len(sparse_vecs) != n:
            raise ValueError(
                "BM25 embedder returned mismatched sparse vectors for queries"
            )
        return [(dense_vecs[i], dense_vecs[i], sparse_vecs[i]) for i in range(n)]

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            if self.qdrant_config.uri is None:
                raise ValueError(
                    "Qdrant URI is required to initialize vector store client"
                )
            self._client = QdrantClient(
                url=self.qdrant_config.uri,
                api_key=self.qdrant_config.api_key,
                grpc_port=self.qdrant_config.grpc_port,
                prefer_grpc=self.qdrant_config.use_grpc,
            )
        return self._client

    def _ontology_collection_name(self) -> str:
        name = self.qdrant_config.ontology_collection
        if name is None:
            raise ValueError(
                "Qdrant ontology_collection is unset; ensure QdrantConfig validation"
                " ran or call apply_tenancy before vector operations"
            )
        return name

    def supports_tenancy_partition(self) -> bool:
        return True

    async def initialize(self) -> None:
        """Create ontology/facts collections and payload indexes if missing."""
        ontology_col = self.qdrant_config.ontology_collection
        facts_col = self.qdrant_config.facts_collection
        assert ontology_col is not None
        assert facts_col is not None
        self._ensure_named_vector_collection(ontology_col)
        self._ensure_named_vector_collection(facts_col)

        self._ensure_payload_index(
            collection_name=ontology_col, field_name="ontology_iri"
        )
        self._ensure_payload_index(
            collection_name=ontology_col, field_name="ontology_version"
        )
        self._ensure_payload_index(
            collection_name=ontology_col, field_name="ontology_hash"
        )
        self._ensure_payload_index(collection_name=ontology_col, field_name="iri")

    async def clean_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
    ) -> None:
        """Delete Qdrant collections named for ``tenant`` / ``project``."""
        t, p = tenant.strip(), project.strip()
        for name in (
            tenant_project_ontologies_name(t, p, sep=sep),
            tenant_project_facts_name(t, p, sep=sep),
        ):
            if self.client.collection_exists(collection_name=name):
                self.client.delete_collection(collection_name=name)
                logger.info("Deleted Qdrant collection %s", name)

    def apply_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
    ) -> None:
        """Point config at collections for ``tenant`` / ``project``.

        Call :meth:`initialize` after.
        """
        t, p = tenant.strip(), project.strip()
        ontology_name = tenant_project_ontologies_name(t, p, sep=sep)
        facts_name = tenant_project_facts_name(t, p, sep=sep)
        self.qdrant_config.ontology_collection = ontology_name
        self.qdrant_config.facts_collection = facts_name
        self.store_config.ontology_table = ontology_name
        self.store_config.facts_table = facts_name

    def _dense_dimension(self) -> int:
        return self.qdrant_config.vector_size or self.embedding_config.dimension

    def _metadata_embedding_dimension(self) -> int:
        return self._dense_dimension()

    def _validate_existing_embedding_contract(
        self, collection: str, info: qdrant_models.CollectionInfo
    ) -> None:
        raw = info.config.metadata
        if raw is None:
            meta: dict[str, Any] = {}
        elif isinstance(raw, dict):
            meta = dict(raw)
        else:
            raise ValueError(
                f"Qdrant collection '{collection}' has unsupported metadata type "
                f"{type(raw).__name__}"
            )
        validate_embedding_contract_metadata(
            collection,
            meta,
            embedding_config=self.embedding_config,
            expected_meta_dim=self._metadata_embedding_dimension(),
        )

    def _vectors_and_sparse_for_create(
        self,
    ) -> tuple[
        dict[str, qdrant_models.VectorParams],
        dict[str, qdrant_models.SparseVectorParams],
    ]:
        distance = self.qdrant_config.distance
        dense_dim = self._dense_dimension()
        vectors: dict[str, qdrant_models.VectorParams] = {
            CORE_VECTOR_NAME: qdrant_models.VectorParams(
                size=dense_dim, distance=distance
            ),
            NEIGHBORHOOD_VECTOR_NAME: qdrant_models.VectorParams(
                size=dense_dim, distance=distance
            ),
        }
        sparse: dict[str, qdrant_models.SparseVectorParams] = {
            BM25_VECTOR_NAME: qdrant_models.SparseVectorParams(modifier=None)
        }
        return (vectors, sparse)

    def _validate_collection_vector_layout(
        self, collection: str, info: qdrant_models.CollectionInfo
    ) -> None:
        distance = self.qdrant_config.distance
        dense_dim = self._dense_dimension()
        params = info.config.params
        raw_vectors = params.vectors
        vectors_map: dict[str, qdrant_models.VectorParams] = (
            dict(raw_vectors) if isinstance(raw_vectors, dict) else {}
        )
        raw_sparse = params.sparse_vectors
        sparse_map: dict[str, qdrant_models.SparseVectorParams] = (
            dict(raw_sparse) if isinstance(raw_sparse, dict) else {}
        )

        def _require_dense(name: str) -> None:
            if name not in vectors_map:
                raise ValueError(
                    f"Qdrant collection '{collection}' missing dense vector {name!r}; "
                    f"have dense keys {set(vectors_map.keys())}"
                )
            cfg = vectors_map[name]
            if cfg.size != dense_dim:
                raise EmbeddingContractMismatchError(
                    f"Qdrant collection '{collection}' vector {name!r} size "
                    f"{cfg.size} does not match configured dense size {dense_dim}. "
                    + embedding_contract_help(backend="Qdrant collection")
                )
            if cfg.distance != distance:
                raise ValueError(
                    f"Qdrant collection '{collection}' vector {name!r} "
                    f"uses distance {cfg.distance!r}; config expects {distance!r}."
                )

        _require_dense(CORE_VECTOR_NAME)
        _require_dense(NEIGHBORHOOD_VECTOR_NAME)

        bm25_cfg = sparse_map.get(BM25_VECTOR_NAME)
        if bm25_cfg is None:
            raise ValueError(
                f"Qdrant collection '{collection}' missing sparse vector "
                f"{BM25_VECTOR_NAME!r}; have sparse keys {set(sparse_map.keys())}"
            )
        if bm25_cfg.modifier is not None:
            raise ValueError(
                f"Qdrant collection '{collection}' sparse vector {BM25_VECTOR_NAME!r} "
                f"uses modifier {bm25_cfg.modifier!r}; expected no modifier "
                "(dot-product sparse scoring). Recreate the collection."
            )

    def _ensure_named_vector_collection(self, collection: str) -> None:
        metadata_dim = self._metadata_embedding_dimension()
        embedding_meta = collection_embedding_metadata(
            self.embedding_config,
            metadata_dim=metadata_dim,
        )
        vectors_cfg, sparse_cfg = self._vectors_and_sparse_for_create()
        if not self.client.collection_exists(collection_name=collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=vectors_cfg,
                sparse_vectors_config=sparse_cfg,
                metadata=embedding_meta,
            )
            logger.info(
                "Created Qdrant collection '%s' metadata_dim=%s distance=%s model=%s",
                collection,
                metadata_dim,
                self.qdrant_config.distance.value,
                embedding_meta[META_EMBEDDING_MODEL],
            )
        else:
            info = self.client.get_collection(collection_name=collection)
            self._validate_collection_vector_layout(collection, info)
            self._validate_existing_embedding_contract(collection, info)

    def index_ontology(self, ontology: Ontology) -> int:
        """Atomize + embed + upsert ontology neighborhoods."""
        atoms = self.atomizer.atomize(source=ontology, depth=1)
        if not atoms:
            return 0
        core_texts = [atom.core_representation for atom in atoms]
        neighborhood_texts = [atom.neighborhood_representation for atom in atoms]
        minimal_texts = [atom.minimal_representation for atom in atoms]

        core_vectors = self._embed_texts_batched(core_texts)
        neighborhood_vectors = self._embed_texts_batched(neighborhood_texts)
        bm25_vectors = self._embed_texts_batched_sparse(minimal_texts)

        if len(core_vectors) != len(atoms) or len(neighborhood_vectors) != len(atoms):
            raise ValueError(
                "Embedding provider returned mismatched vector counts for atoms"
            )
        if len(bm25_vectors) != len(atoms):
            raise ValueError(
                "BM25 embedder returned mismatched sparse vector counts for atoms"
            )

        points: list[qdrant_models.PointStruct] = []
        for i, atom in enumerate(atoms):
            vec_map: dict[str, Any] = {
                CORE_VECTOR_NAME: core_vectors[i],
                NEIGHBORHOOD_VECTOR_NAME: neighborhood_vectors[i],
                BM25_VECTOR_NAME: bm25_vectors[i],
            }
            points.append(
                qdrant_models.PointStruct(
                    id=point_id_for_atom(atom, store_config=self.store_config),
                    vector=vec_map,
                    payload=atom_payload(atom),
                )
            )
        collection = self._ontology_collection_name()
        for points_batch in iter_batches(points, self.qdrant_config.upsert_batch_size):
            self.client.upsert(collection_name=collection, points=points_batch)
        return len(points)

    def search_patches(
        self,
        query: str,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[GraphAtom]:
        """Search ontology atoms by text query using weighted multi-vector fusion."""
        core_q, neigh_q, bm25_q = self._encode_single_query_vectors(query)
        return self.search_by_vector(
            core_vector=core_q,
            neighborhood_vector=neigh_q,
            bm25_query_vector=bm25_q,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )

    def search_patch_hits(
        self,
        query: str,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHit]:
        """Search ontology atoms and return rank-fused scored hit objects."""
        core_q, neigh_q, bm25_q = self._encode_single_query_vectors(query)
        channel_hits = self.search_hits_by_vector(
            core_vector=core_q,
            neighborhood_vector=neigh_q,
            bm25_query_vector=bm25_q,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )
        eff_top_k = effective_top_k(self.store_config, top_k)
        cw, nw, bw = normalized_fusion_weights(self.store_config)
        return rank_fuse_channel_hits(
            channel_hits.core_hits,
            channel_hits.neighborhood_hits,
            channel_hits.bm25_hits,
            core_weight=cw,
            neighborhood_weight=nw,
            bm25_weight=bw,
            limit=eff_top_k,
        )

    def _search_patch_hits_for_query_triples(
        self,
        triples: list[tuple[list[float], list[float], qdrant_models.SparseVector]],
        top_k: int,
        filter_iri: str | None,
        filter_version: str | None,
        filter_hash: str | None,
    ) -> list[OntologySearchHitsByChannel]:
        if not triples:
            return []

        def search_one(
            t: tuple[list[float], list[float], qdrant_models.SparseVector],
        ) -> OntologySearchHitsByChannel:
            core_v, neigh_v, bm25_v = t
            return self.search_hits_by_vector(
                core_vector=core_v,
                neighborhood_vector=neigh_v,
                bm25_query_vector=bm25_v,
                top_k=top_k,
                filter_iri=filter_iri,
                filter_version=filter_version,
                filter_hash=filter_hash,
            )

        workers = min(32, len(triples))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(search_one, triples))

    def _search_patch_hits_many_impl(
        self,
        queries: list[str],
        top_k: int | None,
        filter_iri: str | None,
        filter_version: str | None,
        filter_hash: str | None,
    ) -> list[OntologySearchHitsByChannel]:
        if not queries:
            return []

        eff_top_k = effective_top_k(self.store_config, top_k)
        triples = self._encode_query_vectors_batch(queries)
        return self._search_patch_hits_for_query_triples(
            triples,
            eff_top_k,
            filter_iri,
            filter_version,
            filter_hash,
        )

    def search_patch_hits_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHitsByChannel]:
        """Search ontology atoms for many queries with split-channel outputs."""
        return self._search_patch_hits_many_impl(
            queries,
            top_k,
            filter_iri,
            filter_version,
            filter_hash,
        )

    async def asearch_patch_hits_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHitsByChannel]:
        """Async variant: one batched embed, then parallel split-channel searches."""
        if not queries:
            return []
        eff_top_k = effective_top_k(self.store_config, top_k)
        triples = await asyncio.to_thread(self._encode_query_vectors_batch, queries)
        tasks = [
            asyncio.to_thread(
                self.search_hits_by_vector,
                core_v,
                neigh_v,
                bm25_v,
                eff_top_k,
                filter_iri,
                filter_version,
                filter_hash,
            )
            for core_v, neigh_v, bm25_v in triples
        ]
        return await asyncio.gather(*tasks)

    def _parse_dense_vector(self, raw: Any) -> list[float] | None:
        if isinstance(raw, list):
            if not raw or not all(isinstance(v, int | float) for v in raw):
                return None
            return [float(v) for v in cast(list[int | float], raw)]
        return None

    def fetch_vectors(
        self,
        atom_ids: list[str],
    ) -> dict[str, tuple[list[float], list[float]]]:
        """Batch-fetch dense core/neighborhood vectors for MMR (BM25 not used)."""
        if not atom_ids:
            return {}
        point_id_to_atom_id = {point_id(atom_id): atom_id for atom_id in atom_ids}
        points = self.client.retrieve(
            collection_name=self._ontology_collection_name(),
            ids=list(point_id_to_atom_id.keys()),
            with_vectors=True,
            with_payload=False,
        )
        out: dict[str, tuple[list[float], list[float]]] = {}
        for point in points:
            atom_id = point_id_to_atom_id.get(str(point.id))
            if atom_id is None:
                continue
            point_vector = point.vector
            if not isinstance(point_vector, dict):
                continue
            core_raw = point_vector.get(CORE_VECTOR_NAME)
            neighborhood_raw = point_vector.get(NEIGHBORHOOD_VECTOR_NAME)
            core = self._parse_dense_vector(core_raw)
            neighborhood = self._parse_dense_vector(neighborhood_raw)
            if core is None or neighborhood is None:
                continue
            out[atom_id] = (core, neighborhood)
        return out

    def search_by_vector(
        self,
        core_vector: list[float],
        neighborhood_vector: list[float],
        bm25_query_vector: qdrant_models.SparseVector | None = None,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[GraphAtom]:
        """Search ontology atoms with rank fusion over named vectors."""
        channel_hits = self.search_hits_by_vector(
            core_vector=core_vector,
            neighborhood_vector=neighborhood_vector,
            bm25_query_vector=bm25_query_vector,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )
        eff_top_k = effective_top_k(self.store_config, top_k)
        cw, nw, bw = normalized_fusion_weights(self.store_config)
        fused_hits = rank_fuse_channel_hits(
            channel_hits.core_hits,
            channel_hits.neighborhood_hits,
            channel_hits.bm25_hits,
            core_weight=cw,
            neighborhood_weight=nw,
            bm25_weight=bw,
            limit=eff_top_k,
        )
        return [hit.atom for hit in fused_hits]

    def search_hits_by_vector(
        self,
        core_vector: list[float],
        neighborhood_vector: list[float],
        bm25_query_vector: qdrant_models.SparseVector | None = None,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> OntologySearchHitsByChannel:
        """Search ontology atoms and return channel-separated scored hit objects."""
        eff_top_k = effective_top_k(self.store_config, top_k)
        self._require_embedding_vector_length(core_vector, role="Query core vector")
        self._require_embedding_vector_length(
            neighborhood_vector, role="Query neighborhood vector"
        )
        search_filter = self._build_filter(
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )
        core_hits = self._query_named_vector(
            vector_name=CORE_VECTOR_NAME,
            vector=core_vector,
            limit=eff_top_k,
            search_filter=search_filter,
        )
        neighborhood_hits = self._query_named_vector(
            vector_name=NEIGHBORHOOD_VECTOR_NAME,
            vector=neighborhood_vector,
            limit=eff_top_k,
            search_filter=search_filter,
        )
        bm25_hits_raw: list[Any] = []
        if bm25_query_vector is not None:
            bm25_hits_raw = self._query_named_vector(
                vector_name=BM25_VECTOR_NAME,
                vector=bm25_query_vector,
                limit=eff_top_k,
                search_filter=search_filter,
            )
        core_typed_hits = self._points_to_hits(core_hits)
        neighborhood_typed_hits = self._points_to_hits(
            neighborhood_hits, apply_neighborhood_empty_penalty=True
        )
        bm25_typed_hits = self._points_to_hits(bm25_hits_raw)
        if self.store_config.dedup_query_hits_by_iri:
            core_typed_hits = dedupe_hits_by_identity(
                core_typed_hits, store_config=self.store_config
            )
            neighborhood_typed_hits = dedupe_hits_by_identity(
                neighborhood_typed_hits, store_config=self.store_config
            )
            bm25_typed_hits = dedupe_hits_by_identity(
                bm25_typed_hits, store_config=self.store_config
            )
        return OntologySearchHitsByChannel(
            core_hits=core_typed_hits,
            neighborhood_hits=neighborhood_typed_hits,
            bm25_hits=bm25_typed_hits,
        )

    def _points_to_hits(
        self,
        points: list[Any],
        *,
        apply_neighborhood_empty_penalty: bool = False,
    ) -> list[OntologySearchHit]:
        hits: list[OntologySearchHit] = []
        for point in points:
            score = float(point.score) if point.score is not None else 0.0
            if apply_neighborhood_empty_penalty:
                payload = point.payload or {}
                neighborhood_text = str(payload.get("neighborhood_representation", ""))
                if (
                    neighborhood_text.strip().lower()
                    == "no neighborhood facts available"
                ):
                    score = 0.0
            atom = self._point_to_atom(point)
            atom.score = score
            hits.append(OntologySearchHit(atom=atom, score=score))
        return hits

    def delete_ontology(
        self,
        iri: str,
        version: str | None = None,
        ontology_hash: str | None = None,
    ) -> None:
        """Delete atoms associated with one ontology IRI and optional version/hash."""
        delete_filter = self._build_filter(
            filter_iri=iri, filter_version=version, filter_hash=ontology_hash
        )
        if delete_filter is None:
            return
        self.client.delete(
            collection_name=self._ontology_collection_name(),
            points_selector=qdrant_models.FilterSelector(filter=delete_filter),
        )

    def _build_filter(
        self,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> qdrant_models.Filter | None:
        conditions: list[qdrant_models.Condition] = []
        if filter_iri is not None:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="ontology_iri", match=qdrant_models.MatchValue(value=filter_iri)
                )
            )
        if filter_version is not None:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="ontology_version",
                    match=qdrant_models.MatchValue(value=filter_version),
                )
            )
        if filter_hash is not None:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="ontology_hash",
                    match=qdrant_models.MatchValue(value=filter_hash),
                )
            )
        if not conditions:
            return None
        return qdrant_models.Filter(must=conditions)

    def _point_to_atom(self, point: Any) -> GraphAtom:
        payload = point.payload or {}
        score = float(point.score) if point.score is not None else None
        return atom_from_payload(
            payload,
            score=score,
            default_id=str(point.id),
        )

    def _require_embedding_vector_length(
        self,
        vector: list[float],
        *,
        role: str,
    ) -> None:
        require_embedding_vector_length(
            vector,
            role=role,
            expected=self._dense_dimension(),
        )

    def delete_duplicate_iri_points(self, *, batch_size: int = 512) -> int:
        """Delete duplicate points sharing the same configured identity key."""
        collection_name = self._ontology_collection_name()
        seen_by_key: dict[str, qdrant_models.ExtendedPointId] = {}
        duplicate_ids: list[qdrant_models.ExtendedPointId] = []
        offset: Any = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                limit=batch_size,
            )
            if not points:
                break
            for point in points:
                atom = self._point_to_atom(point)
                key = identity_key_for_atom(atom, store_config=self.store_config)
                if key in seen_by_key:
                    duplicate_ids.append(point.id)
                else:
                    seen_by_key[key] = point.id
            if next_offset is None:
                break
            offset = next_offset
        if not duplicate_ids:
            return 0
        self.client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.PointIdsList(points=duplicate_ids),
        )
        return len(duplicate_ids)

    def count_points_by_ontology_iri(self, *, batch_size: int = 512) -> dict[str, int]:
        """Count indexed atoms grouped by ``ontology_iri`` payload (diagnostics)."""
        collection_name = self._ontology_collection_name()
        counts: dict[str, int] = defaultdict(int)
        offset: Any = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                limit=batch_size,
            )
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                onto_iri = str(payload.get("ontology_iri", ""))
                if onto_iri:
                    counts[onto_iri] += 1
            if next_offset is None:
                break
            offset = next_offset
        return dict(counts)

    def _query_named_vector(
        self,
        vector_name: str,
        vector: ChannelVector,
        limit: int,
        search_filter: qdrant_models.Filter | None,
    ) -> list[Any]:
        response = self.client.query_points(
            collection_name=self._ontology_collection_name(),
            query=vector,
            using=vector_name,
            query_filter=search_filter,
            with_payload=True,
            limit=limit,
        )
        return response.points

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
                self._require_embedding_vector_length(
                    vec,
                    role=f"Index embedding batch offset {len(vectors) + j}",
                )
            vectors.extend(batch_vectors)
        return vectors

    def _embed_texts_batched_sparse(
        self, texts: list[str]
    ) -> list[qdrant_models.SparseVector]:
        if not texts:
            return []
        out: list[qdrant_models.SparseVector] = []
        sparse_tool = self._require_sparse_embedding_tool()
        for batch in iter_batches(texts, self.store_config.embedding_batch_size):
            batch_vectors = sparse_tool.embed_sparse(batch)
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    "BM25 embedder returned mismatched sparse vectors for batch"
                )
            out.extend(batch_vectors)
        return out

    def _ensure_payload_index(self, collection_name: str, field_name: str) -> None:
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.debug(
                "Qdrant payload index '%s' on '%s' already exists",
                field_name,
                collection_name,
            )
