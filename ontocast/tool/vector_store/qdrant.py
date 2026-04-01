"""Qdrant-backed vector store for ontology atoms."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from pydantic import Field, PrivateAttr
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from ontocast.config import QdrantConfig
from ontocast.onto.ontology import Ontology
from ontocast.onto.tenancy import (
    TENANCY_SEP,
    tenant_project_facts_name,
    tenant_project_ontologies_name,
)
from ontocast.tool.vector_store.atomizer import GraphAtomizer
from ontocast.tool.vector_store.core import (
    GraphAtom,
    OntologySearchHit,
    VectorStoreTool,
    canonicalize_entity_role,
)
from ontocast.tool.vector_store.embedding import EmbeddingTool

logger = logging.getLogger(__name__)

CORE_VECTOR_NAME = "core"
NEIGHBORHOOD_VECTOR_NAME = "neighborhood"


class EmbeddingContractMismatchError(ValueError):
    """Embedding vectors or collection metadata disagree with the active embedding config.

    Typical causes: switching ``EMBEDDING_MODEL_NAME`` without recreating the Qdrant
    collection, or a provider returning an unexpected vector length.
    """


def _embedding_contract_help() -> str:
    return (
        "Align EmbeddingConfig (EMBEDDING_*) with the collection: use the same model "
        "and dimension as when the collection was created, or drop the Qdrant "
        "ontology collection and let initialize() recreate it."
    )


# Qdrant collection metadata (CollectionConfig.metadata).
# Values must match EmbeddingConfig on every initialize().
QDRANT_META_EMBEDDING_DIMENSION = "embedding_dimension"
QDRANT_META_EMBEDDING_MODEL = "embedding_model"


class QdrantVectorStore(VectorStoreTool):
    """Stores ontology atoms in Qdrant and supports similarity lookup."""

    config: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingTool = Field(exclude=True)
    atomizer: GraphAtomizer = Field(default_factory=GraphAtomizer, exclude=True)
    _client: QdrantClient | None = PrivateAttr(default=None)

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            if self.config.uri is None:
                raise ValueError(
                    "Qdrant URI is required to initialize vector store client"
                )
            self._client = QdrantClient(
                url=self.config.uri,
                api_key=self.config.api_key,
                grpc_port=self.config.grpc_port,
                prefer_grpc=self.config.use_grpc,
            )
        return self._client

    def _ontology_collection_name(self) -> str:
        name = self.config.ontology_collection
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
        vector_size = self._vector_size()
        ontology_col = self.config.ontology_collection
        facts_col = self.config.facts_collection
        assert ontology_col is not None
        assert facts_col is not None
        self._ensure_named_vector_collection(ontology_col, vector_size)
        self._ensure_named_vector_collection(facts_col, vector_size)

        self._ensure_payload_index(
            collection_name=ontology_col, field_name="ontology_iri"
        )
        self._ensure_payload_index(
            collection_name=ontology_col, field_name="ontology_version"
        )
        self._ensure_payload_index(
            collection_name=ontology_col, field_name="ontology_hash"
        )

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
        self.config.ontology_collection = tenant_project_ontologies_name(t, p, sep=sep)
        self.config.facts_collection = tenant_project_facts_name(t, p, sep=sep)

    def _embedding_model_fingerprint(self) -> str:
        ec = self.embedding.config
        return f"{ec.provider.value}:{ec.model_name}"

    def _collection_embedding_metadata(self, vector_size: int) -> dict[str, Any]:
        return {
            QDRANT_META_EMBEDDING_DIMENSION: vector_size,
            QDRANT_META_EMBEDDING_MODEL: self._embedding_model_fingerprint(),
        }

    def _coerce_metadata_int(self, value: Any, *, field: str, collection: str) -> int:
        if type(value) is bool:
            raise ValueError(
                f"Qdrant collection '{collection}' metadata {field!r} has invalid type"
            )
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip(), 10)
            except ValueError as exc:
                raise ValueError(
                    f"Qdrant collection '{collection}' metadata {field!r} "
                    "is not an integer"
                ) from exc
        raise ValueError(
            f"Qdrant collection '{collection}' metadata {field!r} has invalid type"
        )

    def _validate_existing_embedding_contract(
        self, collection: str, vector_size: int, info: qdrant_models.CollectionInfo
    ) -> None:
        raw = info.config.metadata
        if raw is None:
            meta = {}
        elif isinstance(raw, Mapping):
            meta = dict(raw)
        else:
            raise ValueError(
                f"Qdrant collection '{collection}' has unsupported metadata type "
                f"{type(raw).__name__}"
            )
        dim_key = QDRANT_META_EMBEDDING_DIMENSION
        model_key = QDRANT_META_EMBEDDING_MODEL
        if dim_key not in meta or model_key not in meta:
            raise EmbeddingContractMismatchError(
                f"Qdrant collection '{collection}' is missing OntoCast "
                f"embedding metadata ({dim_key!r}, {model_key!r}). "
                "Drop and recreate the collection. " + _embedding_contract_help()
            )
        stored_dim = self._coerce_metadata_int(
            meta[dim_key], field=dim_key, collection=collection
        )
        stored_model = meta[model_key]
        if not isinstance(stored_model, str):
            raise ValueError(
                f"Qdrant collection '{collection}' metadata {model_key!r} "
                "must be a string"
            )
        expected_model = self._embedding_model_fingerprint()
        if stored_dim != vector_size or stored_model != expected_model:
            raise EmbeddingContractMismatchError(
                f"Qdrant collection '{collection}' embedding contract mismatch: "
                f"collection has dimension={stored_dim}, model={stored_model!r}; "
                f"current config expects dimension={vector_size}, "
                f"model={expected_model!r}. " + _embedding_contract_help()
            )

    def _ensure_named_vector_collection(
        self, collection: str, vector_size: int
    ) -> None:
        embedding_meta = self._collection_embedding_metadata(vector_size)
        distance = self.config.distance
        if not self.client.collection_exists(collection_name=collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config={
                    CORE_VECTOR_NAME: qdrant_models.VectorParams(
                        size=vector_size, distance=distance
                    ),
                    NEIGHBORHOOD_VECTOR_NAME: qdrant_models.VectorParams(
                        size=vector_size, distance=distance
                    ),
                },
                metadata=embedding_meta,
            )
            logger.info(
                "Created Qdrant collection '%s' with vector size %s, distance %s, "
                "and embedding model %s",
                collection,
                vector_size,
                self.config.distance.value,
                embedding_meta[QDRANT_META_EMBEDDING_MODEL],
            )
        else:
            info = self.client.get_collection(collection_name=collection)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict):
                expected_names = {CORE_VECTOR_NAME, NEIGHBORHOOD_VECTOR_NAME}
                existing_names = set(vectors_config.keys())
                if not expected_names.issubset(existing_names):
                    raise ValueError(
                        f"Qdrant collection '{collection}' vectors {existing_names} "
                        f"do not include required vectors {expected_names}"
                    )
                core_size = vectors_config[CORE_VECTOR_NAME].size
                neighborhood_size = vectors_config[NEIGHBORHOOD_VECTOR_NAME].size
                if core_size != vector_size or neighborhood_size != vector_size:
                    raise EmbeddingContractMismatchError(
                        f"Qdrant collection '{collection}' vector sizes "
                        f"({core_size=}, {neighborhood_size=}) do not match "
                        f"configured size {vector_size}. " + _embedding_contract_help()
                    )
                for vec_name in (CORE_VECTOR_NAME, NEIGHBORHOOD_VECTOR_NAME):
                    actual_dist = vectors_config[vec_name].distance
                    if actual_dist != distance:
                        raise ValueError(
                            f"Qdrant collection '{collection}' vector {vec_name!r} "
                            f"uses distance {actual_dist!r}; config expects "
                            f"{distance!r}."
                        )
            else:
                raise ValueError(
                    f"Qdrant collection '{collection}' must use named vectors "
                    f"'{CORE_VECTOR_NAME}' and '{NEIGHBORHOOD_VECTOR_NAME}'"
                )
            self._validate_existing_embedding_contract(collection, vector_size, info)

    def index_ontology(self, ontology: Ontology) -> int:
        """Atomize + embed + upsert ontology neighborhoods."""
        atoms = self.atomizer.atomize(source=ontology, depth=1)
        if not atoms:
            return 0
        core_vectors = self._embed_texts_batched(
            [atom.core_representation for atom in atoms]
        )
        neighborhood_vectors = self._embed_texts_batched(
            [atom.neighborhood_representation for atom in atoms]
        )
        if len(core_vectors) != len(atoms) or len(neighborhood_vectors) != len(atoms):
            raise ValueError(
                "Embedding provider returned mismatched vector counts for atoms"
            )

        points: list[qdrant_models.PointStruct] = []
        for atom, core_vector, neighborhood_vector in zip(
            atoms, core_vectors, neighborhood_vectors
        ):
            points.append(
                qdrant_models.PointStruct(
                    id=self._point_id(atom.atom_id),
                    vector={
                        CORE_VECTOR_NAME: core_vector,
                        NEIGHBORHOOD_VECTOR_NAME: neighborhood_vector,
                    },
                    payload=self._atom_payload(atom),
                )
            )
        collection = self._ontology_collection_name()
        for points_batch in self._iter_batches(points, self.config.upsert_batch_size):
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
        """Search ontology atoms by text query using weighted dual-vector fusion."""
        query_vector = self.embedding.embed_one(query)
        return self.search_by_vector(
            core_vector=query_vector,
            neighborhood_vector=query_vector,
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
        """Search ontology atoms and return explicit scored hit objects."""
        query_vector = self.embedding.embed_one(query)
        return self.search_hits_by_vector(
            core_vector=query_vector,
            neighborhood_vector=query_vector,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )

    def _search_patch_hits_for_query_vectors(
        self,
        query_vectors: list[list[float]],
        top_k: int,
        filter_iri: str | None,
        filter_version: str | None,
        filter_hash: str | None,
    ) -> list[list[OntologySearchHit]]:
        """Run dual-vector fusion search per query vector (parallel in threads)."""
        if not query_vectors:
            return []

        def search_one(query_vector: list[float]) -> list[OntologySearchHit]:
            return self.search_hits_by_vector(
                core_vector=query_vector,
                neighborhood_vector=query_vector,
                top_k=top_k,
                filter_iri=filter_iri,
                filter_version=filter_version,
                filter_hash=filter_hash,
            )

        workers = min(32, len(query_vectors))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(search_one, query_vectors))

    def _search_patch_hits_many_impl(
        self,
        queries: list[str],
        top_k: int | None,
        filter_iri: str | None,
        filter_version: str | None,
        filter_hash: str | None,
    ) -> list[list[OntologySearchHit]]:
        if not queries:
            return []

        eff_top_k = self._effective_top_k(top_k)
        query_vectors = self.embedding.embed(queries)
        if len(query_vectors) != len(queries):
            raise ValueError(
                "Embedding provider returned mismatched vectors for queries"
            )
        for i, vec in enumerate(query_vectors):
            self._require_embedding_vector_length(vec, role=f"Query embedding[{i}]")
        return self._search_patch_hits_for_query_vectors(
            query_vectors,
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
    ) -> list[list[OntologySearchHit]]:
        """Search ontology atoms for many queries with batched embedding calls."""
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
    ) -> list[list[OntologySearchHit]]:
        """Async variant: one batched embed, then parallel Qdrant searches."""
        if not queries:
            return []
        eff_top_k = self._effective_top_k(top_k)
        query_vectors = await asyncio.to_thread(self.embedding.embed, queries)
        if len(query_vectors) != len(queries):
            raise ValueError(
                "Embedding provider returned mismatched vectors for queries"
            )
        for i, vec in enumerate(query_vectors):
            self._require_embedding_vector_length(vec, role=f"Query embedding[{i}]")
        tasks = [
            asyncio.to_thread(
                self.search_hits_by_vector,
                query_vector,
                query_vector,
                eff_top_k,
                filter_iri,
                filter_version,
                filter_hash,
            )
            for query_vector in query_vectors
        ]
        return await asyncio.gather(*tasks)

    def search_by_vector(
        self,
        core_vector: list[float],
        neighborhood_vector: list[float],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[GraphAtom]:
        """Search ontology atoms with weighted fusion over named vectors."""
        hits = self.search_hits_by_vector(
            core_vector=core_vector,
            neighborhood_vector=neighborhood_vector,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )
        return [hit.atom for hit in hits]

    def search_hits_by_vector(
        self,
        core_vector: list[float],
        neighborhood_vector: list[float],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHit]:
        """Search ontology atoms with weighted fusion and explicit score wrapper."""
        eff_top_k = self._effective_top_k(top_k)
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
        fused_hits: dict[str, tuple[Any, float]] = {}
        core_weight = self.config.fusion_core_weight
        neighborhood_weight = self.config.fusion_neighborhood_weight

        for point in core_hits:
            point_id = str(point.id)
            score = float(point.score) if point.score is not None else 0.0
            weighted = core_weight * score
            fused_hits[point_id] = (point, weighted)

        for point in neighborhood_hits:
            point_id = str(point.id)
            score = float(point.score) if point.score is not None else 0.0
            payload = point.payload or {}
            neighborhood_text = str(payload.get("neighborhood_representation", ""))
            effective_weight = (
                0.0
                if neighborhood_text.strip().lower()
                == "no neighborhood facts available"
                else neighborhood_weight
            )
            weighted = effective_weight * score
            if point_id in fused_hits:
                existing_point, existing_score = fused_hits[point_id]
                fused_hits[point_id] = (existing_point, existing_score + weighted)
            else:
                fused_hits[point_id] = (point, weighted)

        ranked_points = sorted(
            fused_hits.values(), key=lambda item: item[1], reverse=True
        )[:eff_top_k]
        hits: list[OntologySearchHit] = []
        for point, fused_score in ranked_points:
            atom = self._point_to_atom(point)
            atom.score = fused_score
            hits.append(OntologySearchHit(atom=atom, score=fused_score))
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

    def reindex_ontology(self, ontology: Ontology) -> int:
        """Replace all atoms for a given ontology and return indexed count."""
        self.delete_ontology(ontology.iri)
        return self.index_ontology(ontology)

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
        created_at_raw = payload.get("created_at")
        created_at = self._parse_created_at(created_at_raw)
        return GraphAtom(
            atom_id=str(payload.get("atom_id", point.id)),
            ontology_iri=str(payload.get("ontology_iri", "")),
            ontology_id=payload.get("ontology_id"),
            ontology_hash=payload.get("ontology_hash"),
            ontology_version=payload.get("ontology_version"),
            iri=str(payload.get("iri", "")),
            entity_role=canonicalize_entity_role(payload.get("entity_role")),
            core_representation=str(payload.get("core_representation", "")),
            neighborhood_representation=str(
                payload.get("neighborhood_representation", "")
            ),
            created_at=created_at,
            score=float(point.score) if point.score is not None else None,
        )

    def _atom_payload(self, atom: GraphAtom) -> dict[str, Any]:
        return {
            "atom_id": atom.atom_id,
            "ontology_iri": atom.ontology_iri,
            "ontology_id": atom.ontology_id,
            "ontology_hash": atom.ontology_hash,
            "ontology_version": atom.ontology_version,
            "iri": atom.iri,
            "entity_role": canonicalize_entity_role(atom.entity_role),
            "core_representation": atom.core_representation,
            "neighborhood_representation": atom.neighborhood_representation,
            "created_at": atom.created_at.isoformat(),
        }

    def _parse_created_at(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def _vector_size(self) -> int:
        return self.config.vector_size or self.embedding.config.dimension

    def _effective_top_k(self, top_k: int | None) -> int:
        """Resolve retrieval depth: explicit ``top_k`` overrides :attr:`QdrantConfig.top_k`."""
        if top_k is not None:
            return top_k
        return self.config.top_k

    def _require_embedding_vector_length(
        self,
        vector: list[float],
        *,
        role: str,
    ) -> None:
        expected = self._vector_size()
        if len(vector) != expected:
            raise EmbeddingContractMismatchError(
                f"{role} vector length {len(vector)} does not match the configured "
                f"collection embedding dimension {expected}. "
                + _embedding_contract_help()
            )

    def _point_id(self, atom_id: str) -> str:
        """Return a Qdrant-compatible point id (UUID string)."""
        try:
            return str(uuid.UUID(atom_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, atom_id))

    def _query_named_vector(
        self,
        vector_name: str,
        vector: list[float],
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
        for batch in self._iter_batches(texts, self.config.embedding_batch_size):
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

    def _iter_batches(self, items: list[Any], batch_size: int) -> list[list[Any]]:
        batches: list[list[Any]] = []
        for index in range(0, len(items), batch_size):
            batches.append(items[index : index + batch_size])
        return batches

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
