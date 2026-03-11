"""Qdrant-backed vector store for ontology atoms."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import Field, PrivateAttr
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from ontocast.config import QdrantConfig
from ontocast.onto.ontology import Ontology
from ontocast.tool.vector_store.atomizer import OntologyAtomizer
from ontocast.tool.vector_store.core import (
    OntologyAtom,
    OntologySearchHit,
    VectorStoreTool,
    canonicalize_entity_role,
)
from ontocast.tool.vector_store.embedding import EmbeddingTool

logger = logging.getLogger(__name__)

CORE_VECTOR_NAME = "core"
NEIGHBORHOOD_VECTOR_NAME = "neighborhood"


class QdrantVectorStore(VectorStoreTool):
    """Stores ontology atoms in Qdrant and supports similarity lookup."""

    config: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingTool = Field(exclude=True)
    atomizer: OntologyAtomizer = Field(default_factory=OntologyAtomizer, exclude=True)
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

    async def initialize(self) -> None:
        """Create collection and payload indexes if missing."""
        vector_size = self._vector_size()
        collection = self.config.collection

        if not self.client.collection_exists(collection_name=collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config={
                    CORE_VECTOR_NAME: qdrant_models.VectorParams(
                        size=vector_size, distance=qdrant_models.Distance.COSINE
                    ),
                    NEIGHBORHOOD_VECTOR_NAME: qdrant_models.VectorParams(
                        size=vector_size, distance=qdrant_models.Distance.COSINE
                    ),
                },
            )
            logger.info(
                "Created Qdrant collection '%s' with vector size %s",
                collection,
                vector_size,
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
                    raise ValueError(
                        f"Qdrant collection '{collection}' vector sizes do not match "
                        f"configured size {vector_size}"
                    )
            else:
                raise ValueError(
                    f"Qdrant collection '{collection}' must use named vectors "
                    f"'{CORE_VECTOR_NAME}' and '{NEIGHBORHOOD_VECTOR_NAME}'"
                )

        self._ensure_payload_index(field_name="ontology_iri")
        self._ensure_payload_index(field_name="ontology_version")
        self._ensure_payload_index(field_name="ontology_hash")

    def index_ontology(self, ontology: Ontology) -> int:
        """Atomize + embed + upsert ontology neighborhoods."""
        atoms = self.atomizer.atomize(ontology=ontology, depth=1)
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
        for points_batch in self._iter_batches(points, self.config.upsert_batch_size):
            self.client.upsert(
                collection_name=self.config.collection, points=points_batch
            )
        return len(points)

    def search_patches(
        self,
        query: str,
        top_k: int = 10,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologyAtom]:
        """Search ontology atoms by text query using weighted dual-vector fusion."""
        core_query_vector = self.embedding.embed_one(query)
        neighborhood_query_vector = self.embedding.embed_one(query)
        return self.search_by_vector(
            core_vector=core_query_vector,
            neighborhood_vector=neighborhood_query_vector,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )

    def search_patch_hits(
        self,
        query: str,
        top_k: int = 10,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHit]:
        """Search ontology atoms and return explicit scored hit objects."""
        core_query_vector = self.embedding.embed_one(query)
        neighborhood_query_vector = self.embedding.embed_one(query)
        return self.search_hits_by_vector(
            core_vector=core_query_vector,
            neighborhood_vector=neighborhood_query_vector,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )

    def search_by_vector(
        self,
        core_vector: list[float],
        neighborhood_vector: list[float],
        top_k: int = 10,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologyAtom]:
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
        top_k: int = 10,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHit]:
        """Search ontology atoms with weighted fusion and explicit score wrapper."""
        search_filter = self._build_filter(
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )
        fetch_limit = max(top_k, self.config.top_k) * 2
        core_hits = self._query_named_vector(
            vector_name=CORE_VECTOR_NAME,
            vector=core_vector,
            limit=fetch_limit,
            search_filter=search_filter,
        )
        neighborhood_hits = self._query_named_vector(
            vector_name=NEIGHBORHOOD_VECTOR_NAME,
            vector=neighborhood_vector,
            limit=fetch_limit,
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
        )[:top_k]
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
            collection_name=self.config.collection,
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

    def _point_to_atom(self, point: Any) -> OntologyAtom:
        payload = point.payload or {}
        created_at_raw = payload.get("created_at")
        created_at = self._parse_created_at(created_at_raw)
        return OntologyAtom(
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

    def _atom_payload(self, atom: OntologyAtom) -> dict[str, Any]:
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
            collection_name=self.config.collection,
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
            vectors.extend(batch_vectors)
        return vectors

    def _iter_batches(self, items: list[Any], batch_size: int) -> list[list[Any]]:
        batches: list[list[Any]] = []
        for index in range(0, len(items), batch_size):
            batches.append(items[index : index + batch_size])
        return batches

    def _ensure_payload_index(self, field_name: str) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.config.collection,
                field_name=field_name,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.debug("Qdrant payload index '%s' already exists", field_name)
