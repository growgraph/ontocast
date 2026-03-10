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
from ontocast.tool.vector_store.core import OntologyAtom, VectorStoreTool
from ontocast.tool.vector_store.embedding import EmbeddingTool

logger = logging.getLogger(__name__)


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
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size, distance=qdrant_models.Distance.COSINE
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' with vector size %s",
                collection,
                vector_size,
            )
        else:
            info = self.client.get_collection(collection_name=collection)
            existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing_size != vector_size:
                raise ValueError(
                    f"Qdrant collection '{collection}' vector size {existing_size} "
                    f"does not match configured size {vector_size}"
                )

        try:
            self.client.create_payload_index(
                collection_name=collection,
                field_name="ontology_iri",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.debug("Qdrant payload index 'ontology_iri' already exists")

    def index_ontology(self, ontology: Ontology) -> int:
        """Atomize + embed + upsert ontology neighborhoods."""
        atoms = self.atomizer.atomize(ontology=ontology, depth=1)
        if not atoms:
            return 0
        vectors = self.embedding.embed([atom.turtle for atom in atoms])
        points: list[qdrant_models.PointStruct] = []
        for atom, vector in zip(atoms, vectors):
            points.append(
                qdrant_models.PointStruct(
                    id=self._point_id(atom.atom_id),
                    vector=vector,
                    payload=self._atom_payload(atom),
                )
            )
        self.client.upsert(collection_name=self.config.collection, points=points)
        return len(points)

    def search_patches(
        self, query: str, top_k: int = 10, filter_iri: str | None = None
    ) -> list[OntologyAtom]:
        """Search ontology atoms by text query."""
        query_vector = self.embedding.embed_one(query)
        return self.search_by_vector(
            vector=query_vector, top_k=top_k, filter_iri=filter_iri
        )

    def search_by_vector(
        self, vector: list[float], top_k: int = 10, filter_iri: str | None = None
    ) -> list[OntologyAtom]:
        """Search ontology atoms by an already computed embedding vector."""
        search_filter = self._build_filter(filter_iri=filter_iri)
        response = self.client.query_points(
            collection_name=self.config.collection,
            query=vector,
            query_filter=search_filter,
            with_payload=True,
            limit=top_k,
        )
        points = response.points
        return [self._point_to_atom(point) for point in points]

    def delete_ontology(self, iri: str) -> None:
        """Delete all atoms associated with one ontology IRI."""
        self.client.delete(
            collection_name=self.config.collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="ontology_iri",
                            match=qdrant_models.MatchValue(value=iri),
                        )
                    ]
                )
            ),
        )

    def reindex_ontology(self, ontology: Ontology) -> int:
        """Replace all atoms for a given ontology and return indexed count."""
        self.delete_ontology(ontology.iri)
        return self.index_ontology(ontology)

    def _build_filter(self, filter_iri: str | None) -> qdrant_models.Filter | None:
        if filter_iri is None:
            return None
        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="ontology_iri", match=qdrant_models.MatchValue(value=filter_iri)
                )
            ]
        )

    def _point_to_atom(self, point: Any) -> OntologyAtom:
        payload = point.payload or {}
        created_at_raw = payload.get("created_at")
        created_at = self._parse_created_at(created_at_raw)
        return OntologyAtom(
            atom_id=str(payload.get("atom_id", point.id)),
            ontology_iri=str(payload.get("ontology_iri", "")),
            ontology_id=payload.get("ontology_id"),
            ontology_hash=payload.get("ontology_hash"),
            node_uri=str(payload.get("node_uri", "")),
            turtle=str(payload.get("turtle", "")),
            created_at=created_at,
            score=float(point.score) if point.score is not None else None,
        )

    def _atom_payload(self, atom: OntologyAtom) -> dict[str, Any]:
        return {
            "atom_id": atom.atom_id,
            "ontology_iri": atom.ontology_iri,
            "ontology_id": atom.ontology_id,
            "ontology_hash": atom.ontology_hash,
            "node_uri": atom.node_uri,
            "turtle": atom.turtle,
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
