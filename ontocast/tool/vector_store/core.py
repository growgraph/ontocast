"""Core contracts and models for ontology vector storage."""

from __future__ import annotations

import abc
from datetime import datetime, timezone

from pydantic import Field, field_validator

from ontocast.onto.model import BasePydanticModel
from ontocast.onto.ontology import Ontology
from ontocast.tool.onto import Tool
from ontocast.tool.representation_contract import combine_embedding_text
from ontocast.tool.representation_text import ROLE_PREDICATE, ROLE_RESOURCE

VECTOR_ENTITY_ROLES = frozenset({ROLE_RESOURCE, ROLE_PREDICATE})


def canonicalize_entity_role(role: str | None) -> str | None:
    """Normalize role labels to vector-store vocabulary."""
    if role is None:
        return None
    normalized = role.strip().lower()
    if normalized in VECTOR_ENTITY_ROLES:
        return normalized
    if normalized in {"property", "predicate"}:
        return ROLE_PREDICATE
    if normalized in {"class", "instance", "resource"}:
        return ROLE_RESOURCE
    return None


class GraphAtom(BasePydanticModel):
    """Embedding-ready ontology entity atom."""

    atom_id: str = Field(
        description="Deterministic hash identifier for the atom content."
    )
    ontology_iri: str = Field(description="Source ontology IRI.")
    ontology_id: str | None = Field(
        default=None, description="Optional source ontology identifier."
    )
    ontology_hash: str | None = Field(
        default=None, description="Hash/version of the source ontology."
    )
    ontology_version: str | None = Field(
        default=None, description="Semantic version of the source ontology."
    )
    iri: str = Field(description="Focal entity IRI represented by this atom.")
    entity_role: str | None = Field(
        default=None,
        description="Role of focal entity in graph context: resource or predicate.",
    )
    core_representation: str = Field(
        description="High-precision natural language text (labels, types, descriptions)."
    )
    minimal_representation: str = Field(
        default="",
        description=(
            "IRI local name with camelCase/PascalCase split into space-separated terms; "
            "used for BM25 (keyword) indexing."
        ),
    )
    neighborhood_representation: str = Field(
        description="Neighborhood relation text for disambiguation context."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Atom creation timestamp (UTC).",
    )
    score: float | None = Field(
        default=None,
        description="Optional similarity score populated by vector search.",
    )

    @field_validator("entity_role", mode="before")
    @classmethod
    def _normalize_entity_role(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return canonicalize_entity_role(str(value))

    @property
    def representation(self) -> str:
        """Combined embedding text view for generic consumers."""
        return combine_embedding_text(self)


class OntologySearchHit(BasePydanticModel):
    """Typed retrieval result that separates atom payload from ranking metadata."""

    atom: GraphAtom
    score: float = Field(description="Channel-specific retrieval score.")


class OntologySearchHitsByChannel(BasePydanticModel):
    """Per-query retrieval hits split by vector channel (dense core/neighborhood + optional BM25)."""

    core_hits: list[OntologySearchHit] = Field(
        default_factory=list,
        description="Top hits from the dense core vector channel.",
    )
    neighborhood_hits: list[OntologySearchHit] = Field(
        default_factory=list,
        description="Top hits from the dense neighborhood vector channel.",
    )
    bm25_hits: list[OntologySearchHit] = Field(
        default_factory=list,
        description="Top hits from the sparse BM25 lane (minimal IRI text).",
    )


class VectorStoreTool(Tool):
    """Abstract interface for vector store implementations."""

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Prepare schema/collections in the backing vector store."""

    @abc.abstractmethod
    def index_ontology(self, ontology: Ontology) -> int:
        """Index an ontology and return number of indexed atoms."""

    @abc.abstractmethod
    def search_patches(
        self,
        query: str,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[GraphAtom]:
        """Search ontology patches by query text (``top_k`` None → store default)."""

    @abc.abstractmethod
    def delete_ontology(
        self,
        iri: str,
        version: str | None = None,
        ontology_hash: str | None = None,
    ) -> None:
        """Delete all indexed atoms for a specific ontology IRI."""

    def supports_tenancy_partition(self) -> bool:
        """True if :meth:`clean_tenancy` clears isolated collections for (tenant, project)."""
        return False

    async def clean_tenancy(self, tenant: str, project: str) -> None:
        """Drop or empty vector collections derived from ``tenant`` / ``project``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not isolate vectors by tenant/project"
        )


CORE_VECTOR_NAME = "core"
NEIGHBORHOOD_VECTOR_NAME = "neighborhood"
BM25_VECTOR_NAME = "bm25"
