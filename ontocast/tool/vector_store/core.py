"""Core contracts and models for ontology vector storage."""

from __future__ import annotations

import abc
import asyncio
import logging
from datetime import datetime, timezone

from pydantic import Field, field_validator

from ontocast.config import VectorStoreConfig
from ontocast.onto.model import BasePydanticModel
from ontocast.onto.ontology import Ontology
from ontocast.onto.tenancy import TENANCY_SEP
from ontocast.tool.onto import Tool
from ontocast.tool.representation_contract import combine_embedding_text
from ontocast.tool.representation_text import ROLE_PREDICATE, ROLE_RESOURCE
from ontocast.tool.vector_store.embedding import (
    EmbeddingTool,
    FastembedBm25SparseTool,
)

logger = logging.getLogger(__name__)

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
    lexical_triggers: list[str] = Field(
        default_factory=list,
        description=(
            "Case-preserved literal tokens (symbols, notations, formula codes) "
            "used by the lexical-trigger retrieval lane for exact text matching."
        ),
    )
    symbol_surfaces: list[str] = Field(
        default_factory=list,
        description=(
            "Case-preserved declared symbol/notation surface forms "
            "(skos:notation, qudt:symbol, qudt:ucumCode). The embedded/BM25 "
            "text is case-folded, so these carry the only case-significant "
            "evidence at merge time — used to demote counterfeit symbol "
            "matches like prose 'meV' retrieving symbol 'MeV'."
        ),
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


class VectorStoreManager(Tool):
    """Abstract interface for vector store implementations."""

    store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingTool | None = Field(default=None, exclude=True)
    sparse_embedding: FastembedBm25SparseTool | None = Field(default=None, exclude=True)

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
    def search_patch_hits(
        self,
        query: str,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHit]:
        """Search ontology atoms and return rank-fused scored hit objects."""

    @abc.abstractmethod
    def search_patch_hits_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHitsByChannel]:
        """Search ontology atoms for many queries with split-channel outputs."""

    @abc.abstractmethod
    async def asearch_patch_hits_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHitsByChannel]:
        """Async variant of :meth:`search_patch_hits_many`."""

    @abc.abstractmethod
    def fetch_vectors(
        self,
        atom_ids: list[str],
    ) -> dict[str, tuple[list[float], list[float]]]:
        """Batch-fetch dense core/neighborhood vectors for MMR."""

    async def afetch_vectors(
        self,
        atom_ids: list[str],
    ) -> dict[str, tuple[list[float], list[float]]]:
        """Async wrapper around :meth:`fetch_vectors`."""
        return await asyncio.to_thread(self.fetch_vectors, atom_ids)

    def fetch_atoms_by_ids(self, atom_ids: list[str]) -> list[GraphAtom]:
        """Batch-fetch atom payloads by ``atom_id`` (for lexical-trigger injection)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support fetch_atoms_by_ids"
        )

    def match_lexical_triggers(
        self, text: str, *, max_atoms: int | None = None
    ) -> list[GraphAtom]:
        """Match raw text against the lexical-trigger index and return atoms."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support lexical trigger matching"
        )

    @abc.abstractmethod
    def delete_ontology(
        self,
        iri: str,
        version: str | None = None,
        ontology_hash: str | None = None,
    ) -> None:
        """Delete all indexed atoms for a specific ontology IRI."""

    def reindex_ontology(self, ontology: Ontology) -> int:
        """Replace all atoms for a given ontology and return indexed count."""
        self.delete_ontology(ontology.iri)
        return self.index_ontology(ontology)

    def list_indexed_ontology_iris(self) -> set[str]:
        """Return distinct ``ontology_iri`` values present in the ontology store."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support listing indexed ontology IRIs"
        )

    def prune_orphan_ontology_iris(self, keep_iris: set[str]) -> list[str]:
        """Delete indexed atoms whose ``ontology_iri`` is not in ``keep_iris``.

        An empty ``keep_iris`` is refused rather than treated as "everything is
        an orphan". Pruning exists to follow IRI renames, and no rename makes
        every ontology disappear at once -- an empty catalog means the source of
        truth could not be read, and deleting the whole index on that basis is
        unrecoverable. Callers that genuinely want an empty store should call
        :meth:`wipe_store`.

        Returns the orphan IRIs that were deleted (sorted); empty when the
        prune was refused.
        """
        indexed = self.list_indexed_ontology_iris()
        if not keep_iris:
            if indexed:
                logger.warning(
                    "Refusing to prune %d indexed ontology IRI(s) against an empty "
                    "catalog -- this usually means the triple store could not be "
                    "read. Use wipe_store() to clear the index deliberately.",
                    len(indexed),
                )
            return []
        orphans = sorted(indexed - keep_iris)
        for iri in orphans:
            self.delete_ontology(iri)
        return orphans

    def close(self) -> None:
        """Release any backend connection held by this store.

        Default is a no-op: backends that open no long-lived handle (LanceDB
        connects per call) have nothing to release.
        """
        return None

    async def wipe_store(self) -> None:
        """Drop the currently configured ontology/facts collections or tables.

        Call :meth:`initialize` afterwards to recreate empty schema.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support wiping the current store"
        )

    def apply_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
    ) -> None:
        """Switch the active tenant/project partition when supported."""
        if not self.supports_tenancy_partition():
            raise NotImplementedError(
                f"{type(self).__name__} does not isolate data by tenant/project"
            )
        raise NotImplementedError(f"{type(self).__name__} must implement apply_tenancy")

    def supports_tenancy_partition(self) -> bool:
        """True if tenancy hooks isolate data by tenant/project."""
        return False

    async def clean_tenancy(self, tenant: str, project: str) -> None:
        """Drop or empty vector collections derived from ``tenant`` / ``project``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not isolate vectors by tenant/project"
        )


CORE_VECTOR_NAME = "core"
NEIGHBORHOOD_VECTOR_NAME = "neighborhood"
BM25_VECTOR_NAME = "bm25"
