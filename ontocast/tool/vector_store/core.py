"""Core contracts and models for ontology vector storage."""

from __future__ import annotations

import abc
from datetime import datetime, timezone

from pydantic import Field

from ontocast.onto.model import BasePydanticModel
from ontocast.onto.ontology import Ontology
from ontocast.tool.onto import Tool


class OntologyAtom(BasePydanticModel):
    """Embedding-ready ontology neighborhood patch."""

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
    node_uri: str = Field(description="Focal node IRI of this neighborhood patch.")
    turtle: str = Field(description="Turtle serialization of the patch graph.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Atom creation timestamp (UTC).",
    )
    score: float | None = Field(
        default=None,
        description="Optional similarity score populated by vector search.",
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
        self, query: str, top_k: int = 10, filter_iri: str | None = None
    ) -> list[OntologyAtom]:
        """Search ontology patches by query text."""

    @abc.abstractmethod
    def delete_ontology(self, iri: str) -> None:
        """Delete all indexed atoms for a specific ontology IRI."""
