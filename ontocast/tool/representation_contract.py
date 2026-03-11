"""Shared structural contracts for embedding-ready representations."""

from __future__ import annotations

from typing import Protocol

from rdflib import URIRef


class EmbeddingRepresentation(Protocol):
    """Structural contract for embedding-oriented text representations."""

    @property
    def iri(self) -> str | URIRef:
        """IRI of the focal entity represented by this object."""
        ...

    @property
    def ontology_iri(self) -> str | None:
        """Source ontology IRI when known in this representation context."""
        ...

    @property
    def core_representation(self) -> str: ...

    @property
    def neighborhood_representation(self) -> str: ...

    @property
    def representation(self) -> str:
        """Combined representation string view."""
        ...


def combine_embedding_text(value: EmbeddingRepresentation) -> str:
    """Return a canonical combined view for embedding-oriented text objects."""
    if value.neighborhood_representation:
        return f"{value.core_representation}. {value.neighborhood_representation}"
    return value.core_representation
