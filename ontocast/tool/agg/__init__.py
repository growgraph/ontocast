"""Embedding-based aggregation pipeline for RDF content unit graphs."""

from .aggregate import (
    EmbeddingBasedAggregator,
)
from .uri_builder import EntityRole, URIBuilder

__all__ = [
    "EmbeddingBasedAggregator",
    "EntityRole",
    "URIBuilder",
]
