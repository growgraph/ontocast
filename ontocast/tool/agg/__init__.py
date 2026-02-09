"""Embedding-based aggregation pipeline for RDF chunk graphs."""

from .aggregate import (
    EmbeddingBasedAggregator,
    aggregate_chunk_graphs,
)
from .uri_builder import EntityRole, URIBuilder

__all__ = [
    "EmbeddingBasedAggregator",
    "EntityRole",
    "URIBuilder",
    "aggregate_chunk_graphs",
]
