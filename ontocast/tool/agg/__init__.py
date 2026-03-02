"""Embedding-based aggregation pipeline for RDF content unit graphs."""

from .aggregate import (
    EmbeddingBasedAggregator,
    aggregate_chunk_graphs,
    aggregate_content_unit_graphs,
)
from .uri_builder import EntityRole, URIBuilder

__all__ = [
    "EmbeddingBasedAggregator",
    "EntityRole",
    "URIBuilder",
    "aggregate_content_unit_graphs",
    "aggregate_chunk_graphs",
]
