"""Graph aggregation for OntoCast.

This module re-exports the embedding-based aggregator as the main aggregation
implementation. Use EmbeddingBasedAggregator for aggregating and disambiguating
RDF graphs from multiple content units.
"""

from ontocast.tool.agg.aggregate import (
    EmbeddingBasedAggregator,
    aggregate_chunk_graphs,
    aggregate_content_unit_graphs,
)

__all__ = [
    "EmbeddingBasedAggregator",
    "aggregate_content_unit_graphs",
    "aggregate_chunk_graphs",
]
