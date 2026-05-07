"""Embedding-based aggregation pipeline for RDF content unit graphs."""

from .aggregate import (
    EmbeddingBasedAggregator,
)
from .matcher import (
    GroundTruthSide,
    MatchRegime,
    TripleSetMatcher,
    TripleSetMatchResult,
)
from .uri_builder import EntityRole, URIBuilder

__all__ = [
    "EmbeddingBasedAggregator",
    "EntityRole",
    "GroundTruthSide",
    "MatchRegime",
    "TripleSetMatchResult",
    "TripleSetMatcher",
    "URIBuilder",
]
