"""Embedding-based aggregation pipeline for RDF content unit graphs."""

from .aggregate import (
    EmbeddingBasedAggregator,
)
from .entity_aligner import EntityAligner
from .match_derivation import derive_pair_matches
from .match_models import (
    EntityAlignmentResult,
    EntityCluster,
    EntityMatch,
    GraphEntityMember,
    MatchMetrics,
    MatchRegime,
    TaggedGraph,
)
from .triple_evaluator import TripleSetEvaluator
from .uri_builder import EntityRole, URIBuilder

__all__ = [
    "EmbeddingBasedAggregator",
    "EntityAligner",
    "EntityAlignmentResult",
    "EntityCluster",
    "EntityMatch",
    "EntityRole",
    "GraphEntityMember",
    "MatchMetrics",
    "MatchRegime",
    "TaggedGraph",
    "TripleSetEvaluator",
    "URIBuilder",
    "derive_pair_matches",
]
