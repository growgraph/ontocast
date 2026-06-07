"""Pydantic request/response models for match API routes."""

from pydantic import BaseModel, ConfigDict, Field

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.match_models import (
    EntityCluster,
    EntityMatch,
    MatchRegime,
)


class TaggedGraphInput(BaseModel):
    id: str
    graph: RDFGraph

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AlignEntitiesRequest(BaseModel):
    graphs: list[TaggedGraphInput]
    regime: MatchRegime = MatchRegime.ONTOLOGY_LOOSE
    similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"


class AlignEntitiesResponse(BaseModel):
    data: dict


class DeriveMatchesRequest(BaseModel):
    clusters: list[EntityCluster]
    predicted_graph_id: str
    gt_graph_id: str
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DeriveMatchesResponse(BaseModel):
    data: dict


class EvaluateMatchRequest(BaseModel):
    predicted_graph: RDFGraph
    gt_graph: RDFGraph
    entity_matches: list[EntityMatch]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EvaluateMatchResponse(BaseModel):
    data: dict
