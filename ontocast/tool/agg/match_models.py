"""Shared models for RDF graph matching and evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rdflib import URIRef

from ontocast.onto.rdfgraph import RDFGraph


def coerce_uri_ref(value: Any) -> URIRef:
    if isinstance(value, URIRef):
        return value
    if isinstance(value, str):
        return URIRef(value)
    raise TypeError(f"Expected URIRef or str, got {type(value).__name__}")


class MatchRegime(StrEnum):
    ONTOLOGY_LOOSE = "ontology_loose"
    ONTOLOGY_STRICT = "ontology_strict"


class EntityMatch(BaseModel):
    predicted_entity: URIRef
    gt_entity: URIRef
    similarity: float

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("predicted_entity", "gt_entity", mode="before")
    @classmethod
    def _coerce_entity_uri(cls, value: Any) -> URIRef:
        return coerce_uri_ref(value)


class MatchMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    predicted_count: int
    ground_truth_count: int
    entity_precision: float
    entity_recall: float
    entity_f1: float
    entity_true_positives: int
    entity_false_positives: int
    entity_false_negatives: int
    domain_entity_matches: int


class TaggedGraph(BaseModel):
    id: str
    graph: RDFGraph

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GraphEntityMember(BaseModel):
    graph_id: str
    entity: URIRef
    similarity: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("entity", mode="before")
    @classmethod
    def _coerce_entity_uri(cls, value: Any) -> URIRef:
        return coerce_uri_ref(value)


class EntityCluster(BaseModel):
    members: list[GraphEntityMember] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EntityAlignmentResult(BaseModel):
    regime: MatchRegime
    similarity_threshold: float
    entity_count: int
    cluster_count: int
    clusters: list[EntityCluster] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
