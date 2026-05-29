"""Shared models for RDF graph matching and evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import core_schema
from rdflib import URIRef

from ontocast.onto.rdfgraph import RDFGraph


def coerce_uri_ref(value: Any) -> URIRef:
    if isinstance(value, URIRef):
        return value
    if isinstance(value, str):
        return URIRef(value)
    raise TypeError(f"Expected URIRef or str, got {type(value).__name__}")


def as_uri_ref(value: URIRef | str) -> URIRef:
    """Normalize a graph or match term to URIRef for rdflib-safe equality."""
    return coerce_uri_ref(value)


class _RdfUriRefAnnotation:
    """Pydantic schema that keeps URIRef instances (avoids str coercion)."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            coerce_uri_ref,
            serialization=core_schema.to_string_ser_schema(),
        )


RdfUriRef = Annotated[URIRef, _RdfUriRefAnnotation()]


class MatchRegime(StrEnum):
    ONTOLOGY_LOOSE = "ontology_loose"
    ONTOLOGY_STRICT = "ontology_strict"


class EntityMatch(BaseModel):
    predicted_entity: RdfUriRef
    gt_entity: RdfUriRef
    similarity: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
    fact_precision: float
    fact_recall: float
    fact_f1: float
    fact_true_positives: int
    fact_false_positives: int
    fact_false_negatives: int
    fact_predicted_count: int
    fact_ground_truth_count: int


class TaggedGraph(BaseModel):
    id: str
    graph: RDFGraph

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GraphEntityMember(BaseModel):
    graph_id: str
    entity: RdfUriRef
    similarity: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
