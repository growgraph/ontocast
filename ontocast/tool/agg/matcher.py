"""Triple-set matcher with entity alignment and PR/F1 evaluation."""

from __future__ import annotations

import logging
from enum import StrEnum
from itertools import product

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rdflib import URIRef
from rdflib.term import Node

from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.rdfgraph import RDFGraph

from .aggregate import EmbeddingBasedAggregator
from .clustering import EntityClusterer
from .normalizer import EntityNormalizer, EntityRepresentation

logger = logging.getLogger(__name__)


class MatchRegime(StrEnum):
    ONTOLOGY_LOOSE = "ontology_loose"
    ONTOLOGY_STRICT = "ontology_strict"


class GroundTruthSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class EntityMatch(BaseModel):
    left_entity: URIRef
    right_entity: URIRef
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


class TripleSetMatchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    regime: MatchRegime
    ground_truth_side: GroundTruthSide
    similarity_threshold: float
    entity_matches: list[EntityMatch] = Field(default_factory=list)
    metrics: MatchMetrics


class TripleSetMatcher:
    """Match two RDF triple sets and evaluate with PR/F1."""

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.80,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.normalizer: EntityNormalizer = EntityNormalizer()
        self.clusterer: EntityClusterer = EntityClusterer(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
        )
        # Reuse symbolic compatibility checks to stay aligned with aggregation logic.
        self._compat = EmbeddingBasedAggregator(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            candidate_similarity_threshold=similarity_threshold,
        )

    @staticmethod
    def _extract_entities(graph: RDFGraph) -> list[URIRef]:
        entities: set[URIRef] = set()
        for subject, predicate, obj in graph:
            if isinstance(subject, URIRef):
                entities.add(subject)
            if isinstance(predicate, URIRef):
                entities.add(predicate)
            if isinstance(obj, URIRef):
                entities.add(obj)
        ordered_entities = list(entities)
        ordered_entities.sort(key=lambda entity: str(entity))
        return ordered_entities

    @staticmethod
    def _namespace_set(types: list[URIRef]) -> set[str]:
        namespaces: set[str] = set()
        for entity_type in types:
            namespace, _ = split_namespace_local(str(entity_type))
            if namespace is not None:
                namespaces.add(namespace)
        return namespaces

    def _strict_types_compatible(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> bool:
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        if not left_rep.types or not right_rep.types:
            return True
        left_namespaces = self._namespace_set(left_rep.types)
        right_namespaces = self._namespace_set(right_rep.types)
        if not left_namespaces or not right_namespaces:
            return False
        return bool(left_namespaces & right_namespaces)

    @staticmethod
    def _cosine_similarity(left_vec: np.ndarray, right_vec: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(left_vec, right_vec) / denominator)

    @staticmethod
    def _map_term(term: Node, mapping: dict[URIRef, URIRef]) -> Node:
        if isinstance(term, URIRef):
            return mapping.get(term, term)
        return term

    def _project_triples(
        self, graph: RDFGraph, mapping: dict[URIRef, URIRef]
    ) -> set[tuple[Node, Node, Node]]:
        projected: set[tuple[Node, Node, Node]] = set()
        for subject, predicate, obj in graph:
            projected.add(
                (
                    self._map_term(subject, mapping),
                    self._map_term(predicate, mapping),
                    self._map_term(obj, mapping),
                )
            )
        return projected

    def _candidate_pairs(
        self,
        left_entities: list[URIRef],
        right_entities: list[URIRef],
        embeddings: dict[URIRef, np.ndarray],
        representations: dict[URIRef, EntityRepresentation],
        regime: MatchRegime,
    ) -> list[EntityMatch]:
        candidates: list[EntityMatch] = []
        for left_entity, right_entity in product(left_entities, right_entities):
            left_embedding = embeddings.get(left_entity)
            right_embedding = embeddings.get(right_entity)
            if left_embedding is None or right_embedding is None:
                continue
            score = self._cosine_similarity(left_embedding, right_embedding)
            if score < self.similarity_threshold:
                continue
            if not self._compat._are_roles_compatible(
                left_entity, right_entity, representations
            ):
                continue
            if not self._compat._are_lexical_aliases(
                left_entity, right_entity, representations
            ):
                continue
            if regime == MatchRegime.ONTOLOGY_STRICT:
                if not self._strict_types_compatible(
                    left_entity,
                    right_entity,
                    representations,
                ):
                    continue
            candidates.append(
                EntityMatch(
                    left_entity=left_entity,
                    right_entity=right_entity,
                    similarity=score,
                )
            )

        candidates.sort(
            key=lambda item: (
                -item.similarity,
                str(item.left_entity),
                str(item.right_entity),
            )
        )
        return candidates

    @staticmethod
    def _greedy_one_to_one(candidates: list[EntityMatch]) -> list[EntityMatch]:
        chosen: list[EntityMatch] = []
        used_left: set[URIRef] = set()
        used_right: set[URIRef] = set()
        for candidate in candidates:
            if (
                candidate.left_entity in used_left
                or candidate.right_entity in used_right
            ):
                continue
            chosen.append(candidate)
            used_left.add(candidate.left_entity)
            used_right.add(candidate.right_entity)
        return chosen

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def match(
        self,
        left_graph: RDFGraph,
        right_graph: RDFGraph,
        *,
        regime: MatchRegime = MatchRegime.ONTOLOGY_LOOSE,
        ground_truth_side: GroundTruthSide = GroundTruthSide.RIGHT,
    ) -> TripleSetMatchResult:
        left_entities = self._extract_entities(left_graph)
        right_entities = self._extract_entities(right_graph)

        left_graphs = {entity: left_graph for entity in left_entities}
        right_graphs = {entity: right_graph for entity in right_entities}
        left_representations = self.normalizer.create_representations_batch(
            left_entities, left_graphs
        )
        right_representations = self.normalizer.create_representations_batch(
            right_entities, right_graphs
        )
        representations: dict[URIRef, EntityRepresentation] = {
            **left_representations,
            **right_representations,
        }

        embeddings = self.clusterer.embed_representations(representations)
        candidates = self._candidate_pairs(
            left_entities=left_entities,
            right_entities=right_entities,
            embeddings=embeddings,
            representations=representations,
            regime=regime,
        )
        entity_matches = self._greedy_one_to_one(candidates)

        left_to_right = {
            matched.left_entity: matched.right_entity for matched in entity_matches
        }
        right_to_left = {
            matched.right_entity: matched.left_entity for matched in entity_matches
        }

        if ground_truth_side == GroundTruthSide.RIGHT:
            predicted = self._project_triples(left_graph, left_to_right)
            ground_truth = set(right_graph)
        else:
            predicted = self._project_triples(right_graph, right_to_left)
            ground_truth = set(left_graph)

        true_positives = len(predicted & ground_truth)
        false_positives = len(predicted - ground_truth)
        false_negatives = len(ground_truth - predicted)
        precision = self._safe_divide(true_positives, len(predicted))
        recall = self._safe_divide(true_positives, len(ground_truth))
        f1 = self._safe_divide(2 * precision * recall, precision + recall)

        return TripleSetMatchResult(
            regime=regime,
            ground_truth_side=ground_truth_side,
            similarity_threshold=self.similarity_threshold,
            entity_matches=entity_matches,
            metrics=MatchMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                predicted_count=len(predicted),
                ground_truth_count=len(ground_truth),
            ),
        )
