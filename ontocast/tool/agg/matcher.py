"""Triple-set matcher with entity alignment and PR/F1 evaluation."""

from __future__ import annotations

import logging
from enum import StrEnum
from itertools import product

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rdflib import RDFS, XSD, URIRef
from rdflib.term import Literal, Node

from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.rdfgraph import RDFGraph

from .aggregate import EmbeddingBasedAggregator
from .clustering import EntityClusterer
from .normalizer import EntityNormalizer, EntityRepresentation

logger = logging.getLogger(__name__)

GENERIC_NAMESPACES = frozenset(
    {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/2001/XMLSchema#",
    }
)


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
    entity_precision: float
    entity_recall: float
    entity_f1: float
    entity_true_positives: int
    entity_false_positives: int
    entity_false_negatives: int
    domain_entity_matches: int


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

    @staticmethod
    def _normalize_literal(node: Node) -> Node:
        if isinstance(node, Literal) and node.datatype == XSD.string:
            return Literal(str(node))
        return node

    @staticmethod
    def _normalize_triple(triple: tuple[Node, Node, Node]) -> tuple[Node, Node, Node]:
        subject, predicate, obj = triple
        return (
            TripleSetMatcher._normalize_literal(subject),
            TripleSetMatcher._normalize_literal(predicate),
            TripleSetMatcher._normalize_literal(obj),
        )

    @staticmethod
    def _is_informative_triple(triple: tuple[Node, Node, Node]) -> bool:
        _, predicate, _ = triple
        return predicate != RDFS.label

    def _prepare_metric_triples(
        self, triples: set[tuple[Node, Node, Node]]
    ) -> set[tuple[Node, Node, Node]]:
        return {
            self._normalize_triple(triple)
            for triple in triples
            if self._is_informative_triple(triple)
        }

    @staticmethod
    def _is_domain_entity(entity: URIRef) -> bool:
        namespace, _ = split_namespace_local(str(entity))
        return namespace is not None and namespace not in GENERIC_NAMESPACES

    @staticmethod
    def _count_domain_entity_matches(entity_matches: list[EntityMatch]) -> int:
        return sum(
            1
            for matched in entity_matches
            if TripleSetMatcher._is_domain_entity(matched.left_entity)
            and TripleSetMatcher._is_domain_entity(matched.right_entity)
        )

    def _compute_prf(
        self,
        true_positives: int,
        predicted_count: int,
        ground_truth_count: int,
    ) -> tuple[float, float, float]:
        precision = self._safe_divide(true_positives, predicted_count)
        recall = self._safe_divide(true_positives, ground_truth_count)
        f1 = self._safe_divide(2 * precision * recall, precision + recall)
        return precision, recall, f1

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
            raw_predicted = self._project_triples(left_graph, left_to_right)
            raw_ground_truth = set(right_graph)
        else:
            raw_predicted = self._project_triples(right_graph, right_to_left)
            raw_ground_truth = set(left_graph)

        predicted = self._prepare_metric_triples(raw_predicted)
        ground_truth = self._prepare_metric_triples(raw_ground_truth)

        true_positives = len(predicted & ground_truth)
        false_positives = len(predicted - ground_truth)
        false_negatives = len(ground_truth - predicted)
        precision, recall, f1 = self._compute_prf(
            true_positives,
            len(predicted),
            len(ground_truth),
        )

        matched_left = {matched.left_entity for matched in entity_matches}
        matched_right = {matched.right_entity for matched in entity_matches}
        entity_true_positives = len(entity_matches)
        entity_false_positives = len(left_entities) - len(matched_left)
        entity_false_negatives = len(right_entities) - len(matched_right)
        entity_precision, entity_recall, entity_f1 = self._compute_prf(
            entity_true_positives,
            len(left_entities),
            len(right_entities),
        )
        domain_entity_matches = self._count_domain_entity_matches(entity_matches)

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
                entity_precision=entity_precision,
                entity_recall=entity_recall,
                entity_f1=entity_f1,
                entity_true_positives=entity_true_positives,
                entity_false_positives=entity_false_positives,
                entity_false_negatives=entity_false_negatives,
                domain_entity_matches=domain_entity_matches,
            ),
        )
