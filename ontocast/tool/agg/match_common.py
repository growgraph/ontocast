"""Stateless helpers for RDF graph matching and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from rdflib import RDFS, XSD, URIRef
from rdflib.term import Literal, Node

from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.rdfgraph import RDFGraph

from .match_models import EntityMatch

GENERIC_NAMESPACES = frozenset(
    {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/2001/XMLSchema#",
    }
)


@dataclass(frozen=True)
class GraphEntityRef:
    graph_id: str
    entity: URIRef


def extract_entities(graph: RDFGraph) -> list[URIRef]:
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


def map_term(term: Node, mapping: dict[URIRef, URIRef]) -> Node:
    if isinstance(term, URIRef):
        return mapping.get(term, term)
    return term


def normalize_literal(node: Node) -> Node:
    if isinstance(node, Literal) and node.datatype == XSD.string:
        return Literal(str(node))
    return node


def normalize_triple(triple: tuple[Node, Node, Node]) -> tuple[Node, Node, Node]:
    subject, predicate, obj = triple
    return (
        normalize_literal(subject),
        normalize_literal(predicate),
        normalize_literal(obj),
    )


def is_informative_triple(triple: tuple[Node, Node, Node]) -> bool:
    _, predicate, _ = triple
    return predicate != RDFS.label


def prepare_metric_triples(
    triples: set[tuple[Node, Node, Node]],
) -> set[tuple[Node, Node, Node]]:
    return {
        normalize_triple(triple) for triple in triples if is_informative_triple(triple)
    }


def is_domain_entity(entity: URIRef) -> bool:
    namespace, _ = split_namespace_local(str(entity))
    return namespace is not None and namespace not in GENERIC_NAMESPACES


def count_domain_entity_matches(entity_matches: list[EntityMatch]) -> int:
    return sum(
        1
        for matched in entity_matches
        if is_domain_entity(matched.predicted_entity)
        and is_domain_entity(matched.gt_entity)
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_prf(
    true_positives: int,
    predicted_count: int,
    ground_truth_count: int,
) -> tuple[float, float, float]:
    precision = safe_divide(true_positives, predicted_count)
    recall = safe_divide(true_positives, ground_truth_count)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def project_triples(
    graph: RDFGraph, mapping: dict[URIRef, URIRef]
) -> set[tuple[Node, Node, Node]]:
    projected: set[tuple[Node, Node, Node]] = set()
    for subject, predicate, obj in graph:
        projected.add(
            (
                map_term(subject, mapping),
                map_term(predicate, mapping),
                map_term(obj, mapping),
            )
        )
    return projected


def cosine_similarity(left_vec: np.ndarray, right_vec: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_vec, right_vec) / denominator)


def greedy_one_to_one(candidates: list[EntityMatch]) -> list[EntityMatch]:
    chosen: list[EntityMatch] = []
    used_predicted: set[URIRef] = set()
    used_gt: set[URIRef] = set()
    for candidate in candidates:
        if (
            candidate.predicted_entity in used_predicted
            or candidate.gt_entity in used_gt
        ):
            continue
        chosen.append(candidate)
        used_predicted.add(candidate.predicted_entity)
        used_gt.add(candidate.gt_entity)
    return chosen


def build_entity_match_candidates(
    predicted_entities: list[URIRef],
    gt_entities: list[URIRef],
    embeddings: dict[URIRef, np.ndarray],
    *,
    similarity_threshold: float,
    pair_compatible,
) -> list[EntityMatch]:
    candidates: list[EntityMatch] = []
    for predicted_entity, gt_entity in product(predicted_entities, gt_entities):
        predicted_embedding = embeddings.get(predicted_entity)
        gt_embedding = embeddings.get(gt_entity)
        if predicted_embedding is None or gt_embedding is None:
            continue
        score = cosine_similarity(predicted_embedding, gt_embedding)
        if score < similarity_threshold:
            continue
        if not pair_compatible(predicted_entity, gt_entity):
            continue
        candidates.append(
            EntityMatch(
                predicted_entity=predicted_entity,
                gt_entity=gt_entity,
                similarity=score,
            )
        )
    candidates.sort(
        key=lambda item: (
            -item.similarity,
            str(item.predicted_entity),
            str(item.gt_entity),
        )
    )
    return candidates
