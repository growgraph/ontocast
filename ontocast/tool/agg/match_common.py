"""Stateless helpers for RDF graph matching and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from rdflib import RDF, RDFS, XSD, URIRef
from rdflib.term import Literal, Node

from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.rdfgraph import RDFGraph

from .match_models import EntityMatch, as_uri_ref

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
        mapped = mapping.get(term)
        if mapped is None:
            return term
        return as_uri_ref(mapped) if isinstance(mapped, str) else mapped
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


SCHEMA_PREDICATES = frozenset({RDF.type, RDFS.subClassOf, RDFS.label, RDFS.comment})


def _is_generic_vocab_entity(entity: URIRef) -> bool:
    namespace, _ = split_namespace_local(str(entity))
    return namespace is not None and namespace in GENERIC_NAMESPACES


def collect_ontology_entities(
    triples: set[tuple[Node, Node, Node]],
) -> frozenset[URIRef]:
    """URIRefs that are class/concept/schema nodes (s/o), not relation predicates."""
    ontology_entities: set[URIRef] = set()
    for subject, predicate, obj in triples:
        if predicate == RDF.type and isinstance(obj, URIRef):
            ontology_entities.add(obj)
        elif predicate == RDFS.subClassOf:
            if isinstance(subject, URIRef):
                ontology_entities.add(subject)
            if isinstance(obj, URIRef):
                ontology_entities.add(obj)
        if isinstance(subject, URIRef) and _is_generic_vocab_entity(subject):
            ontology_entities.add(subject)
        if isinstance(obj, URIRef) and _is_generic_vocab_entity(obj):
            ontology_entities.add(obj)
    return frozenset(ontology_entities)


def is_fact_triple(
    triple: tuple[Node, Node, Node],
    ontology_entities: frozenset[URIRef],
) -> bool:
    if not is_informative_triple(triple):
        return False
    subject, predicate, obj = triple
    if predicate in SCHEMA_PREDICATES:
        return False
    if isinstance(subject, URIRef) and subject in ontology_entities:
        return False
    if isinstance(obj, URIRef) and obj in ontology_entities:
        return False
    return True


def prepare_fact_triples(
    triples: set[tuple[Node, Node, Node]],
    ontology_entities: frozenset[URIRef],
) -> set[tuple[Node, Node, Node]]:
    return {
        normalize_triple(triple)
        for triple in triples
        if is_fact_triple(triple, ontology_entities)
    }


def is_domain_entity(entity: URIRef) -> bool:
    namespace, _ = split_namespace_local(str(entity))
    return namespace is not None and namespace not in GENERIC_NAMESPACES


def count_domain_entity_matches(entity_matches: list[EntityMatch]) -> int:
    return sum(
        1
        for matched in entity_matches
        if is_domain_entity(as_uri_ref(matched.predicted_entity))
        and is_domain_entity(as_uri_ref(matched.gt_entity))
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
        predicted_entity = as_uri_ref(candidate.predicted_entity)
        gt_entity = as_uri_ref(candidate.gt_entity)
        if predicted_entity in used_predicted or gt_entity in used_gt:
            continue
        chosen.append(candidate)
        used_predicted.add(predicted_entity)
        used_gt.add(gt_entity)
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
        predicted_key = as_uri_ref(predicted_entity)
        gt_key = as_uri_ref(gt_entity)
        predicted_embedding = embeddings.get(predicted_key)
        gt_embedding = embeddings.get(gt_key)
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
