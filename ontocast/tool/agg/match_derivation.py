"""Derive pairwise predicted↔gt entity matches from global clusters."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np
from rdflib import URIRef

from .match_common import build_entity_match_candidates, greedy_one_to_one
from .match_models import EntityCluster, EntityMatch


def _members_for_graph(
    cluster: EntityCluster, graph_id: str
) -> list[tuple[URIRef, float | None]]:
    return [
        (member.entity, member.similarity)
        for member in cluster.members
        if member.graph_id == graph_id
    ]


def derive_pair_matches(
    clusters: list[EntityCluster],
    predicted_graph_id: str,
    gt_graph_id: str,
    *,
    similarity_threshold: float = 0.0,
) -> list[EntityMatch]:
    """Map global clusters to 1:1 predicted↔gt entity matches for one graph pair."""
    matches: list[EntityMatch] = []
    for cluster in clusters:
        predicted_members = _members_for_graph(cluster, predicted_graph_id)
        gt_members = _members_for_graph(cluster, gt_graph_id)
        if not predicted_members or not gt_members:
            continue

        if len(predicted_members) == 1 and len(gt_members) == 1:
            predicted_entity, _ = predicted_members[0]
            gt_entity, predicted_similarity = gt_members[0]
            gt_similarity = gt_members[0][1]
            score = predicted_similarity or gt_similarity or 1.0
            matches.append(
                EntityMatch(
                    predicted_entity=predicted_entity,
                    gt_entity=gt_entity,
                    similarity=score,
                )
            )
            continue

        candidates: list[EntityMatch] = []
        for (predicted_entity, predicted_score), (gt_entity, gt_score) in product(
            predicted_members, gt_members
        ):
            score = predicted_score or gt_score or 1.0
            if score < similarity_threshold:
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
        matches.extend(greedy_one_to_one(candidates))

    matches.sort(
        key=lambda item: (
            -item.similarity,
            str(item.predicted_entity),
            str(item.gt_entity),
        )
    )
    return matches


def derive_pair_matches_with_embeddings(
    clusters: list[EntityCluster],
    predicted_graph_id: str,
    gt_graph_id: str,
    embeddings: dict[URIRef, np.ndarray],
    *,
    similarity_threshold: float,
    pair_compatible: Callable[[URIRef, URIRef], bool],
) -> list[EntityMatch]:
    """Derive matches using embedding similarity when a cluster has multiple members per side."""
    matches: list[EntityMatch] = []
    for cluster in clusters:
        predicted_entities = [
            member.entity
            for member in cluster.members
            if member.graph_id == predicted_graph_id
        ]
        gt_entities = [
            member.entity
            for member in cluster.members
            if member.graph_id == gt_graph_id
        ]
        if not predicted_entities or not gt_entities:
            continue
        candidates = build_entity_match_candidates(
            predicted_entities,
            gt_entities,
            embeddings,
            similarity_threshold=similarity_threshold,
            pair_compatible=pair_compatible,
        )
        matches.extend(greedy_one_to_one(candidates))
    return matches
