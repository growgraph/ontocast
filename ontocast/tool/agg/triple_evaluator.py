"""Evaluate predicted vs ground-truth RDF graphs given entity alignments."""

from __future__ import annotations

from ontocast.onto.rdfgraph import RDFGraph

from .match_common import (
    compute_prf,
    count_domain_entity_matches,
    extract_entities,
    prepare_metric_triples,
    project_triples,
)
from .match_models import EntityMatch, MatchMetrics


class TripleSetEvaluator:
    """Compute PR/F1 metrics for aligned predicted and ground-truth graphs."""

    def evaluate(
        self,
        predicted_graph: RDFGraph,
        gt_graph: RDFGraph,
        entity_matches: list[EntityMatch],
    ) -> MatchMetrics:
        predicted_to_gt = {
            matched.predicted_entity: matched.gt_entity for matched in entity_matches
        }

        raw_predicted = project_triples(predicted_graph, predicted_to_gt)
        raw_ground_truth = set(gt_graph)

        predicted = prepare_metric_triples(raw_predicted)
        ground_truth = prepare_metric_triples(raw_ground_truth)

        true_positives = len(predicted & ground_truth)
        false_positives = len(predicted - ground_truth)
        false_negatives = len(ground_truth - predicted)
        precision, recall, f1 = compute_prf(
            true_positives,
            len(predicted),
            len(ground_truth),
        )

        predicted_entities = extract_entities(predicted_graph)
        gt_entities = extract_entities(gt_graph)
        matched_predicted = {matched.predicted_entity for matched in entity_matches}
        matched_gt = {matched.gt_entity for matched in entity_matches}
        entity_true_positives = len(entity_matches)
        entity_false_positives = len(predicted_entities) - len(matched_predicted)
        entity_false_negatives = len(gt_entities) - len(matched_gt)
        entity_precision, entity_recall, entity_f1 = compute_prf(
            entity_true_positives,
            len(predicted_entities),
            len(gt_entities),
        )
        domain_entity_matches = count_domain_entity_matches(entity_matches)

        return MatchMetrics(
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
        )
