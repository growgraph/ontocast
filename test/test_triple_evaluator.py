from __future__ import annotations

from rdflib import RDF, RDFS, XSD, Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.match_models import EntityMatch
from ontocast.tool.agg.triple_evaluator import TripleSetEvaluator


def test_evaluate_perfect_alignment() -> None:
    graph = RDFGraph()
    entity = URIRef("https://example.org/Alpha")
    target = URIRef("https://example.org/Beta")
    graph.add((entity, RDF.type, URIRef("https://types/Person")))
    graph.add((entity, URIRef("https://pred/relatedTo"), target))

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=graph,
        gt_graph=graph,
        entity_matches=[
            EntityMatch(predicted_entity=entity, gt_entity=entity, similarity=1.0),
            EntityMatch(
                predicted_entity=URIRef("https://pred/relatedTo"),
                gt_entity=URIRef("https://pred/relatedTo"),
                similarity=1.0,
            ),
            EntityMatch(
                predicted_entity=URIRef("https://types/Person"),
                gt_entity=URIRef("https://types/Person"),
                similarity=1.0,
            ),
            EntityMatch(predicted_entity=target, gt_entity=target, similarity=1.0),
            EntityMatch(predicted_entity=RDF.type, gt_entity=RDF.type, similarity=1.0),
        ],
    )
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_label_triples_excluded() -> None:
    predicted_graph = RDFGraph()
    gt_graph = RDFGraph()
    entity = URIRef("https://example.org/Alpha")
    predicted_graph.add((entity, RDFS.label, Literal("Alpha")))
    gt_graph.add((entity, RDFS.label, Literal("Alpha")))

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=predicted_graph,
        gt_graph=gt_graph,
        entity_matches=[
            EntityMatch(predicted_entity=entity, gt_entity=entity, similarity=1.0),
            EntityMatch(
                predicted_entity=RDFS.label, gt_entity=RDFS.label, similarity=1.0
            ),
        ],
    )
    assert metrics.ground_truth_count == 0
    assert metrics.predicted_count == 0
    assert metrics.true_positives == 0


def test_xsd_string_literal_normalized() -> None:
    predicate = URIRef("https://pred.example/name")
    predicted_graph = RDFGraph()
    gt_graph = RDFGraph()
    entity = URIRef("https://example.org/entity")
    predicted_graph.add(
        (entity, predicate, Literal("Alan Wright", datatype=XSD.string))
    )
    gt_graph.add((entity, predicate, Literal("Alan Wright")))

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=predicted_graph,
        gt_graph=gt_graph,
        entity_matches=[
            EntityMatch(predicted_entity=entity, gt_entity=entity, similarity=1.0),
            EntityMatch(
                predicted_entity=predicate, gt_entity=predicate, similarity=1.0
            ),
        ],
    )
    assert metrics.true_positives == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
