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


def test_evaluate_projects_distinct_iris_with_str_match_fields() -> None:
    """Simulate legacy str entity fields; evaluator must still project to URIRef triples."""
    author_pred = URIRef("https://growgraph.dev/doc/anders")
    book_pred = URIRef("https://growgraph.dev/doc/book")
    author_gt = URIRef("http://text2kg.bench/anders_jacobsson")
    book_gt = URIRef("http://text2kg.bench/berts_book")
    author_prop = URIRef("https://bench.example/relations/P50")

    predicted_graph = RDFGraph()
    gt_graph = RDFGraph()
    predicted_graph.add((book_pred, author_prop, author_pred))
    gt_graph.add((book_gt, author_prop, author_gt))

    entity_matches = [
        EntityMatch.model_construct(
            predicted_entity=str(book_pred),
            gt_entity=str(book_gt),
            similarity=1.0,
        ),
        EntityMatch.model_construct(
            predicted_entity=str(author_pred),
            gt_entity=str(author_gt),
            similarity=1.0,
        ),
        EntityMatch.model_construct(
            predicted_entity=str(author_prop),
            gt_entity=str(author_prop),
            similarity=1.0,
        ),
    ]

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=predicted_graph,
        gt_graph=gt_graph,
        entity_matches=entity_matches,
    )
    assert metrics.true_positives >= 1
    assert metrics.f1 > 0.0


def test_evaluate_entity_metrics_use_set_membership() -> None:
    predicted_graph = RDFGraph()
    gt_graph = RDFGraph()
    mapped = URIRef("https://example.org/mapped")
    unmatched_pred = URIRef("https://example.org/extra_pred")
    unmatched_gt = URIRef("https://example.org/extra_gt")
    predicate = URIRef("https://example.org/p")

    predicted_graph.add((mapped, predicate, unmatched_pred))
    predicted_graph.add((unmatched_pred, predicate, mapped))
    gt_graph.add((mapped, predicate, mapped))
    gt_graph.add((unmatched_gt, predicate, mapped))

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=predicted_graph,
        gt_graph=gt_graph,
        entity_matches=[
            EntityMatch.model_construct(
                predicted_entity=str(mapped),
                gt_entity=str(mapped),
                similarity=1.0,
            ),
            EntityMatch.model_construct(
                predicted_entity=str(predicate),
                gt_entity=str(predicate),
                similarity=1.0,
            ),
        ],
    )
    assert metrics.entity_false_positives >= 1
    assert metrics.entity_false_negatives >= 1


def test_evaluate_json_validated_matches_produce_true_positives() -> None:
    author_pred = URIRef("https://predicted.example/author")
    book_pred = URIRef("https://predicted.example/book")
    author_gt = URIRef("https://gt.example/author")
    book_gt = URIRef("https://gt.example/book")
    author_prop = URIRef("https://relations.example/P50")

    predicted_graph = RDFGraph()
    gt_graph = RDFGraph()
    predicted_graph.add((book_pred, author_prop, author_pred))
    gt_graph.add((book_gt, author_prop, author_gt))

    entity_matches = [
        EntityMatch.model_validate(
            {
                "predicted_entity": str(book_pred),
                "gt_entity": str(book_gt),
                "similarity": 1.0,
            }
        ),
        EntityMatch.model_validate(
            {
                "predicted_entity": str(author_pred),
                "gt_entity": str(author_gt),
                "similarity": 1.0,
            }
        ),
        EntityMatch.model_validate(
            {
                "predicted_entity": str(author_prop),
                "gt_entity": str(author_prop),
                "similarity": 1.0,
            }
        ),
    ]

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=predicted_graph,
        gt_graph=gt_graph,
        entity_matches=entity_matches,
    )
    assert metrics.true_positives >= 1
