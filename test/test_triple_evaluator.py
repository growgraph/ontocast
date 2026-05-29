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


def test_type_only_overlap_has_triple_metrics_but_no_fact_tp() -> None:
    person_type = URIRef("https://types/Person")
    entity = URIRef("https://example.org/Alpha")
    graph = RDFGraph()
    graph.add((entity, RDF.type, person_type))

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=graph,
        gt_graph=graph,
        entity_matches=[
            EntityMatch(predicted_entity=entity, gt_entity=entity, similarity=1.0),
            EntityMatch(
                predicted_entity=person_type, gt_entity=person_type, similarity=1.0
            ),
            EntityMatch(predicted_entity=RDF.type, gt_entity=RDF.type, similarity=1.0),
        ],
    )
    assert metrics.true_positives == 1
    assert metrics.precision == 1.0
    assert metrics.fact_true_positives == 0
    assert metrics.fact_predicted_count == 0
    assert metrics.fact_ground_truth_count == 0


def test_relational_triple_perfect_fact_metrics() -> None:
    book = URIRef("http://text2kg.bench/prisoners_of_the_sun")
    person = URIRef("http://text2kg.bench/captain_haddock")
    characters = URIRef("https://bench.example/relations/P674")

    graph = RDFGraph()
    graph.add((book, characters, person))

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=graph,
        gt_graph=graph,
        entity_matches=[
            EntityMatch(predicted_entity=book, gt_entity=book, similarity=1.0),
            EntityMatch(predicted_entity=person, gt_entity=person, similarity=1.0),
            EntityMatch(
                predicted_entity=characters, gt_entity=characters, similarity=1.0
            ),
        ],
    )
    assert metrics.fact_true_positives == 1
    assert metrics.fact_precision == 1.0
    assert metrics.fact_recall == 1.0
    assert metrics.fact_f1 == 1.0


def test_subclass_axiom_excluded_from_facts() -> None:
    class_a = URIRef("https://ontology.example/ClassA")
    class_b = URIRef("https://ontology.example/ClassB")
    graph = RDFGraph()
    graph.add((class_a, RDFS.subClassOf, class_b))

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=graph,
        gt_graph=graph,
        entity_matches=[
            EntityMatch(predicted_entity=class_a, gt_entity=class_a, similarity=1.0),
            EntityMatch(predicted_entity=class_b, gt_entity=class_b, similarity=1.0),
            EntityMatch(
                predicted_entity=RDFS.subClassOf,
                gt_entity=RDFS.subClassOf,
                similarity=1.0,
            ),
        ],
    )
    assert metrics.true_positives == 1
    assert metrics.fact_true_positives == 0


def test_extra_relation_lowers_fact_precision() -> None:
    book = URIRef("http://text2kg.bench/book")
    person = URIRef("http://text2kg.bench/person")
    extra = URIRef("http://text2kg.bench/extra")
    p674 = URIRef("https://bench.example/relations/P674")
    p50 = URIRef("https://bench.example/relations/P50")

    gt_graph = RDFGraph()
    gt_graph.add((book, p674, person))

    predicted_graph = RDFGraph()
    predicted_graph.add((book, p674, person))
    predicted_graph.add((book, p50, extra))

    entity_matches = [
        EntityMatch(predicted_entity=book, gt_entity=book, similarity=1.0),
        EntityMatch(predicted_entity=person, gt_entity=person, similarity=1.0),
        EntityMatch(predicted_entity=extra, gt_entity=extra, similarity=1.0),
        EntityMatch(predicted_entity=p674, gt_entity=p674, similarity=1.0),
        EntityMatch(predicted_entity=p50, gt_entity=p50, similarity=1.0),
    ]

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=predicted_graph,
        gt_graph=gt_graph,
        entity_matches=entity_matches,
    )
    assert metrics.fact_true_positives == 1
    assert metrics.fact_predicted_count == 2
    assert metrics.fact_precision == 0.5
    assert metrics.fact_recall == 1.0


def test_wrong_type_excluded_from_facts_but_counts_in_triple_metrics() -> None:
    book = URIRef("http://text2kg.bench/book")
    person = URIRef("http://text2kg.bench/person")
    correct_type = URIRef("https://ontology.example/concepts/Q95074")
    wrong_type = URIRef("https://ontology.example/concepts/Q5")
    p674 = URIRef("https://bench.example/relations/P674")

    gt_graph = RDFGraph()
    gt_graph.add((book, p674, person))
    gt_graph.add((person, RDF.type, correct_type))

    predicted_graph = RDFGraph()
    predicted_graph.add((book, p674, person))
    predicted_graph.add((person, RDF.type, wrong_type))

    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=predicted_graph,
        gt_graph=gt_graph,
        entity_matches=[
            EntityMatch(predicted_entity=book, gt_entity=book, similarity=1.0),
            EntityMatch(predicted_entity=person, gt_entity=person, similarity=1.0),
            EntityMatch(
                predicted_entity=correct_type, gt_entity=correct_type, similarity=1.0
            ),
            EntityMatch(
                predicted_entity=wrong_type, gt_entity=wrong_type, similarity=1.0
            ),
            EntityMatch(predicted_entity=p674, gt_entity=p674, similarity=1.0),
            EntityMatch(predicted_entity=RDF.type, gt_entity=RDF.type, similarity=1.0),
        ],
    )
    assert metrics.fact_true_positives == 1
    assert metrics.fact_f1 == 1.0
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
