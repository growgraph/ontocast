"""Precision/recall helpers and entity extraction for graph evaluation."""

import pytest
from rdflib import OWL, RDF, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.match_common import compute_prf, extract_instance_entities

pytestmark = pytest.mark.unit


def test_prf_is_undefined_without_predictions() -> None:
    """An empty prediction has no precision; zero would bias a macro mean."""
    precision, recall, f1 = compute_prf(0, 0, 4)
    assert precision is None
    assert recall == 0.0
    assert f1 is None


def test_prf_is_undefined_without_ground_truth() -> None:
    precision, recall, f1 = compute_prf(0, 3, 0)
    assert precision == 0.0
    assert recall is None
    assert f1 is None


def test_prf_with_no_overlap_is_zero() -> None:
    assert compute_prf(0, 2, 2) == (0.0, 0.0, 0.0)


def test_prf_on_partial_overlap() -> None:
    precision, recall, f1 = compute_prf(1, 2, 4)
    assert (precision, recall) == (0.5, 0.25)
    assert f1 == pytest.approx(1 / 3)


def test_instance_entities_exclude_predicates_and_ontology_terms() -> None:
    a = URIRef("https://example.org/a")
    b = URIRef("https://example.org/b")
    person = URIRef("https://example.org/onto#Person")
    knows = URIRef("https://example.org/onto#knows")
    graph = RDFGraph()
    graph.add((person, RDF.type, OWL.Class))
    graph.add((knows, RDF.type, OWL.ObjectProperty))
    graph.add((a, RDF.type, person))
    graph.add((a, knows, b))

    assert extract_instance_entities(graph) == [a, b]
