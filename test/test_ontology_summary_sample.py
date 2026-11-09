"""The startup sample of a seed ontology is the same on every load."""

from datetime import UTC, datetime

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.toolbox import sample_ontology_graph
from test.ontology_helpers import stamped_ontology

pytestmark = pytest.mark.unit


def _seed() -> RDFGraph:
    graph = RDFGraph()
    header = URIRef("https://example.org/seed")
    graph.add((header, RDF.type, OWL.Ontology))
    graph.add((header, RDFS.label, Literal("Seed")))
    term = URIRef("https://example.org/seed#Club")
    graph.add((term, RDF.type, OWL.Class))
    graph.add((term, RDFS.label, Literal("club")))
    return graph


def test_two_loads_at_different_times_give_the_same_sample() -> None:
    """The header is stamped when stored; a sample carrying the stamp would
    make the summary prompt, and its cache key, differ on every run."""
    first = stamped_ontology(_seed(), now=datetime(2026, 1, 1, tzinfo=UTC))
    second = stamped_ontology(_seed(), now=datetime(2026, 6, 1, tzinfo=UTC))

    a = sample_ontology_graph(first.graph).serialize_canonical_turtle()
    b = sample_ontology_graph(second.graph).serialize_canonical_turtle()

    assert a == b
    assert "2026-01-01" not in a
    assert "club" in a
