"""Numeric-mention inventory tests."""

from rdflib import Literal, URIRef
from rdflib.namespace import XSD

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.util.numeric_inventory import (
    extract_numeric_tokens,
    missing_numeric_mentions,
    numeric_literals_in_graph,
)


def test_extract_numeric_tokens_basic_and_ranges() -> None:
    text = "a red shift of ~10-15 meV, up to 96 meV, edge length 8.5 ± 0.5 nm"
    tokens = extract_numeric_tokens(text)
    assert {"10", "15", "96", "8.5", "0.5"} <= tokens


def test_extract_numeric_tokens_drops_year_like_integers() -> None:
    tokens = extract_numeric_tokens("published in 2019, measured 2019.5 K at 77 K")
    assert "77" in tokens
    assert "2019.5" in tokens
    assert "2019" not in tokens


def test_extract_numeric_tokens_canonicalizes() -> None:
    assert extract_numeric_tokens("value 230.0 and 230") == {"230"}


def test_numeric_literals_in_graph_covers_typed_and_label_numbers() -> None:
    graph = RDFGraph()
    subject = URIRef("http://x/s")
    graph.add(
        (
            subject,
            URIRef("http://qudt.org/schema/qudt/numericValue"),
            Literal("12.5", datatype=XSD.decimal),
        )
    )
    graph.add(
        (
            subject,
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
            Literal("range 50 - 300 nm"),
        )
    )
    values = numeric_literals_in_graph(graph)
    assert {"12.5", "50", "300"} <= values


def test_missing_numeric_mentions() -> None:
    graph = RDFGraph()
    graph.add(
        (
            URIRef("http://x/s"),
            URIRef("http://qudt.org/schema/qudt/numericValue"),
            Literal("96", datatype=XSD.decimal),
        )
    )
    missing = missing_numeric_mentions("shifts of 96 meV and 12.5 meV", graph)
    assert missing == ["12.5"]
