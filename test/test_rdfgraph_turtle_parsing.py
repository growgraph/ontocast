from rdflib import Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph


def test_from_turtle_coerces_invalid_integer_typed_literal() -> None:
    ttl = """
    @prefix ex: <https://example.com/ns#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    ex:item ex:value "10-15"^^xsd:integer .
    """

    graph = RDFGraph._from_turtle_str(ttl)

    assert len(graph) == 1
    triple = (
        URIRef("https://example.com/ns#item"),
        URIRef("https://example.com/ns#value"),
        Literal("10-15"),
    )
    assert triple in graph


def test_from_turtle_removes_invisible_unicode_chars() -> None:
    ttl = """
    @prefix ex: <https://example.com/ns#> .
    ex:item ex:value\u200b ex:target .
    """

    graph = RDFGraph._from_turtle_str(ttl)

    assert len(graph) == 1
    assert (
        URIRef("https://example.com/ns#item"),
        URIRef("https://example.com/ns#value"),
        URIRef("https://example.com/ns#target"),
    ) in graph


def test_from_turtle_drops_line_missing_object_after_predicate() -> None:
    ttl = """
    @prefix ex: <https://example.com/ns#> .
    ex:broken ex:predicate .
    ex:ok ex:predicate ex:value .
    """

    graph = RDFGraph._from_turtle_str(ttl)

    assert len(graph) == 1
    assert (
        URIRef("https://example.com/ns#ok"),
        URIRef("https://example.com/ns#predicate"),
        URIRef("https://example.com/ns#value"),
    ) in graph
