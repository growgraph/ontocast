from copy import deepcopy

from rdflib import Graph, Literal, URIRef

from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.rdfgraph import (
    RDFGraph,
    finalize_llm_graph,
    format_quarantine_for_prompt,
)


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


def test_from_turtle_coerces_year_date_literal_to_xsd_gyear() -> None:
    ttl = """
    @prefix ex: <https://example.com/ns#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    ex:item ex:established "2001"^^xsd:date .
    """

    graph = RDFGraph._from_turtle_str(ttl)

    assert len(graph) == 1
    triple = (
        URIRef("https://example.com/ns#item"),
        URIRef("https://example.com/ns#established"),
        Literal("2001", datatype=URIRef("http://www.w3.org/2001/XMLSchema#gYear")),
    )
    assert triple in graph


def test_from_turtle_drops_invalid_date_datatype_for_non_iso_dates() -> None:
    ttl = """
    @prefix ex: <https://example.com/ns#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    ex:item ex:established "2026/05/06"^^xsd:date .
    """

    graph = RDFGraph._from_turtle_str(ttl)

    assert len(graph) == 1
    triple = (
        URIRef("https://example.com/ns#item"),
        URIRef("https://example.com/ns#established"),
        Literal("2026/05/06"),
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


def test_from_turtle_sanitizes_prefix_without_terminal_delimiter() -> None:
    ttl = """
    @prefix cd: <https://growgraph.dev/facts> .
    @prefix ex: <https://example.com/ns#> .
    cd:imprisonment1 ex:relatedTo cd:imprisonment2 .
    """

    graph = RDFGraph._from_turtle_str(ttl)

    assert (
        URIRef("https://growgraph.dev/facts/imprisonment1"),
        URIRef("https://example.com/ns#relatedTo"),
        URIRef("https://growgraph.dev/facts/imprisonment2"),
    ) in graph


def test_serialize_canonical_turtle_normalizes_namespace_delimiters() -> None:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix ex: <https://example.org/ns#> .
        @prefix cd: <https://growgraph.dev/facts> .
        cd:item ex:relatedTo cd:target .
        """
    )

    turtle = graph.serialize_canonical_turtle()

    assert "https://growgraph.dev/factsitem" not in turtle
    assert "https://growgraph.dev/facts//item" in turtle


def test_deepcopy_preserves_rdfgraph_type() -> None:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix ex: <https://example.com/ns#> .
        ex:a ex:rel ex:b .
        """
    )

    copied = deepcopy(graph)

    assert isinstance(copied, RDFGraph)
    assert len(copied) == len(graph)


def test_from_turtle_coerces_nan_decimal_to_plain_literal() -> None:
    ttl = """
    @prefix ex: <https://example.com/ns#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    ex:item ex:value "NaN"^^xsd:decimal .
    """

    graph = RDFGraph._from_turtle_str(ttl)

    assert len(graph) == 1
    triple = (
        URIRef("https://example.com/ns#item"),
        URIRef("https://example.com/ns#value"),
        Literal("NaN"),
    )
    assert triple in graph


def test_partition_invalid_decimal_with_full_uri_datatype() -> None:
    graph = RDFGraph()
    graph.add(
        (
            URIRef("https://example.com/ns#item"),
            URIRef("https://example.com/ns#value"),
            Literal(
                "10-15",
                datatype=URIRef("http://www.w3.org/2001/XMLSchema#decimal"),
            ),
        )
    )
    graph.add(
        (
            URIRef("https://example.com/ns#item"),
            URIRef("https://example.com/ns#amount"),
            Literal(
                "42.5",
                datatype=URIRef("http://www.w3.org/2001/XMLSchema#decimal"),
            ),
        )
    )

    clean, rejected = RDFGraph.partition_invalid_typed_literals(graph)

    assert len(clean) == 1
    assert len(rejected) == 1
    assert rejected[0].object_lexical == "10-15"
    assert rejected[0].datatype.endswith("#decimal")


def test_finalize_llm_graph_jsonld_range_decimal() -> None:
    graph = RDFGraph._from_jsonld_obj(
        {
            "@context": {
                "ex": "https://example.com/ns#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
            },
            "@graph": [
                {
                    "@id": "ex:item",
                    "ex:value": {"@value": "10-15", "@type": "xsd:decimal"},
                }
            ],
        }
    )

    clean, rejected = finalize_llm_graph(graph)

    assert len(clean) == 1
    assert len(rejected) == 0
    obj = next(clean.objects())
    assert isinstance(obj, Literal)
    assert obj.datatype is None
    assert str(obj) == "10-15"


def test_format_quarantine_for_prompt_turtle() -> None:
    graph = RDFGraph()
    graph.add(
        (
            URIRef("https://example.com/ns#item"),
            URIRef("https://example.com/ns#value"),
            Literal(
                "10-15",
                datatype=URIRef("http://www.w3.org/2001/XMLSchema#decimal"),
            ),
        )
    )
    _, rejected = RDFGraph.partition_invalid_typed_literals(graph)

    formatted = format_quarantine_for_prompt(rejected, LLMGraphFormat.TURTLE)

    assert '"10-15"^^<http://www.w3.org/2001/XMLSchema#decimal>' in formatted
    assert "ex:item" in formatted or "https://example.com/ns#item" in formatted


def test_content_unit_sanitize_coerces_plain_graph() -> None:
    from ontocast.onto.content_unit import ContentUnit

    plain = Graph()
    plain.add(
        (
            URIRef("https://example.org/s"),
            URIRef("https://example.org/p"),
            Literal("o"),
        )
    )
    unit = ContentUnit(
        text="x",
        index=0,
        doc_iri=URIRef("https://example.org/doc/1"),
    )
    unit.graph = plain  # type: ignore[assignment]

    unit.sanitize()

    assert isinstance(unit.graph, RDFGraph)


def test_from_jsonld_coerces_invalid_decimal_before_parse() -> None:
    graph = RDFGraph._from_jsonld_obj(
        {
            "@context": {
                "ex": "https://example.com/ns#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
            },
            "@graph": [
                {
                    "@id": "ex:item",
                    "ex:value": {"@value": "10-15", "@type": "xsd:decimal"},
                }
            ],
        }
    )

    assert len(graph) == 1
    obj = next(graph.objects())
    assert isinstance(obj, Literal)
    assert obj.datatype is None
    assert str(obj) == "10-15"


def test_from_jsonld_coerces_nan_decimal_in_nquads_path() -> None:
    graph = RDFGraph._from_jsonld_obj(
        {
            "@context": {
                "ex": "https://example.com/ns#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
            },
            "@graph": [
                {
                    "@id": "ex:item",
                    "ex:value": {"@value": "NaN", "@type": "xsd:decimal"},
                }
            ],
        }
    )

    assert len(graph) == 1
    obj = next(graph.objects())
    assert isinstance(obj, Literal)
    assert obj.datatype is None
    assert str(obj) == "NaN"


def test_from_turtle_recovers_unknown_qudt_prefix() -> None:
    from ontocast.onto.constants import WELL_KNOWN_PREFIXES

    ttl = """
    @prefix ex: <https://example.com/ns#> .
    ex:item qudt:quantity "5" .
    """
    graph = RDFGraph._from_turtle_str(ttl)

    assert len(graph) == 1
    predicate = next(graph.predicates())
    assert str(predicate).startswith(WELL_KNOWN_PREFIXES["qudt"])


def test_coerce_invalid_nquads_typed_literals_strips_bad_decimal() -> None:
    nquads = (
        "<https://example.com/ns#item> <https://example.com/ns#value> "
        '"10-15"^^<http://www.w3.org/2001/XMLSchema#decimal> .\n'
    )
    coerced = RDFGraph._coerce_invalid_nquads_typed_literals(nquads)
    assert "^^<http://www.w3.org/2001/XMLSchema#decimal>" not in coerced
    assert '"10-15"' in coerced
