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


def test_from_turtle_preserves_xsd_datetime_literal() -> None:
    """xsd:dateTime must not be corrupted by xsd:date literal coercion."""
    ttl = """
    @prefix doc: <https://growgraph.dev/doc/b9355d00de72/> .
    @prefix prov: <http://www.w3.org/ns/prov#> .
    @prefix schema: <https://schema.org/> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    doc:b9355d00de72 a prov:Entity,
            schema:text ;
        prov:generatedAtTime "2026-06-02T15:04:28.776738+00:00"^^xsd:dateTime ;
        schema:identifier "b9355d00de72"^^xsd:string ;
        schema:position 0 .
    """

    graph = RDFGraph._from_turtle_str(ttl)

    assert len(graph) == 5
    generated_at = Literal(
        "2026-06-02T15:04:28.776738+00:00",
        datatype=URIRef("http://www.w3.org/2001/XMLSchema#dateTime"),
    )
    assert (
        URIRef("https://growgraph.dev/doc/b9355d00de72/b9355d00de72"),
        URIRef("http://www.w3.org/ns/prov#generatedAtTime"),
        generated_at,
    ) in graph


def test_coerce_invalid_date_typed_literals_does_not_match_datetime() -> None:
    turtle = 'prov:generatedAtTime "2026-06-02T15:04:28.776738+00:00"^^xsd:dateTime ;'
    assert RDFGraph._coerce_invalid_date_typed_literals(turtle) == turtle


def test_coerce_invalid_nquads_date_typed_literals_does_not_match_datetime() -> None:
    nquads = (
        '"2026-06-02T15:04:28.776738+00:00"^^'
        "<http://www.w3.org/2001/XMLSchema#dateTime> "
    )
    assert RDFGraph._coerce_invalid_nquads_typed_literals(nquads) == nquads


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
    object.__setattr__(unit, "graph", plain)

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


def test_strip_sparql_update_wrapper_extracts_delete_data_triples() -> None:
    from ontocast.onto.rdfgraph import strip_sparql_update_wrapper

    ttl = """
    @prefix cd: <https://example.com/facts/> .
    @prefix ex: <https://example.com/ns#> .

    cd:entity_1 a ex:SomeClass ;
        ex:label "foo" .

    DELETE DATA {
      cd:old_entity a ex:OtherClass .
    }
    """
    stripped = strip_sparql_update_wrapper(ttl)
    assert "DELETE DATA" not in stripped
    assert "cd:entity_1 a ex:SomeClass" in stripped
    assert "cd:old_entity a ex:OtherClass" in stripped


def test_from_turtle_parses_mixed_turtle_and_delete_data() -> None:
    ttl = """
    @prefix cd: <https://example.com/facts/> .
    @prefix ex: <https://example.com/ns#> .

    cd:entity_1 a ex:SomeClass ;
        ex:label "foo" .

    DELETE DATA {
      cd:old_entity a ex:OtherClass .
    }
    """
    graph = RDFGraph._from_turtle_str(ttl)

    assert len(graph) == 3


def test_normalize_turtle_input_converts_sparql_prefix() -> None:
    ttl = "PREFIX ex: <https://example.com/ns#>\nex:a ex:b ex:c ."
    normalized = RDFGraph._normalize_turtle_input(ttl)
    assert "@prefix ex: <https://example.com/ns#> ." in normalized
    assert "PREFIX ex:" not in normalized


def test_model_validate_uses_explicit_llm_graph_format_context() -> None:
    from ontocast.onto.model import GraphUpdateRenderReport

    payload = {
        "graph_update": {
            "triple_operations": [
                {
                    "type": "insert",
                    "graph": {
                        "@context": {"ex": "http://example.org/"},
                        "@graph": [{"@id": "ex:item", "@type": "ex:Thing"}],
                    },
                }
            ],
        },
    }
    report = GraphUpdateRenderReport.model_validate(
        payload,
        context={"llm_graph_format": LLMGraphFormat.JSONLD},
    )
    assert len(report.graph_update.triple_operations[0].graph) >= 1
