"""Bare-string compact IRIs on IRI-valued positions are coerced back to IRIs.

A JSON-LD bare string (``"qudt:unit": "unit:MilliEV"``) parses as a literal, so
the value node loses its unit link and reads as missing the property.
"""

import pytest
from rdflib import Literal, URIRef

from ontocast.onto.model import FactsUnitFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import BudgetTracker
from ontocast.tool.facts_validation import repair_compact_iri_literals
from ontocast.tool.llm import use_budget_tracker

pytestmark = pytest.mark.unit

QUDT = "http://qudt.org/schema/qudt/"
UNIT = "http://qudt.org/vocab/unit/"
QQVAL = "https://growgraph.dev/ontologies/qqval#"
FACTS = "https://growgraph.dev/facts/"

_PREFIXES = f"""
@prefix cd: <{FACTS}> .
@prefix qudt: <{QUDT}> .
@prefix unit: <{UNIT}> .
@prefix qqval: <{QQVAL}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
"""


def _ontology() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(
        data=_PREFIXES
        + """
        qqval:epistemicQualifier a owl:ObjectProperty .
        qqval:note a owl:DatatypeProperty ; rdfs:range xsd:string .
        """,
        format="turtle",
    )
    return graph


def _facts(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=_PREFIXES + body, format="turtle")
    return graph


def _subject(name: str) -> URIRef:
    return URIRef(f"{FACTS}{name}")


def test_known_term_on_a_rangeless_predicate_is_coerced() -> None:
    # qudt:unit carries no range in the context, as in vendored QUDT subsets;
    # the unit individual is a known term, which is enough.
    graph = _facts('cd:v qudt:unit "unit:MilliEV" .')
    rewritten, records = repair_compact_iri_literals(
        graph, _ontology(), {f"{UNIT}MilliEV"}
    )
    assert rewritten == 1
    assert (_subject("v"), URIRef(f"{QUDT}unit"), URIRef(f"{UNIT}MilliEV")) in graph
    assert [(r.kind, r.source, r.target) for r in records] == [
        (FactsUnitFindingKind.COMPACT_IRI_LITERAL, "unit:MilliEV", f"{UNIT}MilliEV")
    ]


def test_object_property_takes_an_unknown_term() -> None:
    graph = _facts('cd:v qqval:epistemicQualifier "qqval:Approximate" .')
    rewritten, _ = repair_compact_iri_literals(graph, _ontology(), set())
    assert rewritten == 1
    assert (
        _subject("v"),
        URIRef(f"{QQVAL}epistemicQualifier"),
        URIRef(f"{QQVAL}Approximate"),
    ) in graph


def test_xsd_string_literal_is_plain_enough() -> None:
    graph = _facts('cd:v qudt:unit "unit:MilliEV"^^xsd:string .')
    rewritten, _ = repair_compact_iri_literals(graph, _ontology(), {f"{UNIT}MilliEV"})
    assert rewritten == 1


@pytest.mark.parametrize(
    "body",
    [
        # Prose with a colon is never a compact IRI.
        'cd:v qudt:unit "time: 10 minutes" .',
        # A prefix nothing binds is not guessed.
        'cd:v qudt:unit "nobody:MilliEV" .',
        # Language-tagged and non-string typed literals are values.
        'cd:v qudt:unit "unit:MilliEV"@en .',
        'cd:v qudt:unit "unit:MilliEV"^^xsd:token .',
        # A datatype property keeps its string, even one shaped like a CURIE.
        'cd:v qqval:note "qqval:Approximate" .',
        # rdf:type literals belong to repair_literal_type_objects.
        'cd:v a "qqval:Approximate" .',
    ],
)
def test_values_that_are_not_iri_references_are_left_alone(body: str) -> None:
    graph = _facts(body)
    before = set(graph)
    rewritten, records = repair_compact_iri_literals(
        graph, _ontology(), {f"{UNIT}MilliEV"}
    )
    assert rewritten == 0
    assert not records
    assert set(graph) == before


def test_unknown_term_on_a_rangeless_predicate_is_left_alone() -> None:
    graph = _facts('cd:v qudt:unit "unit:NotInCatalog" .')
    rewritten, _ = repair_compact_iri_literals(graph, _ontology(), set())
    assert rewritten == 0
    assert (_subject("v"), URIRef(f"{QUDT}unit"), Literal("unit:NotInCatalog")) in graph


def test_known_prefix_context_resolves_an_undeclared_prefix() -> None:
    graph = RDFGraph()
    graph.add((_subject("v"), URIRef(f"{QUDT}unit"), Literal("unit:MilliEV")))
    RDFGraph.set_known_prefixes({"unit": UNIT})
    try:
        rewritten, _ = repair_compact_iri_literals(
            graph, _ontology(), {f"{UNIT}MilliEV"}
        )
    finally:
        RDFGraph.set_known_prefixes(None)
    assert rewritten == 1


def test_rewrites_are_counted() -> None:
    graph = _facts(
        'cd:v qudt:unit "unit:MilliEV" ; qqval:epistemicQualifier "qqval:Exact" .'
    )
    tracker = BudgetTracker()
    with use_budget_tracker(tracker):
        repair_compact_iri_literals(graph, _ontology(), {f"{UNIT}MilliEV"})
    assert tracker.counters == {"repair/compact_iri_literal": 2}
    assert not [o for o in graph.objects(None, None) if isinstance(o, Literal)]
