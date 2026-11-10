"""Schema IRIs written as JSON-LD bare strings are coerced back to IRIs.

Under the JSON-LD wire ``"rdfs:subClassOf": "ex:Bar"`` parses as a literal, so
a class loses its parent. The ontology render repairs it on the schema
predicates whose object is an IRI by construction.
"""

import json

import pytest
from rdflib import OWL, RDFS, Literal, URIRef

from ontocast.agent.render_ontology import repair_schema_iri_literals
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import BudgetTracker
from ontocast.tool.llm import use_budget_tracker

pytestmark = pytest.mark.unit

EX = "https://example.org/onto#"


def _jsonld_graph(node: dict) -> RDFGraph:
    doc = {
        "@context": {
            "ex": EX,
            "rdfs": str(RDFS),
            "owl": str(OWL),
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        },
        "@graph": [node],
    }
    graph = RDFGraph()
    graph.parse(data=json.dumps(doc), format="json-ld")
    return graph


def test_bare_string_schema_objects_become_iris() -> None:
    graph = _jsonld_graph(
        {
            "@id": "ex:hasPart",
            "@type": "owl:ObjectProperty",
            "rdfs:domain": "ex:Whole",
            "rdfs:range": "ex:Part",
            "rdfs:subPropertyOf": "ex:relatedTo",
            "rdfs:label": "has part",
        }
    )
    tracker = BudgetTracker()
    with use_budget_tracker(tracker):
        repaired, rejected = repair_schema_iri_literals(graph)

    subject = URIRef(EX + "hasPart")
    assert rejected == []
    assert repaired.value(subject, RDFS.domain) == URIRef(EX + "Whole")
    assert repaired.value(subject, RDFS.range) == URIRef(EX + "Part")
    assert repaired.value(subject, RDFS.subPropertyOf) == URIRef(EX + "relatedTo")
    # An annotation stays a literal, even though it is a string.
    assert repaired.value(subject, RDFS.label) == Literal("has part")
    assert tracker.counters["repair/compact_iri_literal"] == 3


def test_parent_class_and_datatype_range_become_iris() -> None:
    graph = _jsonld_graph(
        {"@id": "ex:Bar", "@type": "owl:Class", "rdfs:subClassOf": "ex:Foo"}
    )
    graph += _jsonld_graph(
        {"@id": "ex:name", "@type": "owl:DatatypeProperty", "rdfs:range": "xsd:string"}
    )

    repaired, _ = repair_schema_iri_literals(graph)

    assert repaired.value(URIRef(EX + "Bar"), RDFS.subClassOf) == URIRef(EX + "Foo")
    assert repaired.value(URIRef(EX + "name"), RDFS.range) == URIRef(
        "http://www.w3.org/2001/XMLSchema#string"
    )


def test_a_string_with_an_unbound_prefix_is_left_alone() -> None:
    graph = _jsonld_graph(
        {"@id": "ex:Bar", "@type": "owl:Class", "rdfs:subClassOf": "nope:Foo"}
    )

    repaired, _ = repair_schema_iri_literals(graph)

    assert repaired.value(URIRef(EX + "Bar"), RDFS.subClassOf) == Literal("nope:Foo")
