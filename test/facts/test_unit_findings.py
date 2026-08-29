"""Tests for deterministic literal-object checks on rendered facts graphs."""

import pytest
from rdflib import Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.rdfgraph import RDFGraph, format_quarantine_for_prompt
from ontocast.tool.validate import (
    RDFGraphConnectivityValidator,
    partition_object_property_literal_triples,
)

pytestmark = pytest.mark.unit

QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
CD = Namespace("https://growgraph.dev/facts/")
EX = Namespace("https://example.org/onto#")


def _unit_ontology() -> RDFGraph:
    onto = RDFGraph()
    onto.bind("qudt", QUDT)
    onto.add((QUDT.unit, RDF.type, OWL.ObjectProperty))
    onto.add((QUDT.unit, RDFS.range, QUDT.Unit))
    onto.add((QUDT.numericValue, RDF.type, OWL.DatatypeProperty))
    onto.add((EX.hasStatus, RDFS.range, EX.Status))
    onto.add((EX.hasCode, RDFS.range, XSD.string))
    onto.add((EX.hasNote, RDFS.range, RDFS.Literal))
    onto.add((EX.hasShade, RDFS.range, EX.Shade))
    onto.add((EX.Shade, RDF.type, RDFS.Datatype))
    return onto


def test_object_property_literal_is_quarantined_with_reason() -> None:
    facts = RDFGraph()
    facts.add((CD.v1, QUDT.unit, Literal("meV")))
    facts.add((CD.v1, QUDT.numericValue, Literal("30", datatype=XSD.decimal)))

    clean, rejected = partition_object_property_literal_triples(facts, _unit_ontology())
    assert len(rejected) == 1
    item = rejected[0]
    assert item.object_lexical == "meV"
    assert item.expected_range == str(QUDT.Unit)
    assert item.reason is not None
    assert (CD.v1, QUDT.numericValue, Literal("30", datatype=XSD.decimal)) in clean
    assert (CD.v1, QUDT.unit, Literal("meV")) not in clean


def test_class_range_without_object_property_declaration_also_quarantines() -> None:
    facts = RDFGraph()
    facts.add((CD.x, EX.hasStatus, Literal("active")))
    _, rejected = partition_object_property_literal_triples(facts, _unit_ontology())
    assert len(rejected) == 1
    assert rejected[0].expected_range == str(EX.Status)


def test_literal_compatible_ranges_and_unknown_predicates_pass() -> None:
    facts = RDFGraph()
    facts.add((CD.x, EX.hasCode, Literal("A1")))  # xsd:string range
    facts.add((CD.x, EX.hasNote, Literal("note")))  # rdfs:Literal range
    facts.add((CD.x, EX.hasShade, Literal("dark")))  # declared rdfs:Datatype range
    facts.add((CD.x, EX.undeclared, Literal("free")))  # unknown predicate
    facts.add((CD.x, RDFS.label, Literal("x")))  # annotation vocab

    clean, rejected = partition_object_property_literal_triples(facts, _unit_ontology())
    assert rejected == []
    assert len(clean) == 5


def test_iri_object_on_object_property_is_kept() -> None:
    facts = RDFGraph()
    facts.add((CD.v1, QUDT.unit, UNIT.MilliEV))
    clean, rejected = partition_object_property_literal_triples(facts, _unit_ontology())
    assert rejected == []
    assert (CD.v1, QUDT.unit, UNIT.MilliEV) in clean


def test_quarantine_prompt_includes_expected_range_hint() -> None:
    facts = RDFGraph()
    facts.add((CD.v1, QUDT.unit, Literal("meV")))
    _, rejected = partition_object_property_literal_triples(facts, _unit_ontology())

    turtle = format_quarantine_for_prompt(rejected, LLMGraphFormat.TURTLE)
    assert '"meV"' in turtle
    assert "^^" not in turtle  # no datatype on the plain literal
    assert str(QUDT.Unit) in turtle

    jsonld = format_quarantine_for_prompt(rejected, LLMGraphFormat.JSONLD)
    assert '"@value": "meV"' in jsonld
    assert "@type" not in jsonld
    assert str(QUDT.Unit) in jsonld


def test_validate_predicates_flags_literal_on_class_range() -> None:
    graph = RDFGraph()
    graph.add((EX.hasStatus, RDFS.label, Literal("has status")))
    graph.add((EX.hasStatus, RDFS.range, EX.Status))
    graph.add((CD.x, EX.hasStatus, Literal("active")))

    result = RDFGraphConnectivityValidator(graph).validate_predicates()
    assert not result.domain_range_consistent
    assert any("IRI expected" in v for v in result.domain_range_violations)


def test_validate_predicates_accepts_literal_on_datatype_range() -> None:
    graph = RDFGraph()
    graph.add((EX.hasCode, RDFS.label, Literal("has code")))
    graph.add((EX.hasCode, RDFS.range, XSD.string))
    graph.add((CD.x, EX.hasCode, Literal("A1")))

    result = RDFGraphConnectivityValidator(graph).validate_predicates()
    assert result.domain_range_consistent
