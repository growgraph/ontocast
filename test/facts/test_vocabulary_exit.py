"""Severity decides what a vocabulary finding does when a unit leaves its loop.

Every ``unknown_term`` finding of error severity (mandatory) drops the
statements using its term, and the record carries the terms it suggested,
never applied. A finding of warning severity leaves the graph as it is. Under
a ``strict`` vocabulary a term outside the catalog and the well-known
vocabularies is an error; under ``open`` it is no unit finding at all. A term
misspelled in the catalog's namespace is never merged into the term it likely
meant. Test data is neutral.
"""

from __future__ import annotations

from typing import Literal

import pytest
from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.model import (
    FactsGateRepairKind,
    FactsUnitFinding,
    FactsUnitFindingKind,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.stategraph.unit_exit import exit_unit
from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator
from ontocast.tool.facts_validation import ValidationPolicy
from ontocast.tool.facts_validation.statements import StatementContract
from ontocast.tool.facts_validation.terms import CatalogRoles, value_class_base
from ontocast.tool.facts_validation.unit_findings import collect_unit_findings

pytestmark = pytest.mark.unit

EX = "https://ex.org/v#"
OTHER = "https://other.org/v#"
CD = DEFAULT_IRI
_PREFIXES = f"""
@prefix ex:    <{EX}> .
@prefix other: <{OTHER}> .
@prefix cd:    <{CD}> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:   <http://www.w3.org/2002/07/owl#> .
"""
CATALOG = """
ex:Item a owl:Class .
ex:partOf a owl:ObjectProperty .
"""


def _graph(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=_PREFIXES + body, format="turtle")
    return graph


def _contract(catalog: RDFGraph) -> StatementContract:
    return CatalogRoles.of([catalog]).statement_contract(value_class_base(None, {}))


def _findings(
    graph: RDFGraph, vocabulary: Literal["strict", "open"]
) -> list[FactsUnitFinding]:
    catalog = _graph(CATALOG)
    return collect_unit_findings(
        graph=graph,
        ontology_graph=catalog,
        quarantined=[],
        extraction_text="",
        fact_namespaces=[CD],
        coverage_limit=0,
        policy=ValidationPolicy(vocabulary=vocabulary),
        contract=_contract(catalog),
    )


def _unknown(findings: list[FactsUnitFinding]) -> list[FactsUnitFinding]:
    return [f for f in findings if f.kind == FactsUnitFindingKind.UNKNOWN_TERM]


_FOREIGN = """
cd:a a ex:Item ; rdfs:label "first item" ; other:relatesTo cd:b .
cd:b a ex:Item ; rdfs:label "second item" .
"""


def test_a_term_outside_a_strict_vocabulary_is_dropped_at_exit() -> None:
    graph = _graph(_FOREIGN)
    findings = _findings(graph, "strict")

    (finding,) = _unknown(findings)
    assert finding.mandatory
    result = exit_unit(graph, findings, [], _contract(_graph(CATALOG)))

    assert not list(graph.triples((None, URIRef(OTHER + "relatesTo"), None)))
    assert [record.kind for record in result.records] == [
        FactsGateRepairKind.UNKNOWN_TERM_DROPPED
    ]


def test_a_term_outside_an_open_vocabulary_stays() -> None:
    graph = _graph(_FOREIGN)
    before = set(graph)
    findings = _findings(graph, "open")

    assert not _unknown(findings)
    assert exit_unit(graph, findings, [], _contract(_graph(CATALOG))).records == []
    assert set(graph) == before


def test_a_warning_vocabulary_finding_leaves_the_graph_as_it_is() -> None:
    graph = _graph(_FOREIGN)
    before = set(graph)
    warning = FactsUnitFinding(
        kind=FactsUnitFindingKind.UNKNOWN_TERM,
        mandatory=False,
        message="advisory",
        predicate=OTHER + "relatesTo",
        facet="non_catalog_strict",
    )

    assert exit_unit(graph, [warning], [], _contract(_graph(CATALOG))).records == []
    assert set(graph) == before


def test_a_class_minted_in_the_facts_namespace_is_dropped_at_exit() -> None:
    graph = _graph('cd:a a cd:Widget ; rdfs:label "first item" .')
    findings = _findings(graph, "open")

    (finding,) = _unknown(findings)
    assert finding.facet == "fact_namespace"
    exit_unit(graph, findings, [], _contract(_graph(CATALOG)))

    assert not list(graph.triples((None, None, URIRef(CD + "Widget"))))
    assert list(graph.triples((URIRef(CD + "a"), None, None)))


def test_a_misspelled_catalog_term_is_reported_with_its_suggestion_and_dropped() -> (
    None
):
    graph = _graph(
        'cd:a a ex:Item ; rdfs:label "first item" ; ex:partof cd:b .\n'
        'cd:b a ex:Item ; rdfs:label "second item" .'
    )
    findings = _findings(graph, "open")

    (finding,) = _unknown(findings)
    assert finding.facet == "closed_namespace_missing"
    assert finding.suggestions[0] == EX + "partOf"
    result = exit_unit(graph, findings, [], _contract(_graph(CATALOG)))

    (record,) = result.records
    assert record.source == EX + "partof"
    assert record.suggestions[0] == EX + "partOf"
    # Reported, never applied.
    assert not list(graph.triples((None, URIRef(EX + "partOf"), None)))
    assert not list(graph.triples((None, URIRef(EX + "partof"), None)))


def test_a_misspelled_catalog_term_is_never_merged_into_the_term_it_meant(
    monkeypatch,
) -> None:
    def unit(index: int, body: str) -> ContentUnit:
        return ContentUnit(
            text=f"unit {index}",
            index=index,
            doc_iri=URIRef("https://ex.org/doc/1"),
            graph=_graph(body),
            type=OutputType.FACTS,
        )

    units = [
        unit(0, 'cd:a a ex:Item ; rdfs:label "first item" ; ex:partOf cd:b .'),
        unit(1, 'cd:c a ex:Item ; rdfs:label "third item" ; ex:partof cd:d .'),
    ]
    aggregator = EmbeddingBasedAggregator()
    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        lambda representations: ([list(representations)], {}),
    )
    result = aggregator.postprocess_facts_units(units, _graph(CATALOG))

    predicates = set(result.graph.predicates())
    assert URIRef(EX + "partOf") in predicates
    assert URIRef(EX + "partof") in predicates
