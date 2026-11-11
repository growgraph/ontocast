"""Nodes standing for a date get the typed ISO value their label names."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDF, Literal, URIRef
from rdflib.namespace import TIME, XSD

from ontocast.config import FactsValidationConfig
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import FactsGateRepairKind
from ontocast.onto.prefixes import PrefixTable
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.facts_gate import run_facts_gate
from ontocast.tool.facts_validation.date_values import (
    add_date_values,
    parse_date_label,
)
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

CD = f"{DEFAULT_IRI}/"
EX = "https://x.org/schema#"

PREFIXES = f"""
@prefix cd:   <{CD}> .
@prefix ex:   <{EX}> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix sosa: <http://www.w3.org/ns/sosa/> .
@prefix time: <http://www.w3.org/2006/time#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
"""


def _graph(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


def _cd(name: str) -> URIRef:
    return URIRef(f"{CD}{name}")


@pytest.mark.parametrize(
    ("label", "allow_year", "expected"),
    [
        ("June 2021", False, ("2021-06", XSD.gYearMonth)),
        ("Sept. 2020", False, ("2020-09", XSD.gYearMonth)),
        ("27 June 2021", False, ("2021-06-27", XSD.date)),
        ("June 27th, 2021", False, ("2021-06-27", XSD.date)),
        ("2021-06", False, ("2021-06", XSD.gYearMonth)),
        ("2021-06-27", False, ("2021-06-27", XSD.date)),
        ("2021-06-27T10:30:00", False, ("2021-06-27T10:30:00", XSD.dateTime)),
        ("2021", True, ("2021", XSD.gYear)),
        ("2021", False, None),
        ("mid-June 2021", False, None),
        ("summer 2021", False, None),
        ("31 June 2021", False, None),
        ("2021-13", False, None),
        ("20210627", False, None),
        ("storm front", False, None),
    ],
)
def test_a_label_parses_only_when_it_is_a_whole_date(
    label: str, allow_year: bool, expected: tuple[str, URIRef] | None
) -> None:
    parsed = parse_date_label(label, allow_year=allow_year)

    if expected is None:
        assert parsed is None
    else:
        assert parsed is not None
        assert (str(parsed), parsed.datatype) == expected


def test_a_bare_month_under_a_temporal_property_gets_its_value() -> None:
    graph = _graph(
        """
        cd:obs a sosa:Observation ; rdfs:label "obs" ; sosa:phenomenonTime cd:june_2021 .
        cd:june_2021 a prov:Entity ; rdfs:label "June 2021" .
        """
    )

    records = add_date_values(graph, RDFGraph(), [CD])

    assert (
        _cd("june_2021"),
        RDF.value,
        Literal("2021-06", datatype=XSD.gYearMonth),
    ) in graph
    assert [(r.kind, r.source) for r in records] == [
        (FactsGateRepairKind.DATE_VALUE_ADDED, str(_cd("june_2021")))
    ]


def test_an_instant_takes_the_owl_time_property() -> None:
    graph = _graph('cd:day a time:Instant ; rdfs:label "27 June 2021" .')

    add_date_values(graph, RDFGraph(), [CD])

    assert (
        _cd("day"),
        TIME.inXSDDate,
        Literal("2021-06-27", datatype=XSD.date),
    ) in graph


def test_a_bare_year_counts_only_in_a_temporal_position() -> None:
    graph = _graph(
        """
        cd:event a ex:Event ; rdfs:label "event" ; ex:heldIn cd:y2021 .
        cd:y2021 a ex:Year ; rdfs:label "2021" .
        cd:model a ex:Model ; rdfs:label "2021" .
        """
    )
    ontology = _graph(
        """
        ex:heldIn rdfs:range ex:Year .
        ex:Year rdfs:subClassOf time:TemporalEntity .
        """
    )

    add_date_values(graph, ontology, [CD])

    assert (_cd("y2021"), RDF.value, Literal("2021", datatype=XSD.gYear)) in graph
    assert (_cd("model"), RDF.value, None) not in graph


def test_a_blank_value_under_a_temporal_property_is_dated() -> None:
    graph = _graph(
        'cd:obs a sosa:Observation ; rdfs:label "obs" ; '
        'sosa:resultTime [ rdfs:label "March 2020" ] .'
    )

    add_date_values(graph, RDFGraph(), [CD])

    assert (None, RDF.value, Literal("2020-03", datatype=XSD.gYearMonth)) in graph


def test_a_dated_or_unparsed_node_is_left_alone() -> None:
    graph = _graph(
        """
        cd:obs a sosa:Observation ; rdfs:label "obs" ;
            sosa:phenomenonTime cd:dated , cd:season .
        cd:dated a time:Instant ; rdfs:label "June 2021" ;
            time:inXSDgYearMonth "2021-06"^^xsd:gYearMonth .
        cd:season a prov:Entity ; rdfs:label "summer 2021" .
        """
    )
    before = len(graph)

    assert add_date_values(graph, RDFGraph(), [CD]) == []
    assert len(graph) == before


def _fake_tools() -> ToolBox:
    facts_validation = FactsValidationConfig()
    return cast(
        ToolBox,
        SimpleNamespace(
            prefix_table=lambda: PrefixTable({}, {}),
            shapes_catalog=SimpleNamespace(graph=lambda: None),
            shapes_prompt_contract=lambda: ("", (), False),
            shapes_chapter_terms=lambda context_terms: frozenset(),
            ontology_manager=SimpleNamespace(catalog_terms=lambda: set()),
            config=SimpleNamespace(
                get_tool_config=lambda: SimpleNamespace(
                    facts_validation=facts_validation
                )
            ),
        ),
    )


def test_gate_dates_bare_date_nodes_and_counts_them() -> None:
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.aggregated_facts = _graph(
        """
        cd:obs a sosa:Observation ; rdfs:label "obs" ; sosa:phenomenonTime cd:june_2021 .
        cd:june_2021 a prov:Entity ; rdfs:label "June 2021" .
        """
    )

    run_facts_gate(state, RDFGraph(), _fake_tools(), merge_repair=False)

    assert state.retrieval_metrics[RetrievalMetric.FACTS_DATE_VALUES_ADDED] == 1
    assert (
        None,
        RDF.value,
        Literal("2021-06", datatype=XSD.gYearMonth),
    ) in state.aggregated_facts
    assert [
        record.kind
        for record in state.facts_gate_repairs
        if record.kind == FactsGateRepairKind.DATE_VALUE_ADDED
    ] == [FactsGateRepairKind.DATE_VALUE_ADDED]
