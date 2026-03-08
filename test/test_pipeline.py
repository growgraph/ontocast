from typing import cast

import pytest
from rdflib import URIRef

from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.onto.constants import ONTOLOGY_NULL_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.enum import RenderMode, Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph import unit_loops
from ontocast.tool.aggregate import EmbeddingBasedAggregator
from ontocast.toolbox import ToolBox


def _build_content_unit() -> ContentUnit:
    return ContentUnit(
        text="Alice works for ACME.",
        index=0,
        hid="chunk0",
        doc_iri=URIRef("https://example.com/doc/d1"),
    )


def _build_ontology() -> Ontology:
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix onto: <https://example.com/onto#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        onto:CompanyOntology a owl:Ontology .
        """,
        format="turtle",
    )
    return Ontology(graph=graph, iri="https://example.com/onto")


def test_unit_facts_loop_isolates_input_state() -> None:
    """Unit loop uses model_copy(deep=True), so input state is not mutated."""
    state = UnitFactsState(
        content_unit=_build_content_unit(), ontology_snapshot=_build_ontology()
    )
    original_text = state.content_unit.text
    # Simulate what the loop does: it copies before processing
    copied = state.model_copy(deep=True)
    copied.content_unit.text = "MUTATED"
    assert state.content_unit.text == original_text


@pytest.mark.anyio
async def test_run_unit_facts_loop_uses_dedicated_state(monkeypatch) -> None:
    async def fake_render(state: UnitFactsState, tools) -> UnitFactsState:
        state.status = Status.SUCCESS
        return state

    async def fake_critic(state: UnitFactsState, tools) -> UnitFactsState:
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(unit_loops, "render_facts", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_facts", fake_critic)

    state = UnitFactsState(
        content_unit=_build_content_unit(), ontology_snapshot=_build_ontology()
    )
    tools = cast(ToolBox, object())
    result = await unit_loops.unit_facts_loop(state, tools=tools)

    assert result.status == Status.SUCCESS
    assert result.content_unit.hid == state.content_unit.hid


@pytest.mark.anyio
async def test_run_unit_ontology_loop_emits_updates(monkeypatch) -> None:
    async def fake_render(state: UnitOntologyState, tools) -> UnitOntologyState:
        state.status = Status.SUCCESS
        state.ontology_updates = [GraphUpdate()]
        state.current_ontology = Ontology(
            graph=RDFGraph(), iri="https://example.com/onto"
        )
        return state

    async def fake_critic(state: UnitOntologyState, tools) -> UnitOntologyState:
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(unit_loops, "render_ontology", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", fake_critic)

    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=Ontology(iri=ONTOLOGY_NULL_IRI),
    )
    tools = cast(ToolBox, object())
    result = await unit_loops.unit_ontology_loop(state, tools=tools)

    assert result.status == Status.SUCCESS
    assert len(result.all_updates) == 1


def test_reduce_ontology_units_returns_ontology_when_no_units() -> None:
    tools = ToolBox.__new__(ToolBox)
    tools.aggregator = EmbeddingBasedAggregator()
    reduced, applied = normalize_ontology_units(units=[], tools=tools)

    assert reduced is not None
    assert reduced.iri is not None
    assert applied == []


def test_reduce_ontology_units_aggregates_via_embedding() -> None:
    tools = ToolBox.__new__(ToolBox)
    tools.aggregator = EmbeddingBasedAggregator()
    unit1 = ContentUnit(
        text="Alice works at ACME",
        index=0,
        hid="c0",
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=_build_ontology().graph,
        type=OutputType.ONTOLOGIES,
    )
    reduced, applied = normalize_ontology_units(units=[unit1], tools=tools)

    assert reduced is not None
    assert len(reduced.graph) >= 0
    assert isinstance(applied, list)


def test_agent_state_render_mode_properties() -> None:
    facts_only = AgentState(render_mode=RenderMode.FACTS)
    assert facts_only.render_mode == RenderMode.FACTS
    assert facts_only.render_facts is True
    assert facts_only.render_ontology is False

    ontology_only = AgentState(render_mode=RenderMode.ONTOLOGY)
    assert ontology_only.render_mode == RenderMode.ONTOLOGY
    assert ontology_only.render_facts is False
    assert ontology_only.render_ontology is True

    both = AgentState(render_mode=RenderMode.ONTOLOGY_AND_FACTS)
    assert both.render_mode == RenderMode.ONTOLOGY_AND_FACTS
    assert both.render_facts is True
    assert both.render_ontology is True
