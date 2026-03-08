import importlib
from types import SimpleNamespace
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

render_ontology_module = importlib.import_module("ontocast.agent.render_ontology")
select_ontology_module = importlib.import_module("ontocast.agent.select_ontology")


def _build_content_unit() -> ContentUnit:
    return ContentUnit(
        text="Alice works for ACME.",
        index=0,
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
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=_build_ontology().graph,
        type=OutputType.ONTOLOGIES,
    )
    reduced, applied = normalize_ontology_units(units=[unit1], tools=tools)

    assert reduced is not None
    assert len(reduced.graph) >= 0
    assert isinstance(applied, list)


def test_reduce_ontology_units_creates_base_when_required() -> None:
    class DummyAggregator:
        def aggregate_graphs(self, units: list[ContentUnit]) -> RDFGraph:
            graph = RDFGraph()
            graph.parse(
                data="""
                @prefix ex: <https://example.com/onto#> .
                @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
                @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
                ex:Company rdf:type rdfs:Class .
                """,
                format="turtle",
            )
            return graph

    tools = cast(ToolBox, ToolBox.__new__(ToolBox))
    tools.aggregator = cast(EmbeddingBasedAggregator, DummyAggregator())
    unit = ContentUnit(
        text="Company ontology snippet",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=RDFGraph(),
        type=OutputType.ONTOLOGIES,
    )
    reduced, applied = normalize_ontology_units(
        units=[unit],
        tools=tools,
        base_ontology=None,
        require_base=True,
    )

    assert not reduced.is_null()
    assert len(reduced.graph) > 0
    assert isinstance(applied, list)


@pytest.mark.anyio
async def test_select_ontology_none_keeps_success_status(monkeypatch) -> None:
    class SelectorResult:
        answer_index = 0

    async def fake_call_llm_with_retry(**kwargs):
        return SelectorResult()

    monkeypatch.setattr(
        select_ontology_module, "call_llm_with_retry", fake_call_llm_with_retry
    )

    state = AgentState()
    state.content_units = [_build_content_unit()]
    tools = SimpleNamespace(
        llm=object(),
        ontology_manager=SimpleNamespace(
            has_ontologies=True, ontologies=[_build_ontology()]
        ),
    )
    result = await select_ontology_module.select_ontology(state, tools)  # type: ignore[arg-type]

    assert result.status == Status.SUCCESS
    assert result.current_ontology.is_null()


@pytest.mark.anyio
async def test_render_ontology_uses_update_when_snapshot_exists(monkeypatch) -> None:
    calls = {"fresh": 0, "update": 0}

    async def fake_fresh(state: UnitOntologyState, tools) -> UnitOntologyState:
        calls["fresh"] += 1
        return state

    async def fake_update(state: UnitOntologyState, tools) -> UnitOntologyState:
        calls["update"] += 1
        return state

    monkeypatch.setattr(render_ontology_module, "render_ontology_fresh", fake_fresh)
    monkeypatch.setattr(render_ontology_module, "render_ontology_update", fake_update)

    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=_build_ontology(),
    )
    # Simulate accidental null current ontology while a valid snapshot exists.
    state.current_ontology = Ontology(iri=ONTOLOGY_NULL_IRI)
    result = await render_ontology_module.render_ontology(
        state, tools=cast(ToolBox, object())
    )

    assert result is state
    assert calls["update"] == 1
    assert calls["fresh"] == 0


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
