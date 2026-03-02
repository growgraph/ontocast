from typing import cast

import pytest
from rdflib import URIRef

from ontocast.agent import unit_loops
from ontocast.agent.reduce_results import reduce_ontology_updates
from ontocast.onto.constants import ONTOLOGY_NULL_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.parallel_state import UnitFactsState, UnitOntologyState
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState
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


def test_unit_state_to_agent_state_is_isolated() -> None:
    state = UnitFactsState(
        content_unit=_build_content_unit(), ontology_snapshot=_build_ontology()
    )
    agent_state = state.to_agent_state()
    agent_state.current_content_unit.text = "MUTATED"

    assert state.content_unit.text == "Alice works for ACME."


@pytest.mark.anyio
async def test_run_unit_facts_loop_uses_dedicated_state(monkeypatch) -> None:
    async def fake_render(state: AgentState, tools) -> AgentState:
        state.status = Status.SUCCESS
        return state

    async def fake_critic(state: AgentState, tools) -> AgentState:
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(unit_loops, "render_facts", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_facts", fake_critic)

    state = UnitFactsState(
        content_unit=_build_content_unit(), ontology_snapshot=_build_ontology()
    )
    tools = cast(ToolBox, object())
    result = await unit_loops.run_unit_facts_loop(state, tools=tools)

    assert result.status == Status.SUCCESS
    assert result.output_unit is not None
    assert result.output_unit.hid == state.content_unit.hid


@pytest.mark.anyio
async def test_run_unit_ontology_loop_emits_updates(monkeypatch) -> None:
    async def fake_render(state: AgentState, tools) -> AgentState:
        state.status = Status.SUCCESS
        state.ontology_updates = [GraphUpdate()]
        return state

    async def fake_critic(state: AgentState, tools) -> AgentState:
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(unit_loops, "render_ontology", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", fake_critic)

    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=Ontology(iri=ONTOLOGY_NULL_IRI),
    )
    tools = cast(ToolBox, object())
    result = await unit_loops.run_unit_ontology_loop(state, tools=tools)

    assert result.status == Status.SUCCESS
    assert len(result.output_updates) == 1


def test_reduce_ontology_updates_returns_baseline_when_empty() -> None:
    baseline = _build_ontology()
    reduced = reduce_ontology_updates(baseline, updates=[], ontology_max_triples=100)

    assert reduced is baseline


def _insert_update(s: str, p: str, o: str) -> GraphUpdate:
    graph = RDFGraph()
    graph.parse(
        data=f"""
        @prefix ex: <https://example.com/onto#> .
        ex:{s} ex:{p} ex:{o} .
        """,
        format="turtle",
    )
    from ontocast.onto.sparql_models import TripleOp

    return GraphUpdate(triple_operations=[TripleOp(type="insert", graph=graph)])


def test_reduce_ontology_updates_is_deterministic_and_deduplicates() -> None:
    baseline = _build_ontology()
    update_a = _insert_update("A", "relatedTo", "B")
    update_b = _insert_update("C", "relatedTo", "D")

    reduced_1 = reduce_ontology_updates(
        baseline,
        updates=[update_b, update_a, update_b],
        ontology_max_triples=1000,
    )
    reduced_2 = reduce_ontology_updates(
        baseline,
        updates=[update_a, update_b],
        ontology_max_triples=1000,
    )

    assert len(reduced_1.graph) == len(reduced_2.graph)
    assert set(reduced_1.graph) == set(reduced_2.graph)
