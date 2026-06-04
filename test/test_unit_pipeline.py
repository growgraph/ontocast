from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.onto.docling_helpers import plain_text_to_docling_doc
from ontocast.onto.enum import OntologyAssemblyMode, RenderMode, Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph import unit_pipeline
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.toolbox import ToolBox


def _build_ontology(iri: str = "https://example.com/onto") -> Ontology:
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix onto: <https://example.com/onto#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        onto:CompanyOntology a owl:Ontology .
        """,
        format="turtle",
    )
    return Ontology(graph=graph, iri=iri)


def _minimal_agent_state() -> AgentState:
    return AgentState(
        raw_input={"doc.txt": b'"Alice works for ACME."'},
        docling_doc=plain_text_to_docling_doc("Alice works for ACME.", "doc.txt"),
        doc_iri=URIRef("https://example.com/doc/d1"),
        render_mode=RenderMode.ONTOLOGY_AND_FACTS,
    )


@pytest.mark.anyio
async def test_run_unit_pipeline_feeds_ontology_loop_output_to_facts(
    monkeypatch,
) -> None:
    """Facts loop must not re-resolve context after ontology loop on the same unit."""
    evolved = _build_ontology("https://example.com/onto/evolved")
    captured_context: UnitOntologyContext | None = None
    captured_ontology: Ontology | None = None

    async def fake_ontology_loop(
        state: UnitOntologyState, tools: ToolBox, document_state: AgentState
    ) -> UnitOntologyState:
        state.status = Status.SUCCESS
        state.ontology_snapshot = _build_ontology("https://example.com/onto/seed")
        state.current_ontology = evolved
        state.assembly_anchor_iri = "https://example.com/onto/seed"
        state.ontology_patch_sources = ["https://example.com/onto/seed"]
        state.assembly_mode_used = OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY
        return state

    async def fake_facts_loop(
        state: UnitFactsState,
        tools: ToolBox,
        document_state: AgentState,
        *,
        pre_resolved_ontology: Ontology | None = None,
        pre_resolved_context: UnitOntologyContext | None = None,
    ) -> UnitFactsState:
        nonlocal captured_context, captured_ontology
        captured_context = pre_resolved_context
        captured_ontology = pre_resolved_ontology
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(unit_pipeline, "convert_document", lambda _s, _t: None)
    monkeypatch.setattr(unit_pipeline, "ontology_loop", fake_ontology_loop)
    monkeypatch.setattr(unit_pipeline, "facts_loop", fake_facts_loop)

    agent_state = _minimal_agent_state()
    tools = cast(
        ToolBox,
        SimpleNamespace(
            config=SimpleNamespace(
                server=SimpleNamespace(
                    max_visits_per_node=3,
                    ontology_max_triples=50_000,
                )
            )
        ),
    )

    onto_result, facts_result = await unit_pipeline.run_unit_pipeline(
        agent_state, tools
    )

    assert onto_result is not None
    assert facts_result is not None
    ctx = captured_context
    assert ctx is not None
    assert captured_ontology is None
    assert ctx.ontology_snapshot is evolved
    assert ctx.anchor_iri == "https://example.com/onto/seed"
    assert ctx.patch_sources == ["https://example.com/onto/seed"]
    assert ctx.assembly_mode == OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY
    assert agent_state.reduced_ontology_artifacts == [evolved]


@pytest.mark.anyio
async def test_run_unit_pipeline_uses_agent_state_max_visits(monkeypatch) -> None:
    """Per-request max_visits on AgentState drives unit loop limits, not server config."""
    captured: list[int] = []

    async def fake_ontology_loop(
        state: UnitOntologyState, tools: ToolBox, document_state: AgentState
    ) -> UnitOntologyState:
        captured.append(state.max_visits_per_node)
        state.status = Status.SUCCESS
        return state

    async def fake_facts_loop(
        state: UnitFactsState,
        tools: ToolBox,
        document_state: AgentState,
        *,
        pre_resolved_ontology: Ontology | None = None,
        pre_resolved_context: UnitOntologyContext | None = None,
    ) -> UnitFactsState:
        captured.append(state.max_visits_per_node)
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(unit_pipeline, "convert_document", lambda _s, _t: None)
    monkeypatch.setattr(unit_pipeline, "ontology_loop", fake_ontology_loop)
    monkeypatch.setattr(unit_pipeline, "facts_loop", fake_facts_loop)

    agent_state = _minimal_agent_state()
    agent_state.max_visits = 7
    tools = cast(
        ToolBox,
        SimpleNamespace(
            config=SimpleNamespace(
                server=SimpleNamespace(
                    max_visits_per_node=2,
                    ontology_max_triples=50_000,
                )
            )
        ),
    )

    await unit_pipeline.run_unit_pipeline(agent_state, tools)

    assert captured == [7, 7]
