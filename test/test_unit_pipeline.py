import importlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.onto.docling_helpers import plain_text_to_docling_doc
from ontocast.onto.enum import (
    LLMOutputLayout,
    OntologyAssemblyMode,
    OntologyChapterFormat,
    OntologyContextScope,
    RenderMode,
    Status,
)
from ontocast.onto.identity import OntologyIdentity
from ontocast.onto.model import OntologyIdentityProposal
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_condense import TextCaps
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState, InputDocument
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph import unit_pipeline
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.toolbox import ToolBox
from test.snapshot_helpers import snapshot_from_ontology

pytestmark = pytest.mark.unit

resolve_home_ontology_module = importlib.import_module(
    "ontocast.agent.resolve_home_ontology"
)


def _fake_tools(*, max_visits_per_node: int) -> ToolBox:
    """A toolbox for the unit pipeline: config, an empty catalog, a stub LLM."""

    async def get_llm_tool(_budget_tracker: object) -> object:
        return object()

    return cast(
        ToolBox,
        SimpleNamespace(
            config=SimpleNamespace(
                server=SimpleNamespace(
                    max_visits_per_node=max_visits_per_node,
                    ontology_max_triples=50_000,
                    ontology_context_max_triples=4000,
                    ontology_chapter_format=OntologyChapterFormat.INHERIT,
                    llm_output_layout=LLMOutputLayout.COMPACT,
                    ontology_text_caps=TextCaps(),
                    ontology_context_scope=OntologyContextScope.UNIT,
                    fanout_warmup_units=0,
                )
            ),
            shapes_prompt_contract=lambda: ("", (), False),
            ontology_manager=OntologyManager(),
            get_llm_tool=get_llm_tool,
        ),
    )


def _fake_identity_call(monkeypatch) -> None:
    """The home-ontology step's one model call answers with a fixed proposal."""

    async def fake_call_llm_with_retry(**_kwargs) -> OntologyIdentityProposal:
        return OntologyIdentityProposal(
            ontology_id="acme",
            title="ACME ontology",
            description="Who works where.",
        )

    monkeypatch.setattr(
        resolve_home_ontology_module, "call_llm_with_retry", fake_call_llm_with_retry
    )


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
        input_document=InputDocument(
            filename="doc.txt", data=b'"Alice works for ACME."'
        ),
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
    captured_home: OntologyIdentity | None = None

    async def fake_ontology_loop(
        state: UnitOntologyState, tools: ToolBox, document_state: AgentState
    ) -> UnitOntologyState:
        nonlocal captured_home
        captured_home = state.home_ontology
        state.status = Status.SUCCESS
        state.ontology_snapshot = snapshot_from_ontology(
            _build_ontology("https://example.com/onto/seed")
        )
        state.fresh_ontology = evolved
        state.working_graph = evolved.graph.copy()
        state.writable_iris = ["https://example.com/onto/seed"]
        state.assembly_anchor_iri = "https://example.com/onto/seed"
        state.ontology_patch_sources = ["https://example.com/onto/seed"]
        state.assembly_mode_used = OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY
        return state

    async def fake_facts_loop(
        state: UnitFactsState,
        tools: ToolBox,
        document_state: AgentState,
        *,
        pre_resolved_context: UnitOntologyContext | None = None,
    ) -> UnitFactsState:
        nonlocal captured_context, captured_ontology
        captured_context = pre_resolved_context
        captured_ontology = None
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(unit_pipeline, "convert_document", lambda _s, _t: None)
    monkeypatch.setattr(unit_pipeline, "ontology_loop", fake_ontology_loop)
    monkeypatch.setattr(unit_pipeline, "facts_loop", fake_facts_loop)
    _fake_identity_call(monkeypatch)

    agent_state = _minimal_agent_state()
    tools = _fake_tools(max_visits_per_node=3)

    onto_result, facts_result = await unit_pipeline.run_unit_pipeline(
        agent_state, tools
    )

    assert onto_result is not None
    assert facts_result is not None
    # The home ontology is resolved once, before the loop, and handed to it.
    assert agent_state.home_ontology is not None
    assert agent_state.home_ontology.iri == f"{agent_state.current_domain}/acme"
    assert captured_home == agent_state.home_ontology
    ctx = captured_context
    assert ctx is not None
    assert captured_ontology is None
    assert ctx.snapshot.source_iris == [evolved.iri]
    assert ctx.primary_writable_iri == "https://example.com/onto/seed"
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
    _fake_identity_call(monkeypatch)

    agent_state = _minimal_agent_state()
    agent_state.max_visits = 7
    tools = _fake_tools(max_visits_per_node=2)

    await unit_pipeline.run_unit_pipeline(agent_state, tools)

    assert captured == [7, 7]
