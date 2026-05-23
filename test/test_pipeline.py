import importlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import OWL, RDF, BNode, Literal, URIRef

from ontocast.agent.chunk_text import chunk_text as _chunk_text
from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.config import (
    Config,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    PathConfig,
    ToolConfig,
)
from ontocast.onto.constants import ONTOLOGY_NULL_IRI, PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType, SourceUnit
from ontocast.onto.enum import (
    LLMGraphFormat,
    OntologyAssemblyMode,
    OntologyContextMode,
    RenderMode,
    Status,
    WorkflowNode,
)
from ontocast.onto.model import (
    ExternalEvidenceCacheEntry,
    ExternalEvidencePlan,
    ExternalEvidenceRequest,
    GraphUpdateRenderReport,
    OntologyCritiqueReport,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GenericSparqlQuery, GraphUpdate, TripleOp
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph import create_agent_graph
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.helpers import build_ontology_delta_graph
from ontocast.stategraph.node_factories import make_normalize_ontology_node
from ontocast.stategraph.routing import (
    route_after_chunk,
    route_after_ontology_consolidation,
)
from ontocast.tool import EmbeddingBasedAggregator
from ontocast.tool.atomic import AtomicToolBox, SearchHit
from ontocast.toolbox import ToolBox

render_ontology_module = importlib.import_module("ontocast.agent.render_ontology")
criticise_ontology_module = importlib.import_module("ontocast.agent.criticise_ontology")
unit_loops = importlib.import_module("ontocast.stategraph.atomic")
external_evidence_module = importlib.import_module("ontocast.agent.external_evidence")


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
    async def fake_render(state: UnitFactsState, tools, **kwargs) -> UnitFactsState:
        state.status = Status.SUCCESS
        return state

    async def fake_critic(state: UnitFactsState, tools) -> UnitFactsState:
        state.status = Status.SUCCESS
        return state

    async def fake_resolve(_state, _tools, _unit):
        return UnitOntologyContext(
            anchor_iri="https://example.org/o",
            ontology_snapshot=_build_ontology(),
            patch_sources=[],
            assembly_mode=OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
            confidence=1.0,
        )

    monkeypatch.setattr(unit_loops, "render_facts", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_facts", fake_critic)
    monkeypatch.setattr(
        unit_loops, "resolve_effective_facts_ontology_context", fake_resolve
    )

    state = UnitFactsState(
        content_unit=_build_content_unit(), ontology_snapshot=_build_ontology()
    )
    toolbox = cast(
        ToolBox,
        SimpleNamespace(get_atomic_tools=lambda: cast(AtomicToolBox, object())),
    )
    document_state = AgentState(render_mode=RenderMode.FACTS)
    result = await unit_loops.facts_loop(state, toolbox, document_state)

    assert result.status == Status.SUCCESS
    assert result.content_unit.hid == state.content_unit.hid


@pytest.mark.anyio
async def test_run_unit_ontology_loop_emits_updates(monkeypatch) -> None:
    async def fake_render(
        state: UnitOntologyState, tools, **kwargs
    ) -> UnitOntologyState:
        state.status = Status.SUCCESS
        state.ontology_updates = [GraphUpdate()]
        state.current_ontology = Ontology(
            graph=RDFGraph(), iri="https://example.com/onto"
        )
        return state

    async def fake_critic(state: UnitOntologyState, tools) -> UnitOntologyState:
        state.status = Status.SUCCESS
        return state

    async def fake_resolve(_state, _tools, _unit):
        return UnitOntologyContext(
            anchor_iri="https://example.com/onto",
            ontology_snapshot=Ontology(iri=ONTOLOGY_NULL_IRI),
            patch_sources=[],
            assembly_mode=OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
            confidence=1.0,
        )

    monkeypatch.setattr(unit_loops, "render_ontology", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", fake_critic)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", fake_resolve)

    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=Ontology(iri=ONTOLOGY_NULL_IRI),
    )
    toolbox = cast(
        ToolBox,
        SimpleNamespace(get_atomic_tools=lambda: cast(AtomicToolBox, object())),
    )
    document_state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
    )
    result = await unit_loops.ontology_loop(state, toolbox, document_state)

    assert result.status == Status.SUCCESS
    assert len(result.all_updates) == 1


def test_reduce_ontology_units_returns_ontology_when_no_units() -> None:
    tools = ToolBox.__new__(ToolBox)
    tools.aggregator = EmbeddingBasedAggregator()
    reduced, applied, provenance = normalize_ontology_units(units=[], tools=tools)

    assert reduced is not None
    assert reduced.iri is not None
    assert applied == []
    assert len(provenance) == 0


def test_reduce_ontology_units_merges_unit_graphs_without_aggregator() -> None:
    tools = ToolBox.__new__(ToolBox)
    tools.aggregator = EmbeddingBasedAggregator()
    unit1 = ContentUnit(
        text="Alice works at ACME",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=_build_ontology().graph,
        type=OutputType.ONTOLOGIES,
    )
    reduced, applied, provenance = normalize_ontology_units(units=[unit1], tools=tools)

    assert reduced is not None
    assert len(reduced.graph) > 0
    assert len(applied) == 1
    assert len(applied[0].triple_operations) == 1
    assert len(provenance) == 0
    assert isinstance(applied, list)


def test_reduce_ontology_units_creates_base_when_required() -> None:
    tools = cast(ToolBox, ToolBox.__new__(ToolBox))
    tools.aggregator = EmbeddingBasedAggregator()
    delta_graph = RDFGraph()
    delta_graph.parse(
        data="""
        @prefix ex: <https://example.com/onto#> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:Company rdf:type rdfs:Class .
        """,
        format="turtle",
    )
    unit = ContentUnit(
        text="Company ontology snippet",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=delta_graph,
        type=OutputType.ONTOLOGIES,
    )
    reduced, applied, provenance = normalize_ontology_units(
        units=[unit],
        tools=tools,
        base_ontology=None,
        require_base=True,
    )

    assert not reduced.is_null()
    assert len(reduced.graph) > 0
    assert len(provenance) == 0
    assert isinstance(applied, list)


def test_reduce_ontology_units_strips_provenance_and_stores_artifact() -> None:
    tools = ToolBox.__new__(ToolBox)
    tools.aggregator = EmbeddingBasedAggregator()
    doc_iri = URIRef("https://growgraph.dev/doc/test")
    court = URIRef("https://growgraph.dev/fcaont#Court")
    appeal_court = URIRef("https://growgraph.dev/fcaont#AppealCourt")
    reifier = BNode()
    source_chunk = URIRef(f"{doc_iri}/chunk-1")

    graph = RDFGraph(store="oxigraph")
    graph.add((appeal_court, RDF.type, court))
    graph.add((appeal_court, OWL.sameAs, court))
    graph.add((source_chunk, RDF.type, PROV.Entity))
    graph.add((source_chunk, SCHEMA.identifier, Literal("chunk-1")))
    graph.add((reifier, RDF_REIFIES, Literal("quoted-triple")))
    graph.add((reifier, PROV.wasDerivedFrom, source_chunk))

    unit = ContentUnit(
        text="Appeal court ontology unit",
        index=0,
        doc_iri=doc_iri,
        graph=graph,
        type=OutputType.ONTOLOGIES,
    )
    reduced, _, provenance = normalize_ontology_units(units=[unit], tools=tools)

    assert (appeal_court, RDF.type, court) in reduced.graph
    assert (appeal_court, OWL.sameAs, court) not in reduced.graph
    assert (source_chunk, SCHEMA.identifier, Literal("chunk-1")) not in reduced.graph

    assert (appeal_court, OWL.sameAs, court) in provenance
    assert list(provenance.triples((None, RDF_REIFIES, None)))
    assert list(provenance.triples((None, PROV.wasDerivedFrom, source_chunk)))


def test_normalize_ontology_node_feeds_clean_graph_to_consolidation() -> None:
    class DummyTools:
        aggregator = EmbeddingBasedAggregator()

    normalize_node = make_normalize_ontology_node(cast(ToolBox, DummyTools()))

    doc_iri = URIRef("https://growgraph.dev/doc/test-node")
    class_uri = URIRef("https://growgraph.dev/fcaont#Judgement")
    source_chunk = URIRef(f"{doc_iri}/chunk-1")
    graph = RDFGraph()
    graph.add(
        (class_uri, RDF.type, URIRef("http://www.w3.org/2000/01/rdf-schema#Class"))
    )
    graph.add((source_chunk, RDF.type, PROV.Entity))
    graph.add((source_chunk, SCHEMA.identifier, Literal("chunk-1")))
    graph.add((class_uri, OWL.sameAs, URIRef("https://growgraph.dev/fcaont#Judgment")))

    state = AgentState(render_mode=RenderMode.ONTOLOGY)
    state.reduced_ontology_artifacts = [_build_ontology()]
    state.ontology_artifacts = list(state.reduced_ontology_artifacts)
    state.ontology_units = [
        ContentUnit(
            text="Ontology delta",
            index=0,
            doc_iri=doc_iri,
            graph=graph,
            type=OutputType.ONTOLOGIES,
        )
    ]

    updated = normalize_node(state)
    ontology_ttl = updated.reduced_ontology_artifacts[0].graph.serialize(
        format="turtle"
    )

    assert "rdf:reifies" not in ontology_ttl
    assert f"{doc_iri}/chunk-1" not in ontology_ttl
    assert "owl:sameAs" not in ontology_ttl
    assert len(updated.ontology_provenance_artifact) > 0


def test_normalize_ontology_node_skips_global_reduce_for_multi_anchor_artifacts(
    caplog,
) -> None:
    """Multi-anchor documents skip global normalization by design.

    When more than one anchor artifact is present the normalize node returns
    early without applying base-ontology versioning or provenance stripping.
    This is an intentional short-circuit: cross-anchor reconciliation is not
    yet implemented.  A WARNING must be emitted so operators can observe
    that normalization was bypassed.
    """

    class DummyTools:
        aggregator = EmbeddingBasedAggregator()

    normalize_node = make_normalize_ontology_node(cast(ToolBox, DummyTools()))
    a1 = _build_ontology()
    a2 = _build_ontology()
    state = AgentState(render_mode=RenderMode.ONTOLOGY)
    state.reduced_ontology_artifacts = [a1, a2]
    state.ontology_artifacts = [a1, a2]
    state.ontology_units = [
        ContentUnit(
            text="Ontology delta",
            index=0,
            doc_iri=URIRef("https://growgraph.dev/doc/test-node"),
            graph=RDFGraph(),
            type=OutputType.ONTOLOGIES,
        )
    ]

    with caplog.at_level(logging.WARNING, logger="ontocast.stategraph.node_factories"):
        updated = normalize_node(state)

    assert updated.reduced_ontology_artifacts == [a1, a2]
    assert updated.ontology_artifacts == [a1, a2]
    assert len(updated.ontology_provenance_artifact) == 0
    assert updated.ontology_reduce_metrics["normalized_ontology_updates"] == 0
    assert any(
        "normalization" in record.message.lower() or "anchor" in record.message.lower()
        for record in caplog.records
    ), "Expected a warning log about skipped normalization for multi-anchor documents"


@pytest.mark.anyio
async def test_render_ontology_uses_update_when_snapshot_exists(monkeypatch) -> None:
    calls = {"fresh": 0, "update": 0}

    async def fake_fresh(
        state: UnitOntologyState, tools, **kwargs
    ) -> UnitOntologyState:
        calls["fresh"] += 1
        return state

    async def fake_update(
        state: UnitOntologyState, tools, **kwargs
    ) -> UnitOntologyState:
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
        state, tools=cast(AtomicToolBox, object())
    )

    assert result is state
    assert calls["update"] == 1
    assert calls["fresh"] == 0


@pytest.mark.anyio
async def test_render_ontology_update_adds_external_evidence_when_enabled(
    monkeypatch,
) -> None:
    captured_prompt_kwargs: dict[str, object] = {}

    async def fake_call_llm_with_retry(**kwargs):
        captured_prompt_kwargs.update(kwargs["prompt_kwargs"])
        return GraphUpdateRenderReport(graph_update=GraphUpdate())

    async def fake_get_llm_tool(_budget_tracker):
        return object()

    monkeypatch.setattr(
        render_ontology_module, "call_llm_with_retry", fake_call_llm_with_retry
    )
    tools = cast(
        AtomicToolBox,
        SimpleNamespace(
            get_llm_tool=fake_get_llm_tool,
        ),
    )
    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=_build_ontology(),
    )
    state.external_evidence_text = (
        "### EXTERNAL EVIDENCE (WEB SEARCH)\n"
        "1. Ontology engineering patterns | https://example.org/ontology\n"
        "   Use consistent subclass hierarchies and explicit domains."
    )

    await render_ontology_module.render_ontology_update(state, tools=tools)

    external_evidence = str(captured_prompt_kwargs.get("external_evidence", ""))
    assert "EXTERNAL EVIDENCE" in external_evidence
    assert "https://example.org/ontology" in external_evidence


@pytest.mark.anyio
async def test_criticise_ontology_skips_external_evidence_when_disabled(
    monkeypatch,
) -> None:
    captured_prompt_kwargs: dict[str, object] = {}

    async def fake_call_llm_with_retry(**kwargs):
        captured_prompt_kwargs.update(kwargs["prompt_kwargs"])
        return OntologyCritiqueReport(
            success=True,
            score=95,
            systemic_critique_summary="Looks good.",
            actionable_ontology_fixes=[],
        )

    async def fake_get_llm_tool(_budget_tracker):
        return object()

    monkeypatch.setattr(
        criticise_ontology_module, "call_llm_with_retry", fake_call_llm_with_retry
    )
    tools = cast(
        AtomicToolBox,
        SimpleNamespace(
            get_llm_tool=fake_get_llm_tool,
        ),
    )
    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=_build_ontology(),
    )

    await criticise_ontology_module.criticise_ontology(state, tools=tools)

    assert captured_prompt_kwargs.get("external_evidence") == ""


@pytest.mark.anyio
async def test_criticise_ontology_prompt_includes_graph_format_instruction(
    monkeypatch,
) -> None:
    captured_prompt_kwargs: dict[str, object] = {}

    async def fake_call_llm_with_retry(**kwargs):
        captured_prompt_kwargs.update(kwargs["prompt_kwargs"])
        return OntologyCritiqueReport(
            success=True,
            score=95,
            systemic_critique_summary="Looks good.",
            actionable_ontology_fixes=[],
        )

    async def fake_get_llm_tool(_budget_tracker):
        return object()

    monkeypatch.setattr(
        criticise_ontology_module, "call_llm_with_retry", fake_call_llm_with_retry
    )
    tools = cast(
        AtomicToolBox,
        SimpleNamespace(
            get_llm_tool=fake_get_llm_tool,
        ),
    )
    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=_build_ontology(),
        llm_graph_format=LLMGraphFormat.JSONLD,
    )

    await criticise_ontology_module.criticise_ontology(state, tools=tools)

    instruction = str(captured_prompt_kwargs.get("graph_format_instruction", ""))
    assert "LLM_GRAPH_FORMAT=jsonld" in instruction
    assert "incorrect_value" in instruction


@pytest.mark.anyio
async def test_plan_external_evidence_uses_fallback_when_planner_disabled() -> None:
    tools = cast(
        AtomicToolBox,
        SimpleNamespace(
            web_grounding_enabled_for_node=lambda _node: True,
            web_search_reuse_evidence_across_attempt=False,
            web_search_planner_enabled=False,
            web_search_planner_min_query_chars=8,
            web_search_planner_max_queries=3,
            web_search_planner_min_confidence=0.35,
        ),
    )
    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=_build_ontology(),
        ontology_user_instruction="Clarify company ontology terms.",
    )
    state.set_external_evidence_request(
        WorkflowNode.TEXT_TO_ONTOLOGY,
        ExternalEvidenceRequest(
            initiate_search=True,
            rationale="Need targeted terminology lookup for ontology refinement.",
        ),
    )

    planned = await external_evidence_module.plan_external_evidence_for_node(
        state, tools, WorkflowNode.TEXT_TO_ONTOLOGY
    )

    assert planned.external_evidence_plan.should_search is True
    assert planned.external_evidence_plan.queries
    assert planned.external_evidence_planned_at_node == WorkflowNode.TEXT_TO_ONTOLOGY


@pytest.mark.anyio
async def test_fetch_external_evidence_filters_domains_and_dedupes() -> None:
    async def fake_search(query: str, max_results: int | None = None):
        _ = query, max_results
        return [
            SearchHit(
                title="Good result",
                url="https://example.org/ontology",
                snippet="This is a sufficiently detailed snippet for ontology guidance.",
            ),
            SearchHit(
                title="Duplicate URL",
                url="https://example.org/ontology",
                snippet="Different text but same URL should be deduped.",
            ),
            SearchHit(
                title="Other domain",
                url="https://noise.test/entry",
                snippet="This snippet is long enough but should be filtered by allowlist.",
            ),
        ]

    tools = cast(
        AtomicToolBox,
        SimpleNamespace(
            web_grounding_enabled_for_node=lambda _node: True,
            search=fake_search,
            web_search_allowed_domains={"example.org"},
            web_search_blocked_domains=set(),
            web_search_min_snippet_chars=20,
            web_search_max_snippet_chars=180,
            web_search_max_total_chars=1200,
        ),
    )
    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=_build_ontology(),
    )
    state.set_external_evidence_request(
        WorkflowNode.TEXT_TO_ONTOLOGY,
        ExternalEvidenceRequest(
            initiate_search=True,
            rationale="Need clarification",
            query_hints=["ontology engineering patterns"],
            confidence=0.9,
        ),
    )
    state.set_external_evidence_cache_entry(
        WorkflowNode.TEXT_TO_ONTOLOGY,
        ExternalEvidenceCacheEntry(
            plan=ExternalEvidencePlan(
                should_search=True,
                rationale="Need clarification",
                intent="definition",
                confidence=0.9,
                queries=["ontology engineering patterns"],
            ),
        ),
    )

    fetched = await external_evidence_module.fetch_external_evidence_for_node(
        state, tools, WorkflowNode.TEXT_TO_ONTOLOGY
    )

    assert fetched.external_evidence_source_count == 1
    assert fetched.external_evidence_domains == ["example.org"]
    assert "https://example.org/ontology" in fetched.external_evidence_text


@pytest.mark.anyio
async def test_ontology_loop_runs_external_evidence_nodes(monkeypatch) -> None:
    called_nodes: list[WorkflowNode] = []

    async def fake_plan(state: UnitOntologyState, tools, target_node: WorkflowNode):
        _ = tools
        called_nodes.append(target_node)
        return state

    async def fake_fetch(state: UnitOntologyState, tools, target_node: WorkflowNode):
        _ = tools, target_node
        return state

    async def fake_render(
        state: UnitOntologyState, tools, **kwargs
    ) -> UnitOntologyState:
        _ = tools
        state.status = Status.SUCCESS
        return state

    async def fake_critic(state: UnitOntologyState, tools) -> UnitOntologyState:
        _ = tools
        state.status = Status.SUCCESS
        return state

    async def fake_resolve(_state, _tools, _unit):
        return UnitOntologyContext(
            anchor_iri="https://example.com/onto",
            ontology_snapshot=Ontology(iri=ONTOLOGY_NULL_IRI),
            patch_sources=[],
            assembly_mode=OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
            confidence=1.0,
        )

    monkeypatch.setattr(unit_loops, "plan_external_evidence_for_node", fake_plan)
    monkeypatch.setattr(unit_loops, "fetch_external_evidence_for_node", fake_fetch)
    monkeypatch.setattr(unit_loops, "render_ontology", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", fake_critic)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", fake_resolve)

    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=Ontology(iri=ONTOLOGY_NULL_IRI),
    )
    toolbox = cast(
        ToolBox,
        SimpleNamespace(get_atomic_tools=lambda: cast(AtomicToolBox, object())),
    )
    document_state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
    )
    result = await unit_loops.ontology_loop(state, toolbox, document_state)

    assert result.status == Status.SUCCESS
    assert called_nodes == []


@pytest.mark.anyio
async def test_ontology_loop_plans_search_when_critic_requests_it(monkeypatch) -> None:
    called_nodes: list[WorkflowNode] = []

    async def fake_plan(state: UnitOntologyState, tools, target_node: WorkflowNode):
        _ = tools
        called_nodes.append(target_node)
        return state

    async def fake_fetch(state: UnitOntologyState, tools, target_node: WorkflowNode):
        _ = tools
        called_nodes.append(target_node)
        return state

    async def fake_render(
        state: UnitOntologyState, tools, **kwargs
    ) -> UnitOntologyState:
        _ = tools
        state.status = Status.SUCCESS
        return state

    critic_calls = {"count": 0}

    async def fake_critic(state: UnitOntologyState, tools) -> UnitOntologyState:
        _ = tools
        critic_calls["count"] += 1
        if critic_calls["count"] == 1:
            state.status = Status.FAILED
            state.set_external_evidence_request(
                WorkflowNode.CRITICISE_ONTOLOGY,
                ExternalEvidenceRequest(
                    initiate_search=True,
                    rationale="Need domain standard disambiguation.",
                    query_hints=["ontology modeling standard pattern"],
                ),
            )
            return state
        state.status = Status.SUCCESS
        return state

    async def fake_resolve(_state, _tools, _unit):
        return UnitOntologyContext(
            anchor_iri="https://example.com/onto",
            ontology_snapshot=Ontology(iri=ONTOLOGY_NULL_IRI),
            patch_sources=[],
            assembly_mode=OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
            confidence=1.0,
        )

    monkeypatch.setattr(unit_loops, "plan_external_evidence_for_node", fake_plan)
    monkeypatch.setattr(unit_loops, "fetch_external_evidence_for_node", fake_fetch)
    monkeypatch.setattr(unit_loops, "render_ontology", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", fake_critic)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", fake_resolve)

    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=Ontology(iri=ONTOLOGY_NULL_IRI),
        # Need a later render attempt possible so critic runs (final render skips critic).
        max_visits_per_node=2,
    )
    toolbox = cast(
        ToolBox,
        SimpleNamespace(get_atomic_tools=lambda: cast(AtomicToolBox, object())),
    )
    document_state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
    )
    result = await unit_loops.ontology_loop(state, toolbox, document_state)

    assert result.status == Status.SUCCESS
    assert called_nodes == [
        WorkflowNode.CRITICISE_ONTOLOGY,
        WorkflowNode.CRITICISE_ONTOLOGY,
    ]


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


def test_route_after_ontology_consolidation_respects_ontology_only_mode() -> None:
    ontology_only = AgentState(render_mode=RenderMode.ONTOLOGY)
    assert (
        route_after_ontology_consolidation(ontology_only)
        == WorkflowNode.STRUCTURAL_CHECK
    )

    ontology_and_facts = AgentState(render_mode=RenderMode.ONTOLOGY_AND_FACTS)
    assert (
        route_after_ontology_consolidation(ontology_and_facts)
        == WorkflowNode.STRUCTURAL_CHECK
    )


def test_agent_graph_structural_check_not_reached_from_facts_edges() -> None:
    # Use a minimal config that enables the filesystem triple-store backend.
    # This keeps the graph build lightweight and avoids external services.
    config = Config(
        tool_config=ToolConfig(
            path_config=PathConfig(working_directory=Path("/tmp")),
            llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name=OllamaModel.LLAMA3_1,
                base_url="http://localhost:11434",
            ),
        ),
    )
    toolbox = ToolBox(config)
    app = create_agent_graph(toolbox)
    graph = app.get_graph()

    structural_check = WorkflowNode.STRUCTURAL_CHECK
    facts_sources = {WorkflowNode.RENDER_FACTS, WorkflowNode.MERGE_FACTS}

    incoming_from_facts = [
        (start, end)
        for start, end, _data, _conditional in graph.edges
        if end == structural_check and start in facts_sources
    ]
    assert incoming_from_facts == []


def test_route_after_chunk_facts_only_skips_ontology() -> None:
    facts_only = AgentState(render_mode=RenderMode.FACTS)
    assert route_after_chunk(facts_only) == WorkflowNode.RENDER_FACTS


def test_toolbox_serialize_skips_facts_in_ontology_only_mode() -> None:
    class RecordingOntologyManager:
        def __init__(self) -> None:
            self.added = 0

        def add_ontology(self, ontology: Ontology) -> None:
            self.added += 1

    class RecordingStore:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str | None]] = []

        def serialize(self, payload: object, graph_uri: str | None = None) -> None:
            self.calls.append((payload, graph_uri))

    state = AgentState(render_mode=RenderMode.ONTOLOGY)
    state.reduced_ontology_artifacts = [_build_ontology()]
    state.ontology_artifacts = list(state.reduced_ontology_artifacts)
    store = RecordingStore()
    toolbox = SimpleNamespace(
        ontology_manager=RecordingOntologyManager(),
        filesystem_manager=store,
        triple_store_manager=None,
    )

    ToolBox.serialize(cast(ToolBox, toolbox), state)

    assert len(store.calls) == 1
    assert isinstance(store.calls[0][0], Ontology)
    assert store.calls[0][1] is None


def test_toolbox_serialize_includes_facts_when_render_facts_enabled() -> None:
    class RecordingOntologyManager:
        def add_ontology(self, ontology: Ontology) -> None:
            return None

    class RecordingStore:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str | None]] = []

        def serialize(self, payload: object, graph_uri: str | None = None) -> None:
            self.calls.append((payload, graph_uri))

    state = AgentState(render_mode=RenderMode.ONTOLOGY_AND_FACTS)
    state.reduced_ontology_artifacts = [_build_ontology()]
    state.ontology_artifacts = list(state.reduced_ontology_artifacts)
    store = RecordingStore()
    toolbox = SimpleNamespace(
        ontology_manager=RecordingOntologyManager(),
        filesystem_manager=store,
        triple_store_manager=None,
    )

    ToolBox.serialize(cast(ToolBox, toolbox), state)

    assert len(store.calls) == 2
    assert isinstance(store.calls[0][0], Ontology)
    assert isinstance(store.calls[1][0], RDFGraph)
    assert store.calls[1][1] == state.graph_uri


def test_toolbox_serialize_persists_all_ontology_artifacts() -> None:
    class RecordingOntologyManager:
        def __init__(self) -> None:
            self.added = 0

        def add_ontology(self, ontology: Ontology) -> None:
            _ = ontology
            self.added += 1

    class RecordingStore:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str | None]] = []

        def serialize(self, payload: object, graph_uri: str | None = None) -> None:
            self.calls.append((payload, graph_uri))

    state = AgentState(render_mode=RenderMode.ONTOLOGY)
    state.ontology_artifacts = [_build_ontology(), _build_ontology()]
    store = RecordingStore()
    manager = RecordingOntologyManager()
    toolbox = SimpleNamespace(
        ontology_manager=manager,
        filesystem_manager=store,
        triple_store_manager=None,
    )

    ToolBox.serialize(cast(ToolBox, toolbox), state)

    assert manager.added == 2
    assert len(store.calls) == 2
    assert all(isinstance(payload, Ontology) for payload, _ in store.calls)


def test_render_updated_graph_splits_compound_sparql_insert_updates() -> None:
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix ex: <http://example.org/> .
        ex:Existing ex:kept ex:Value .
        """,
        format="turtle",
    )
    update = GraphUpdate(
        sparql_operations=[
            GenericSparqlQuery(
                query=(
                    "PREFIX ex: <http://example.org/>\n"
                    "INSERT DATA { ex:Person ex:label ex:Alice }\n"
                    "INSERT DATA { ex:Person ex:status ex:Active }"
                )
            )
        ]
    )

    updated_graph, was_applied = AgentState.render_updated_graph(graph, [update])

    assert was_applied is True
    assert (
        URIRef("http://example.org/Person"),
        URIRef("http://example.org/label"),
        URIRef("http://example.org/Alice"),
    ) in updated_graph
    assert (
        URIRef("http://example.org/Person"),
        URIRef("http://example.org/status"),
        URIRef("http://example.org/Active"),
    ) in updated_graph


def test_render_updated_graph_splits_compound_sparql_with_many_prefixes() -> None:
    """Regression: shared PREFIX block + second INSERT at ~line 44 (Text2KGBench style)."""
    from ontocast.onto.sparql_models import STANDARD_PREFIXES

    graph = RDFGraph()
    graph.parse(
        data="@prefix ex: <http://example.org/> . ex:Existing ex:kept ex:Value .",
        format="turtle",
    )
    prefix_block = "\n".join(
        f"PREFIX {prefix}: <{uri}>" for prefix, uri in STANDARD_PREFIXES.items()
    )
    compound_query = (
        f"{prefix_block}\n"
        "INSERT DATA { <http://example.org/a> <http://example.org/p1> "
        "<http://example.org/o1> . }\n"
        "INSERT DATA { <http://example.org/a> <http://example.org/p2> "
        "<http://example.org/o2> . }"
    )
    update = GraphUpdate(sparql_operations=[GenericSparqlQuery(query=compound_query)])

    updated_graph, was_applied = AgentState.render_updated_graph(graph, [update])

    assert was_applied is True
    assert len(list(updated_graph)) >= 3


def test_apply_update_query_splits_insert_where_plus_insert_data() -> None:
    graph = RDFGraph()
    graph.parse(
        data="@prefix ex: <http://example.org/> . ex:a ex:p1 ex:o1 .",
        format="turtle",
    )
    query = (
        "PREFIX ex: <http://example.org/>\n"
        "INSERT { ?s ?p ?o } WHERE { ?s ?p ?o . FILTER(?p = ex:p1) }\n"
        "INSERT DATA { ex:a ex:p2 ex:o2 . }"
    )
    AgentState._apply_update_query(graph, query)
    assert (
        URIRef("http://example.org/a"),
        URIRef("http://example.org/p2"),
        URIRef("http://example.org/o2"),
    ) in graph


def test_build_ontology_delta_graph_warns_and_drops_delete_operations(
    caplog,
) -> None:
    """Delete triples in unit GraphUpdates must be warned about and discarded.

    Policy: the ontology map-reduce stage is insert-only. Delete operations
    produced by a unit loop cannot be safely applied across parallel results
    and are intentionally dropped with a warning log.
    """
    base_graph = RDFGraph()
    base_graph.parse(
        data="""
        @prefix ex: <https://example.com/onto#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        ex:Thing a owl:Class .
        ex:obsolete a owl:Class .
        """,
        format="turtle",
    )
    delete_op = GraphUpdate(
        triple_operations=[
            TripleOp(
                type="delete",
                graph=base_graph,
            )
        ]
    )
    insert_graph = RDFGraph()
    insert_graph.parse(
        data="""
        @prefix ex: <https://example.com/onto#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        ex:NewThing a owl:Class .
        """,
        format="turtle",
    )
    insert_op = GraphUpdate(
        triple_operations=[TripleOp(type="insert", graph=insert_graph)]
    )

    onto = _build_ontology()
    onto.graph = base_graph
    state = UnitOntologyState(
        content_unit=SourceUnit(
            text="test",
            index=0,
            doc_iri=URIRef("https://example.com/doc/d1"),
        ),
        ontology_snapshot=onto,
        ontology_updates_applied=[delete_op],
        ontology_updates=[insert_op],
    )

    with caplog.at_level(logging.WARNING, logger="ontocast.stategraph.helpers"):
        delta = build_ontology_delta_graph(state)

    assert any("delete" in record.message.lower() for record in caplog.records), (
        "Expected a warning about dropped delete triples"
    )
    ex_new = URIRef("https://example.com/onto#NewThing")
    ex_obsolete = URIRef("https://example.com/onto#obsolete")
    assert (ex_new, RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class")) in delta
    assert (
        ex_obsolete,
        RDF.type,
        URIRef("http://www.w3.org/2002/07/owl#Class"),
    ) not in delta


def test_chunk_text_resets_content_units_on_each_call() -> None:
    """chunk_text must clear state.content_units before appending new chunks.

    Without this reset a reused AgentState accumulates stale units from
    previous invocations, leading to duplicate processing.
    """

    class FakeChunker:
        def __call__(self, text: str) -> list[str]:
            return [text]

    tools = SimpleNamespace(chunker=FakeChunker())
    state = AgentState(render_mode=RenderMode.ONTOLOGY)
    state.set_text("first invocation text")
    _chunk_text(state, cast(ToolBox, tools))
    assert len(state.content_units) == 1

    state.set_text("second invocation text")
    _chunk_text(state, cast(ToolBox, tools))
    assert len(state.content_units) == 1, (
        "content_units should be reset per call, not accumulated across calls"
    )
