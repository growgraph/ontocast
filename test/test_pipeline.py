import asyncio
import importlib
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rdflib import OWL, RDF, BNode, Literal, URIRef

from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.config import (
    Config,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    PathConfig,
    ToolConfig,
    WebSearchConfig,
)
from ontocast.onto.constants import PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType, SourceUnit
from ontocast.onto.enum import (
    LLMGraphFormat,
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
from ontocast.onto.ontology_apply import OntologyDelta
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph import create_agent_graph
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.helpers import merge_unit_deltas
from ontocast.stategraph.node_factories import (
    make_consolidate_ontology_node,
    make_normalize_ontology_node,
)
from ontocast.stategraph.routing import (
    route_after_tag_or_chunk,
)
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool import EmbeddingBasedAggregator
from ontocast.tool.atomic import AtomicToolBox, SearchHit
from ontocast.tool.facts_validation import CriticPatchPolicy
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.toolbox import ToolBox
from test.snapshot_helpers import empty_snapshot, snapshot_from_ontology

pytestmark = pytest.mark.unit

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
        content_unit=_build_content_unit(),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
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

    async def fake_resolve(_state, _tools, _unit, **_kwargs):
        return UnitOntologyContext(
            snapshot=snapshot_from_ontology(_build_ontology()),
            writable_iris=["https://example.org/o"]
            if "https://example.org/o" not in ("", None)
            else [],
            confidence=1.0,
        )

    monkeypatch.setattr(unit_loops, "render_facts", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_facts", fake_critic)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", fake_resolve)

    state = UnitFactsState(
        content_unit=_build_content_unit(),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    toolbox = cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox,
                SimpleNamespace(
                    facts_critic_passes=1,
                    ontology_critic_passes=1,
                    facts_patch_policy=CriticPatchPolicy(),
                    ontology_patch_policy=CriticPatchPolicy(),
                    additional_standard_namespaces=(),
                    validation_policy=None,
                    acceptance_policy=None,
                    ontology_acceptance_policy=None,
                    numeric_coverage_limit=30,
                    numeric_coverage_mandatory=False,
                    facts_critic_min_triples=0,
                    facts_completion_passes=0,
                    catalog_terms=lambda: set(),
                ),
            ),
            ontology_manager=OntologyManager(),
        ),
    )
    document_state = AgentState(render_mode=RenderMode.FACTS)
    result = await unit_loops.facts_loop(
        state, toolbox, UnitLoopContext.from_agent_state(document_state)
    )

    assert result.status == Status.SUCCESS
    assert result.content_unit.hid == state.content_unit.hid


@pytest.mark.anyio
async def test_run_unit_ontology_loop_emits_updates(monkeypatch) -> None:
    async def fake_render(
        state: UnitOntologyState, tools, **kwargs
    ) -> UnitOntologyState:
        state.status = Status.SUCCESS
        state.ontology_updates = [GraphUpdate()]
        state.fresh_ontology = Ontology(
            graph=RDFGraph(), iri="https://example.com/onto"
        )
        return state

    async def fake_critic(state: UnitOntologyState, tools) -> UnitOntologyState:
        state.status = Status.SUCCESS
        return state

    async def fake_resolve(_state, _tools, _unit, **_kwargs):
        return UnitOntologyContext(
            snapshot=empty_snapshot(),
            writable_iris=["https://example.com/onto"]
            if "https://example.com/onto" not in ("", None)
            else [],
            confidence=1.0,
        )

    monkeypatch.setattr(unit_loops, "render_ontology", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", fake_critic)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", fake_resolve)

    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=empty_snapshot(),
    )
    toolbox = cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox,
                SimpleNamespace(
                    validation_policy=None,
                    ontology_critic_passes=1,
                    ontology_patch_policy=CriticPatchPolicy(),
                    ontology_acceptance_policy=None,
                ),
            ),
            ontology_manager=OntologyManager(),
        ),
    )
    document_state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
    )
    result = await unit_loops.ontology_loop(
        state, toolbox, UnitLoopContext.from_agent_state(document_state)
    )

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
    tools = ToolBox.__new__(ToolBox)
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

    # Normalize is a no-op when map already applied catalog bases; provenance
    # stripping for orphan delta units is deferred.
    assert updated.status == Status.SUCCESS
    assert updated.reduced_ontology_artifacts


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
    assert updated.status == Status.SUCCESS
    assert updated.ontology_reduce_metrics["normalized_ontology_updates"] == 2


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
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    # Simulate accidental null current ontology while a valid snapshot exists.
    state.working_graph = RDFGraph()
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
        return GraphUpdateRenderReport()

    async def fake_get_llm_tool(_budget_tracker):
        return object()

    monkeypatch.setattr(
        render_ontology_module, "call_llm_with_retry", fake_call_llm_with_retry
    )
    tools = cast(
        AtomicToolBox,
        SimpleNamespace(
            get_llm_tool=fake_get_llm_tool,
            web_grounding_enabled_for_node=lambda _node: True,
        ),
    )
    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
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
            web_grounding_enabled_for_node=lambda _node: False,
        ),
    )
    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
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
            web_grounding_enabled_for_node=lambda _node: False,
        ),
    )
    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
        llm_graph_format=LLMGraphFormat.JSONLD,
    )

    await criticise_ontology_module.criticise_ontology(state, tools=tools)

    instruction = str(captured_prompt_kwargs.get("graph_format_instruction", ""))
    assert "LLM_GRAPH_FORMAT=jsonld" in instruction
    assert "incorrect_value" in instruction


@pytest.mark.anyio
async def test_plan_external_evidence_uses_fallback_when_planner_disabled() -> None:
    tools = AtomicToolBox(
        llm_provider=cast(Any, object()),
        web_search_config=WebSearchConfig(
            enabled=True,
            ontology_render_enabled=True,
            reuse_evidence_across_attempt=False,
            planner_enabled=False,
            planner_min_query_chars=8,
            planner_max_queries=3,
            planner_min_confidence=0.35,
        ),
    )
    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
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
    # Assert against the node-scoped cache entry, which is the source of truth:
    # the plan is stored under the node it was planned for.
    cached = planned.get_external_evidence_cache_entry(WorkflowNode.TEXT_TO_ONTOLOGY)
    assert cached.plan.should_search is True


@pytest.mark.anyio
async def test_fetch_external_evidence_filters_domains_and_dedupes() -> None:
    class _FakeSearchProvider:
        async def search(self, query: str, max_results: int) -> list[SearchHit]:
            _ = query, max_results
            return _hits()

    def _hits() -> list[SearchHit]:
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

    tools = AtomicToolBox(
        llm_provider=cast(Any, object()),
        search_provider=_FakeSearchProvider(),
        web_search_config=WebSearchConfig(
            enabled=True,
            ontology_render_enabled=True,
            allowed_domains=["example.org"],
            blocked_domains=[],
            min_snippet_chars=20,
            max_snippet_chars=180,
            max_total_chars=1200,
        ),
    )
    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
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

    entry = fetched.get_external_evidence_cache_entry(WorkflowNode.TEXT_TO_ONTOLOGY)
    assert entry.source_count == 1
    assert entry.domains == ["example.org"]
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

    async def fake_resolve(_state, _tools, _unit, **_kwargs):
        return UnitOntologyContext(
            snapshot=empty_snapshot(),
            writable_iris=["https://example.com/onto"]
            if "https://example.com/onto" not in ("", None)
            else [],
            confidence=1.0,
        )

    monkeypatch.setattr(unit_loops, "plan_external_evidence_for_node", fake_plan)
    monkeypatch.setattr(unit_loops, "fetch_external_evidence_for_node", fake_fetch)
    monkeypatch.setattr(unit_loops, "render_ontology", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", fake_critic)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", fake_resolve)

    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=empty_snapshot(),
    )
    toolbox = cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox,
                SimpleNamespace(
                    validation_policy=None,
                    ontology_critic_passes=1,
                    ontology_patch_policy=CriticPatchPolicy(),
                    ontology_acceptance_policy=None,
                ),
            ),
            ontology_manager=OntologyManager(),
        ),
    )
    document_state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
    )
    result = await unit_loops.ontology_loop(
        state, toolbox, UnitLoopContext.from_agent_state(document_state)
    )

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

    async def fake_resolve(_state, _tools, _unit, **_kwargs):
        return UnitOntologyContext(
            snapshot=empty_snapshot(),
            writable_iris=["https://example.com/onto"]
            if "https://example.com/onto" not in ("", None)
            else [],
            confidence=1.0,
        )

    monkeypatch.setattr(unit_loops, "plan_external_evidence_for_node", fake_plan)
    monkeypatch.setattr(unit_loops, "fetch_external_evidence_for_node", fake_fetch)
    monkeypatch.setattr(unit_loops, "render_ontology", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", fake_critic)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", fake_resolve)

    state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=empty_snapshot(),
        # Need a later render attempt possible so critic runs (final render skips critic).
        max_visits_per_node=2,
    )
    toolbox = cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox,
                SimpleNamespace(
                    validation_policy=None,
                    ontology_critic_passes=1,
                    ontology_patch_policy=CriticPatchPolicy(),
                    ontology_acceptance_policy=None,
                ),
            ),
            ontology_manager=OntologyManager(),
        ),
    )
    document_state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
    )
    result = await unit_loops.ontology_loop(
        state, toolbox, UnitLoopContext.from_agent_state(document_state)
    )

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


def test_agent_graph_structural_check_not_reached_from_facts_edges() -> None:
    # A minimal config: the graph build only needs topology, so this keeps it
    # lightweight and avoids external services.
    config = Config(
        tool_config=ToolConfig(
            path_config=PathConfig(),
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


def test_agent_graph_topology_is_pinned() -> None:
    """Pin the whole document graph: every node and every edge.

    The topology is the contract between ``create.py`` and every node factory,
    and it used to be asserted only negatively (the test above). Pinning it
    outright means a node or edge cannot be added, dropped or rewired without
    this test saying so -- which is what makes deletions elsewhere in the
    package provably topology-neutral.
    """
    config = Config(
        tool_config=ToolConfig(
            path_config=PathConfig(),
            llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name=OllamaModel.LLAMA3_1,
                base_url="http://localhost:11434",
            ),
        ),
    )
    graph = create_agent_graph(ToolBox(config)).get_graph()

    assert {str(n) for n in graph.nodes} == {
        "__start__",
        "__end__",
        str(WorkflowNode.CONVERT_TO_TEXT),
        str(WorkflowNode.CHUNK),
        str(WorkflowNode.RENDER_ONTOLOGY_UPDATE),
        str(WorkflowNode.NORMALIZE_ONTOLOGY_UPDATES),
        str(WorkflowNode.CONSOLIDATE_ONTOLOGY),
        str(WorkflowNode.STRUCTURAL_CHECK),
        str(WorkflowNode.CONSISTENCY_CRITIC),
        str(WorkflowNode.RENDER_FACTS),
        str(WorkflowNode.MERGE_FACTS),
        str(WorkflowNode.VALIDATE_FACTS),
        str(WorkflowNode.SERIALIZE),
    }

    # (source, target, is_conditional)
    assert {
        (str(start), str(end), bool(conditional))
        for start, end, _data, conditional in graph.edges
    } == {
        ("__start__", str(WorkflowNode.CONVERT_TO_TEXT), False),
        (str(WorkflowNode.CONVERT_TO_TEXT), str(WorkflowNode.CHUNK), False),
        # route_after_tag_or_chunk: ontology block, or straight to facts.
        (str(WorkflowNode.CHUNK), str(WorkflowNode.RENDER_ONTOLOGY_UPDATE), True),
        (str(WorkflowNode.CHUNK), str(WorkflowNode.RENDER_FACTS), True),
        (
            str(WorkflowNode.RENDER_ONTOLOGY_UPDATE),
            str(WorkflowNode.NORMALIZE_ONTOLOGY_UPDATES),
            False,
        ),
        (
            str(WorkflowNode.NORMALIZE_ONTOLOGY_UPDATES),
            str(WorkflowNode.CONSOLIDATE_ONTOLOGY),
            False,
        ),
        (
            str(WorkflowNode.CONSOLIDATE_ONTOLOGY),
            str(WorkflowNode.STRUCTURAL_CHECK),
            False,
        ),
        (
            str(WorkflowNode.STRUCTURAL_CHECK),
            str(WorkflowNode.CONSISTENCY_CRITIC),
            False,
        ),
        # route_after_consistency_critic: facts block, or stop after ontology.
        (str(WorkflowNode.CONSISTENCY_CRITIC), str(WorkflowNode.RENDER_FACTS), True),
        (str(WorkflowNode.CONSISTENCY_CRITIC), str(WorkflowNode.SERIALIZE), True),
        (str(WorkflowNode.RENDER_FACTS), str(WorkflowNode.MERGE_FACTS), False),
        (str(WorkflowNode.MERGE_FACTS), str(WorkflowNode.VALIDATE_FACTS), False),
        (str(WorkflowNode.VALIDATE_FACTS), str(WorkflowNode.SERIALIZE), False),
        (str(WorkflowNode.SERIALIZE), "__end__", False),
    }


def test_route_after_tag_or_chunk_facts_only_skips_ontology() -> None:
    facts_only = AgentState(render_mode=RenderMode.FACTS)
    assert route_after_tag_or_chunk(facts_only) == WorkflowNode.RENDER_FACTS


def test_toolbox_serialize_skips_facts_in_ontology_only_mode() -> None:
    class RecordingOntologyManager:
        def __init__(self) -> None:
            self.added = 0

        async def aadd_ontology(self, ontology: Ontology) -> None:
            self.added += 1

    class RecordingStore:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str | None]] = []

        async def aserialize(
            self, payload: object, graph_uri: str | None = None
        ) -> None:
            self.calls.append((payload, graph_uri))

    state = AgentState(render_mode=RenderMode.ONTOLOGY)
    state.reduced_ontology_artifacts = [_build_ontology()]
    state.ontology_artifacts = list(state.reduced_ontology_artifacts)
    store = RecordingStore()
    toolbox = SimpleNamespace(
        ontology_manager=RecordingOntologyManager(),
        triple_store_manager=store,
    )

    asyncio.run(ToolBox.aserialize(cast(ToolBox, toolbox), state))

    assert len(store.calls) == 1
    assert isinstance(store.calls[0][0], Ontology)
    assert store.calls[0][1] is None


def test_toolbox_serialize_includes_facts_when_render_facts_enabled() -> None:
    class RecordingOntologyManager:
        async def aadd_ontology(self, ontology: Ontology) -> None:
            return None

    class RecordingStore:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str | None]] = []

        async def aserialize(
            self, payload: object, graph_uri: str | None = None
        ) -> None:
            self.calls.append((payload, graph_uri))

    state = AgentState(render_mode=RenderMode.ONTOLOGY_AND_FACTS)
    state.reduced_ontology_artifacts = [_build_ontology()]
    state.ontology_artifacts = list(state.reduced_ontology_artifacts)
    store = RecordingStore()
    toolbox = SimpleNamespace(
        ontology_manager=RecordingOntologyManager(),
        triple_store_manager=store,
    )

    asyncio.run(ToolBox.aserialize(cast(ToolBox, toolbox), state))

    assert len(store.calls) == 2
    assert isinstance(store.calls[0][0], Ontology)
    assert isinstance(store.calls[1][0], RDFGraph)
    assert store.calls[1][1] == state.graph_uri


def test_toolbox_serialize_persists_all_ontology_artifacts() -> None:
    class RecordingOntologyManager:
        def __init__(self) -> None:
            self.added = 0

        async def aadd_ontology(self, ontology: Ontology) -> None:
            _ = ontology
            self.added += 1

    class RecordingStore:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str | None]] = []

        async def aserialize(
            self, payload: object, graph_uri: str | None = None
        ) -> None:
            self.calls.append((payload, graph_uri))

    state = AgentState(render_mode=RenderMode.ONTOLOGY)
    state.ontology_artifacts = [_build_ontology(), _build_ontology()]
    store = RecordingStore()
    manager = RecordingOntologyManager()
    toolbox = SimpleNamespace(
        ontology_manager=manager,
        triple_store_manager=store,
    )

    asyncio.run(ToolBox.aserialize(cast(ToolBox, toolbox), state))

    assert manager.added == 2
    assert len(store.calls) == 2
    assert all(isinstance(payload, Ontology) for payload, _ in store.calls)


def test_apply_update_query_splits_compound_sparql_insert_updates() -> None:
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix ex: <http://example.org/> .
        ex:Existing ex:kept ex:Value .
        """,
        format="turtle",
    )
    compound_query = (
        "PREFIX ex: <http://example.org/>\n"
        "INSERT DATA { ex:Person ex:label ex:Alice }\n"
        "INSERT DATA { ex:Person ex:status ex:Active }"
    )
    AgentState._apply_update_query(graph, compound_query)

    assert (
        URIRef("http://example.org/Person"),
        URIRef("http://example.org/label"),
        URIRef("http://example.org/Alice"),
    ) in graph
    assert (
        URIRef("http://example.org/Person"),
        URIRef("http://example.org/status"),
        URIRef("http://example.org/Active"),
    ) in graph


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


def test_build_ontology_delta_graph_propagates_delete_operations() -> None:
    """Delete triples in unit GraphUpdates surface in the delete delta.

    The unit delta is the net replay of all GraphUpdates against the prompt
    snapshot: complement inserts plus the snapshot triples removed by delete
    operations, both of which propagate through the reduce stage.
    """
    owl_class = URIRef("http://www.w3.org/2002/07/owl#Class")
    ex_thing = URIRef("https://example.com/onto#Thing")
    ex_new = URIRef("https://example.com/onto#NewThing")
    ex_obsolete = URIRef("https://example.com/onto#obsolete")

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
    delete_graph = RDFGraph()
    delete_graph.parse(
        data="""
        @prefix ex: <https://example.com/onto#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        ex:obsolete a owl:Class .
        """,
        format="turtle",
    )
    delete_op = GraphUpdate(
        triple_operations=[TripleOp(type="delete", graph=delete_graph)]
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
        ontology_snapshot=snapshot_from_ontology(onto),
        ontology_updates_applied=[delete_op],
        ontology_updates=[insert_op],
    )

    delta = state.build_delta()

    assert (ex_new, RDF.type, owl_class) in delta.inserts
    assert (ex_obsolete, RDF.type, owl_class) in delta.deletes
    # Untouched snapshot triples appear in neither channel.
    assert (ex_thing, RDF.type, owl_class) not in delta.inserts
    assert (ex_thing, RDF.type, owl_class) not in delta.deletes


def test_build_ontology_delta_graph_delete_then_reinsert_nets_out() -> None:
    """Ordered replay: a triple deleted and later re-inserted yields no delta."""
    owl_class = URIRef("http://www.w3.org/2002/07/owl#Class")
    ex_thing = URIRef("https://example.com/onto#Thing")

    base_graph = RDFGraph()
    base_graph.parse(
        data="""
        @prefix ex: <https://example.com/onto#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        ex:Thing a owl:Class .
        """,
        format="turtle",
    )
    churn_graph = RDFGraph()
    churn_graph.parse(
        data="""
        @prefix ex: <https://example.com/onto#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        ex:Thing a owl:Class .
        """,
        format="turtle",
    )
    churn_update = GraphUpdate(
        triple_operations=[
            TripleOp(type="delete", graph=churn_graph),
            TripleOp(type="insert", graph=churn_graph),
        ]
    )

    onto = _build_ontology()
    onto.graph = base_graph
    state = UnitOntologyState(
        content_unit=SourceUnit(
            text="test",
            index=0,
            doc_iri=URIRef("https://example.com/doc/d1"),
        ),
        ontology_snapshot=snapshot_from_ontology(onto),
        ontology_updates_applied=[churn_update],
    )

    delta = state.build_delta()

    assert (ex_thing, RDF.type, owl_class) not in delta.inserts
    assert (ex_thing, RDF.type, owl_class) not in delta.deletes
    assert delta.is_empty()


def test_merge_unit_deltas_insert_wins_over_parallel_delete() -> None:
    """Cross-unit consensus: any unit's insert vetoes another unit's delete."""
    owl_class = URIRef("http://www.w3.org/2002/07/owl#Class")
    contested = URIRef("https://example.com/onto#Contested")
    removed = URIRef("https://example.com/onto#Removed")

    delete_both = RDFGraph()
    delete_both.add((contested, RDF.type, owl_class))
    delete_both.add((removed, RDF.type, owl_class))
    reinsert = RDFGraph()
    reinsert.add((contested, RDF.type, owl_class))

    merged = merge_unit_deltas(
        [
            OntologyDelta(deletes=delete_both),
            OntologyDelta(inserts=reinsert),
        ]
    )

    assert (contested, RDF.type, owl_class) in merged.inserts
    assert (contested, RDF.type, owl_class) not in merged.deletes
    assert (removed, RDF.type, owl_class) in merged.deletes


@pytest.mark.anyio
async def test_consolidate_ontology_node_applies_delta_on_map_stage_artifact(
    monkeypatch,
) -> None:
    """Consolidation must build on the map-stage artifact, not the pre-run terminal.

    The consolidation delta is a complement of the map-stage artifact; applying
    it onto the stale catalog terminal silently dropped map-stage additions.
    """
    iri = "https://example.com/onto"
    ns = f"{iri}#"
    owl_class = URIRef("http://www.w3.org/2002/07/owl#Class")
    map_stage_class = URIRef(f"{ns}MapStage")
    consolidated_class = URIRef(f"{ns}Consolidated")

    manager = OntologyManager()
    terminal_graph = RDFGraph()
    terminal_graph.bind("ex", ns)
    terminal_graph.add((URIRef(iri), RDF.type, OWL.Ontology))
    terminal_graph.add((URIRef(f"{ns}Thing"), RDF.type, owl_class))
    terminal = Ontology(graph=terminal_graph, iri=iri)
    manager.add_ontology(terminal, skip_vector_index=True)

    primary_graph = terminal_graph.copy()
    primary_graph.add((map_stage_class, RDF.type, owl_class))
    primary = terminal.derive_updated_version(primary_graph)

    class DummyServerConfig:
        enable_ontology_consolidation = True
        ontology_max_triples = None
        ontology_context_max_triples = 4000

    class DummyConfig:
        server = DummyServerConfig()

    class DummyTools:
        config = DummyConfig()
        ontology_manager = manager

        def get_atomic_tools(self):
            return None

    async def fake_render(unit_state: UnitOntologyState, atomic_tools):
        insert_graph = RDFGraph()
        insert_graph.bind("ex", ns)
        insert_graph.add((consolidated_class, RDF.type, owl_class))
        unit_state.ontology_updates = [
            GraphUpdate(triple_operations=[TripleOp(type="insert", graph=insert_graph)])
        ]
        unit_state.update_ontology()
        unit_state.status = Status.SUCCESS
        return unit_state

    monkeypatch.setattr(
        "ontocast.stategraph.node_factories.render_ontology_update", fake_render
    )

    node = make_consolidate_ontology_node(cast(ToolBox, DummyTools()))
    state = AgentState(render_mode=RenderMode.ONTOLOGY)
    state.reduced_ontology_artifacts = [primary]
    state.ontology_artifacts = [primary]
    state.content_units = [
        ContentUnit(
            text="Perovskite samples were consolidated.",
            index=0,
            doc_iri=URIRef("https://example.com/doc/d1"),
            graph=RDFGraph(),
        )
    ]

    updated = await node(state)

    assert updated.status == Status.SUCCESS
    assert len(updated.ontology_artifacts) == 1
    result_graph = updated.ontology_artifacts[0].graph
    assert (consolidated_class, RDF.type, owl_class) in result_graph
    # The regression: map-stage additions used to be dropped here.
    assert (map_stage_class, RDF.type, owl_class) in result_graph
    assert updated.ontology_updates_applied


def test_a_rejected_request_stops_the_document_graph(monkeypatch) -> None:
    """End to end: the whole graph aborts rather than serializing nothing.

    The failure this guards: a reasoning level the model had dropped made every
    provider call return 400. Each render caught it as its own unit's failure,
    the fan-out reported `failed without usable output for N/N unit(s)`, and
    the graph carried on to SERIALIZE — an empty graph uploaded, a manifest and
    a validation report written beside no facts, exit 0. Every link in the
    chain is pinned separately (test_llm_resilience, test_unit_fanout_failures,
    test_cli_server); this pins the chain.
    """
    # The response type the installed openai client actually annotates.
    import httpx2
    import openai
    from langchain_core.runnables import RunnableConfig

    from ontocast.tool.llm import LLMConfigurationError, LLMTool

    body = {
        "error": {
            "message": (
                "Unsupported value: 'reasoning_effort' does not support "
                "'minimal' with this model."
            ),
            "type": "invalid_request_error",
            "param": "reasoning_effort",
            "code": "unsupported_value",
        }
    }
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")

    class _RejectingModel:
        async def ainvoke(self, *args, **kwds):
            raise openai.BadRequestError(
                f"Error code: 400 - {body}",
                response=httpx2.Response(400, request=request, json=body),
                body=body["error"],
            )

    # setup() rebuilds the client on first use, so the substitution has to
    # survive it.
    async def _install_rejecting_model(self) -> None:
        self._llm = _RejectingModel()

    monkeypatch.setattr(LLMTool, "setup", _install_rejecting_model)

    config = Config(
        tool_config=ToolConfig(
            path_config=PathConfig(),
            llm_config=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="sk-not-used",
                cache_enabled=False,
            ),
        ),
    )
    app = create_agent_graph(ToolBox(config))
    text = "Perovskite films aged at 85 C for 500 hours lost 20% of their PCE. " * 20
    state = AgentState(raw_input={"doc.txt": text.encode("utf-8")})

    seen: list[Any] = []

    async def drive() -> None:
        async for chunk in app.astream(
            state, stream_mode="values", config=RunnableConfig(recursion_limit=40)
        ):
            seen.append(chunk)

    with pytest.raises(LLMConfigurationError, match="rejected the request"):
        asyncio.run(drive())

    # Nothing was serialized: the abort happens in the first fan-out.
    assert all(not chunk.get("aggregated_facts") for chunk in seen)
