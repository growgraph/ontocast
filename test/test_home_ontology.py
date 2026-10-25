"""The document's home ontology: resolved once, minted by code, shared by the fan-out."""

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.config import (
    Config,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    PathConfig,
    ToolConfig,
    WebSearchConfig,
)
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import LLMGraphFormat, RenderMode, Status, WorkflowNode
from ontocast.onto.identity import OntologyIdentity
from ontocast.onto.model import FreshOntologyRenderReport, OntologyIdentityProposal
from ontocast.onto.ontology_header import DivergedOntologyError, mint_ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState, InputDocument
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.create import build_agent_graph, create_agent_graph
from ontocast.stategraph.routing import route_after_tag_or_chunk
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.llm import LLMTool
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.toolbox import ToolBox
from test.ontology_helpers import DOMAIN, next_version, stamped_ontology
from test.snapshot_helpers import empty_snapshot

pytestmark = pytest.mark.unit

resolve_home_ontology_module = importlib.import_module(
    "ontocast.agent.resolve_home_ontology"
)
render_ontology_module = importlib.import_module("ontocast.agent.render_ontology")
create_module = importlib.import_module("ontocast.stategraph.create")
unit_loops = importlib.import_module("ontocast.stategraph.atomic")

OWL_CLASS = URIRef("http://www.w3.org/2002/07/owl#Class")


def _terms(namespace: str, *names: str) -> RDFGraph:
    graph = RDFGraph()
    for name in names:
        graph.add((URIRef(f"{namespace}{name}"), RDF.type, OWL_CLASS))
        graph.add((URIRef(f"{namespace}{name}"), RDFS.label, Literal(name)))
    return graph


def _proposal(ontology_id: str) -> OntologyIdentityProposal:
    return OntologyIdentityProposal(
        ontology_id=ontology_id,
        title=f"{ontology_id} ontology",
        description="What the document is about.",
    )


def _document_state(text: str = "Alice works for ACME.") -> AgentState:
    return AgentState(
        render_mode=RenderMode.ONTOLOGY,
        current_domain=DOMAIN,
        content_units=[
            ContentUnit(
                text=text, index=0, doc_iri=URIRef("https://example.org/doc/d1")
            )
        ],
    )


def _tools_with_catalog(manager: OntologyManager) -> ToolBox:
    async def get_llm_tool(_budget_tracker: object) -> object:
        return object()

    return cast(
        ToolBox, SimpleNamespace(ontology_manager=manager, get_llm_tool=get_llm_tool)
    )


def _fake_identity_call(monkeypatch, proposals: dict[str, str]) -> list[str]:
    """Answer the identity call with the proposal whose key appears in the text.

    Returns the list the fake appends each prompt text to, one entry per call.
    """
    calls: list[str] = []

    async def fake_call_llm_with_retry(**kwargs) -> OntologyIdentityProposal:
        text = str(kwargs["prompt_kwargs"]["text"])
        calls.append(text)
        for marker, ontology_id in proposals.items():
            if marker in text:
                return _proposal(ontology_id)
        raise AssertionError(f"no proposal for the excerpt {text!r}")

    monkeypatch.setattr(
        resolve_home_ontology_module, "call_llm_with_retry", fake_call_llm_with_retry
    )
    return calls


# --- resolve_home_ontology ---------------------------------------------------


@pytest.mark.anyio
async def test_resolve_home_ontology_mints_a_new_id_under_current_domain(
    monkeypatch,
) -> None:
    calls = _fake_identity_call(monkeypatch, {"Alice": "tmi"})
    state = _document_state()

    await resolve_home_ontology_module.resolve_home_ontology(
        state, _tools_with_catalog(OntologyManager())
    )

    assert len(calls) == 1
    assert state.home_ontology == OntologyIdentity(
        ontology_id="tmi",
        iri="https://example.org/tmi",
        title="tmi ontology",
        description="What the document is about.",
    )
    assert state.status == Status.SUCCESS


@pytest.mark.anyio
async def test_resolve_home_ontology_extends_a_catalog_ontology(monkeypatch) -> None:
    """An id the catalog knows resolves to that ontology's head, not a new one."""
    _fake_identity_call(monkeypatch, {"Alice": "tmi"})
    manager = OntologyManager()
    head = stamped_ontology(
        _terms("https://example.org/tmi#", "Sample"),
        ontology_id="tmi",
        title="Test ontology",
        description="Stored before this run.",
    )
    manager.add_ontology(head, skip_vector_index=True)
    state = _document_state()

    await resolve_home_ontology_module.resolve_home_ontology(
        state, _tools_with_catalog(manager)
    )

    home = state.home_ontology
    assert home is not None
    assert home.iri == head.iri
    assert home.ontology_id == "tmi"
    # The head's own title wins over what the model proposed.
    assert home.title == "Test ontology"
    assert home.description == "Stored before this run."


@pytest.mark.anyio
async def test_resolve_home_ontology_refuses_an_iri_held_under_another_id(
    monkeypatch,
) -> None:
    _fake_identity_call(monkeypatch, {"Alice": "tmi"})
    manager = OntologyManager()
    manager.add_ontology(
        stamped_ontology(
            _terms("https://example.org/tmi#", "Sample"),
            ontology_id="other",
            iri="https://example.org/tmi",
        ),
        skip_vector_index=True,
    )

    with pytest.raises(ValueError, match="already in the catalog under another id"):
        await resolve_home_ontology_module.resolve_home_ontology(
            _document_state(), _tools_with_catalog(manager)
        )


@pytest.mark.anyio
async def test_resolve_home_ontology_is_skipped_for_facts_only_runs(
    monkeypatch,
) -> None:
    calls = _fake_identity_call(monkeypatch, {"Alice": "tmi"})
    state = _document_state()
    state.render_mode = RenderMode.FACTS
    state.home_ontology = OntologyIdentity(
        ontology_id="stale", iri="https://example.org/stale", title="Stale"
    )

    await resolve_home_ontology_module.resolve_home_ontology(
        state, _tools_with_catalog(OntologyManager())
    )

    assert state.home_ontology is None
    assert calls == []


@pytest.mark.anyio
async def test_resolve_home_ontology_refuses_a_diverged_ontology(monkeypatch) -> None:
    _fake_identity_call(monkeypatch, {"Alice": "tmi"})
    manager = OntologyManager()
    namespace = "https://example.org/tmi#"
    root = stamped_ontology(_terms(namespace, "Sample"), ontology_id="tmi")
    left = next_version(root, _terms(namespace, "Sample", "Left"))
    right = next_version(root, _terms(namespace, "Sample", "Right"))
    for version in (root, left, right):
        manager.add_ontology(version, skip_vector_index=True)

    with pytest.raises(DivergedOntologyError):
        await resolve_home_ontology_module.resolve_home_ontology(
            _document_state(), _tools_with_catalog(manager)
        )


# --- the fan-out on the compiled graph ----------------------------------------


def _config() -> Config:
    return Config(
        tool_config=ToolConfig(
            path_config=PathConfig(),
            llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name=OllamaModel.LLAMA3_1,
                base_url="http://localhost:11434",
            ),
        ),
    )


def _fake_fanout_graph(monkeypatch, proposals: dict[str, str]) -> tuple[Any, list[str]]:
    """The document graph with every model call faked.

    Conversion is a no-op, chunking splits the document on blank lines, the
    identity call answers from ``proposals``, every fresh unit mints one term
    under its home ontology, and the nodes after the ontology fan-out pass the
    state through. Returns the compiled graph and the identity-call log.
    """
    calls = _fake_identity_call(monkeypatch, proposals)

    def fake_convert(state: AgentState, tools: ToolBox) -> AgentState:
        return state

    async def fake_chunk(state: AgentState, tools: ToolBox) -> AgentState:
        assert state.input_document is not None
        parts = state.input_document.data.decode("utf-8").split("\n\n")
        state.content_units = [
            ContentUnit(text=part, index=index, doc_iri=state.doc_iri)
            for index, part in enumerate(parts)
        ]
        state.status = Status.SUCCESS
        return state

    async def fake_render(
        state: UnitOntologyState, tools: AtomicToolBox, **kwargs
    ) -> UnitOntologyState:
        home = state.home_ontology
        assert home is not None
        ontology = mint_ontology(
            home, _terms(home.namespace, f"Term{state.content_unit.index}")
        )
        state.fresh_ontology = ontology
        state.working_graph = ontology.graph.copy()
        state.writable_iris = [ontology.iri]
        state.assembly_anchor_iri = ontology.iri
        state.status = Status.SUCCESS
        return state

    async def fake_critic(state: UnitOntologyState, tools) -> UnitOntologyState:
        state.status = Status.SUCCESS
        return state

    async def fake_resolve(_state, _tools, _unit, **_kwargs) -> UnitOntologyContext:
        return UnitOntologyContext(
            snapshot=empty_snapshot(), writable_iris=[], confidence=1.0
        )

    def passthrough_factory(_tools: ToolBox):
        async def node(state: AgentState) -> AgentState:
            return state

        return node

    async def passthrough_serialize(state: AgentState, tools: ToolBox) -> AgentState:
        return state

    monkeypatch.setattr(create_module, "convert_document", fake_convert)
    monkeypatch.setattr(create_module, "chunk_text", fake_chunk)
    monkeypatch.setattr(create_module, "serialize", passthrough_serialize)
    for factory in (
        "make_normalize_ontology_node",
        "make_consolidate_ontology_node",
        "make_structural_check_node",
        "make_consistency_critic_node",
    ):
        monkeypatch.setattr(create_module, factory, passthrough_factory)
    monkeypatch.setattr(unit_loops, "render_ontology", fake_render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", fake_critic)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", fake_resolve)

    return create_agent_graph(ToolBox(_config())), calls


def _document(text: str) -> AgentState:
    return AgentState(
        input_document=InputDocument(filename="doc.txt", data=text.encode("utf-8")),
        render_mode=RenderMode.ONTOLOGY,
        current_domain=DOMAIN,
    )


def test_fresh_units_of_one_document_share_one_home_ontology(monkeypatch) -> None:
    """N fresh units: one identity call, one root artifact under the home IRI."""
    app, calls = _fake_fanout_graph(monkeypatch, {"alpha": "alpha"})

    final = asyncio.run(app.ainvoke(_document("alpha one\n\nalpha two\n\nalpha three")))

    assert len(calls) == 1
    home = final["home_ontology"]
    assert home is not None
    assert home.iri == f"{DOMAIN}/alpha"
    artifacts = final["ontology_artifacts"]
    assert [artifact.iri for artifact in artifacts] == [home.iri]
    terms = {
        str(s) for s, _, _ in artifacts[0].graph.triples((None, RDF.type, OWL_CLASS))
    }
    assert terms == {f"{home.namespace}Term{index}" for index in range(3)}
    assert final["ontology_reduce_metrics"]["fresh_ontologies_merged"] == 2


def test_concurrent_documents_never_share_a_home_ontology(monkeypatch) -> None:
    app, calls = _fake_fanout_graph(monkeypatch, {"alpha": "alpha", "beta": "beta"})

    async def drive() -> list[dict[str, Any]]:
        return await asyncio.gather(
            app.ainvoke(_document("alpha one\n\nalpha two")),
            app.ainvoke(_document("beta one\n\nbeta two")),
        )

    finals = asyncio.run(drive())

    assert len(calls) == 2
    homes = {final["home_ontology"].iri for final in finals}
    assert homes == {f"{DOMAIN}/alpha", f"{DOMAIN}/beta"}
    for final in finals:
        assert [artifact.iri for artifact in final["ontology_artifacts"]] == [
            final["home_ontology"].iri
        ]


# --- render_ontology_fresh -------------------------------------------------------


class _FakeLLMProvider:
    async def get_llm_tool(self, budget_tracker: object) -> LLMTool:
        return cast(LLMTool, object())


def _atomic_tools() -> AtomicToolBox:
    return AtomicToolBox(
        llm_provider=_FakeLLMProvider(),
        web_search_config=WebSearchConfig(enabled=False),
    )


def _unit_state(home: OntologyIdentity | None) -> UnitOntologyState:
    return UnitOntologyState(
        content_unit=ContentUnit(
            text="Samples age.", index=0, doc_iri=URIRef("https://example.org/doc/d1")
        ),
        ontology_snapshot=empty_snapshot(),
        home_ontology=home,
    )


@pytest.mark.anyio
async def test_render_ontology_fresh_drops_a_model_written_header(monkeypatch) -> None:
    home = OntologyIdentity(
        ontology_id="tmi", iri="https://example.org/tmi", title="Test ontology"
    )
    answer = RDFGraph()
    answer.parse(
        data="""
        @prefix tmi: <https://example.org/tmi#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        tmi:Ontology a owl:Ontology ; rdfs:label "x" .
        tmi:Sample a owl:Class ; rdfs:label "Sample" .
        """,
        format="turtle",
    )

    async def fake_call_llm_with_retry(**_kwargs) -> FreshOntologyRenderReport:
        return FreshOntologyRenderReport(graph=answer)

    monkeypatch.setattr(
        render_ontology_module, "call_llm_with_retry", fake_call_llm_with_retry
    )

    result = await render_ontology_module.render_ontology_fresh(
        _unit_state(home), _atomic_tools()
    )

    assert result.status == Status.SUCCESS
    fresh = result.fresh_ontology
    assert fresh is not None
    assert fresh.iri == home.iri
    assert result.writable_iris == [home.iri]
    assert list(fresh.graph.triples((None, RDF.type, OWL.Ontology))) == []
    assert (
        list(fresh.graph.triples((URIRef(f"{home.namespace}Ontology"), None, None)))
        == []
    )
    assert (URIRef(f"{home.namespace}Sample"), RDF.type, OWL_CLASS) in fresh.graph
    assert result.budget_tracker.counters["ontology/model_header_triples_dropped"] == 2


@pytest.mark.anyio
async def test_render_ontology_fresh_needs_a_home_ontology() -> None:
    with pytest.raises(RuntimeError, match="home ontology"):
        await render_ontology_module.render_ontology_fresh(
            _unit_state(None), _atomic_tools()
        )


# --- the wire schema ---------------------------------------------------------------


def _schema_keys(node: Any) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {
            key for value in node.values() for key in _schema_keys(value)
        }
    if isinstance(node, list):
        return {key for value in node for key in _schema_keys(value)}
    return set()


def test_fresh_render_report_schema_carries_no_identity_or_lineage() -> None:
    """The model renders terms only; identity and versions are written by code."""
    keys = _schema_keys(FreshOntologyRenderReport.model_json_schema())
    assert keys.isdisjoint(
        {"hash", "parent_hashes", "created_at", "current_domain", "iri"}
    )


# --- routing ----------------------------------------------------------------------


def test_route_after_chunk_goes_to_the_home_ontology_step() -> None:
    assert (
        route_after_tag_or_chunk(AgentState(render_mode=RenderMode.ONTOLOGY))
        == WorkflowNode.RESOLVE_HOME_ONTOLOGY
    )
    assert (
        route_after_tag_or_chunk(AgentState(render_mode=RenderMode.ONTOLOGY_AND_FACTS))
        == WorkflowNode.RESOLVE_HOME_ONTOLOGY
    )
    assert (
        route_after_tag_or_chunk(AgentState(render_mode=RenderMode.FACTS))
        == WorkflowNode.RENDER_FACTS
    )


def test_agent_graph_resolves_the_home_ontology_before_the_fan_out() -> None:
    workflow = build_agent_graph(ToolBox(_config()))

    assert WorkflowNode.RESOLVE_HOME_ONTOLOGY in workflow.nodes
    assert (
        WorkflowNode.RESOLVE_HOME_ONTOLOGY,
        WorkflowNode.RENDER_ONTOLOGY_UPDATE,
    ) in workflow.edges


# --- the prompt chapter -----------------------------------------------------------


def test_ontology_chapter_shows_terms_without_the_version_header() -> None:
    stamped = stamped_ontology(
        _terms("https://example.org/tmi#", "Sample"),
        ontology_id="tmi",
        title="Test ontology",
    )
    assert (URIRef(stamped.iri), OWL.versionIRI, None) in stamped.graph

    chapter = get_graph_format_profile(LLMGraphFormat.TURTLE).format_ontology_chapter(
        stamped.graph
    )

    assert "versionIRI" not in chapter
    assert "vann:" not in chapter
    assert "preferredNamespace" not in chapter
    assert "Sample" in chapter
