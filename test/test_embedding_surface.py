"""Tests for the embedding surface: graph builders, lifecycle, in-memory vectors.

Covers what an application integrating OntoCast touches directly, as opposed to
what the HTTP server and CLI drive.
"""

import asyncio
import inspect
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from ontocast.config import Config, ToolConfig
from ontocast.config.settings import PathConfig
from ontocast.integrations.langgraph import text_in_turtle_out
from ontocast.onto.enum import VectorStoreBackend, WorkflowNode
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph import build_agent_graph, create_agent_graph
from ontocast.stategraph.create import _timed
from ontocast.tool.llm import LLMTool
from ontocast.toolbox import ToolBox
from ontocast.util.loop import require_no_running_loop

pytestmark = pytest.mark.unit

ONTOLOGY_TTL = """
@prefix ex: <http://example.org/onto#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://example.org/onto> a owl:Ontology .
ex:Perovskite a owl:Class ; rdfs:label "Perovskite" ;
  rdfs:comment "A crystal structure family used in solar cells" .
ex:Catalyst a owl:Class ; rdfs:label "Catalyst" ;
  rdfs:comment "A substance that speeds up a chemical reaction" .
"""


#: Stands in for a real LLMTool. None of these tests reach the model, and
#: building one would need provider credentials.
STUB_LLM = cast(LLMTool, object())


def _config(tmp_path, backend: VectorStoreBackend) -> Config:
    config = Config.in_memory(tool_config=ToolConfig(path_config=PathConfig()))
    config.tool_config.vector_store.backend = backend
    return config


@pytest.fixture
def toolbox(tmp_path) -> ToolBox:
    return ToolBox(_config(tmp_path, VectorStoreBackend.NONE), llm=STUB_LLM)


# -- public surface --------------------------------------------------------


def test_documented_entry_points_are_exported() -> None:
    """Everything the embedding guide tells users to import must resolve."""
    import ontocast

    for name in (
        "AgentState",
        "Config",
        "ToolBox",
        "build_agent_graph",
        "create_agent_graph",
        "make_ontocast_node",
        "ontocast_tool_diagnostics",
        "ontocast_tool_names",
        "ontocast_tools",
        "run_unit_pipeline",
        "text_in_turtle_out",
    ):
        assert getattr(ontocast, name) is not None
        assert name in ontocast.__all__


# -- graph builders --------------------------------------------------------


def test_build_agent_graph_returns_uncompiled(toolbox: ToolBox) -> None:
    graph = build_agent_graph(toolbox)
    assert isinstance(graph, StateGraph)
    assert not isinstance(graph, CompiledStateGraph)


def test_build_and_create_agree_on_topology(toolbox: ToolBox) -> None:
    built = set(build_agent_graph(toolbox).nodes)
    compiled = set(create_agent_graph(toolbox).get_graph().nodes) - {
        "__start__",
        "__end__",
    }
    assert built == compiled


def test_create_agent_graph_accepts_a_checkpointer(toolbox: ToolBox) -> None:
    """Durable execution requires injecting a saver, which needs the kwarg."""
    compiled = create_agent_graph(
        toolbox, checkpointer=InMemorySaver(), name="ontocast"
    )
    assert isinstance(compiled, CompiledStateGraph)


def test_compiled_graph_can_be_named(toolbox: ToolBox) -> None:
    """An unnamed subgraph shows as 'LangGraph' in a parent graph's traces."""
    assert create_agent_graph(toolbox, name="ontocast").name == "ontocast"


# -- node timing wrapper ---------------------------------------------------


def _fresh_state() -> AgentState:
    return AgentState()


def test_timed_keeps_a_sync_node_sync() -> None:
    """LangGraph only threads nodes it sees as sync; wrapping must not hide that.

    A sync node coerced into a coroutine runs inline on the event loop, which
    blocks it and breaks backends whose sync writes call ``asyncio.run``.
    """

    def node(state: AgentState) -> AgentState:
        return state

    wrapped = _timed("probe", node)
    assert not inspect.iscoroutinefunction(wrapped)

    state = _fresh_state()
    assert wrapped(state) is state
    assert "probe" in state.budget_tracker.node_durations


def test_timed_keeps_an_async_node_async() -> None:
    async def node(state: AgentState) -> AgentState:
        return state

    wrapped = _timed("probe", node)
    assert inspect.iscoroutinefunction(wrapped)

    state = _fresh_state()
    assert asyncio.run(wrapped(state)) is state
    assert "probe" in state.budget_tracker.node_durations


def test_timed_records_a_duration_when_the_node_raises() -> None:
    def node(state: AgentState) -> AgentState:
        raise ValueError("boom")

    state = _fresh_state()
    with pytest.raises(ValueError, match="boom"):
        _timed("probe", node)(state)
    assert "probe" in state.budget_tracker.node_durations


def _runs_in_a_worker_thread(graph: StateGraph, node: WorkflowNode) -> bool:
    """Whether LangGraph will offload ``node`` instead of running it on the loop.

    ``coerce_to_runnable`` gives a sync callable a ``func`` plus an executor
    ``afunc``; an async one gets ``func is None``. The spec types the runnable
    as the node union, so reaching the field needs a cast.
    """
    return cast(Any, graph.nodes[node].runnable).func is not None


def test_sync_nodes_stay_threadable_in_the_built_graph(toolbox: ToolBox) -> None:
    """Regression: SERIALIZE reaches Fuseki's sync ``asyncio.run`` write path."""
    graph = build_agent_graph(toolbox)
    assert _runs_in_a_worker_thread(graph, WorkflowNode.SERIALIZE)
    assert _runs_in_a_worker_thread(graph, WorkflowNode.MERGE_FACTS)
    assert not _runs_in_a_worker_thread(graph, WorkflowNode.RENDER_FACTS)


@pytest.mark.anyio
async def test_sync_node_runs_off_the_event_loop() -> None:
    """End-to-end guard: a sync node may drive coroutines via ``asyncio.run``."""

    def node(state: AgentState) -> AgentState:
        async def write() -> None:
            return None

        asyncio.run(write())
        return state

    graph = StateGraph(AgentState)
    graph.add_node("probe", _timed("probe", node))
    graph.add_edge(START, "probe")
    graph.add_edge("probe", END)

    result = await graph.compile().ainvoke(AgentState())
    assert result is not None


# -- state mapping ---------------------------------------------------------


def test_text_in_turtle_out_maps_both_directions() -> None:
    to_state, from_state = text_in_turtle_out()

    state = to_state({"input": "some source text"})
    assert isinstance(state, AgentState)
    assert state.raw_input == {"input.txt": b"some source text"}

    graph = RDFGraph()
    graph.parse(data=ONTOLOGY_TTL, format="turtle")
    final = AgentState(aggregated_facts=graph)
    delta = from_state(final, {"input": "some source text"})
    assert set(delta) == {"ontology_ttl", "facts_ttl"}
    assert "Perovskite" in delta["facts_ttl"]


def test_text_in_turtle_out_honours_custom_keys() -> None:
    to_state, from_state = text_in_turtle_out(
        text_key="doc", ontology_key="onto", facts_key="triples"
    )
    assert to_state({"doc": "x"}).raw_input == {"doc.txt": b"x"}
    assert set(from_state(AgentState(), {"doc": "x"})) == {"onto", "triples"}


def test_text_in_turtle_out_rejects_a_non_string() -> None:
    to_state, _ = text_in_turtle_out()
    with pytest.raises(TypeError, match="Expected a string"):
        to_state({"input": 42})


# -- lifecycle -------------------------------------------------------------


def test_loop_guard_is_quiet_outside_a_loop() -> None:
    require_no_running_loop("Thing.create", "Thing.acreate")


def test_loop_guard_names_the_async_alternative() -> None:
    async def run() -> None:
        with pytest.raises(RuntimeError, match="Await Thing.acreate"):
            require_no_running_loop("Thing.create", "Thing.acreate")

    asyncio.run(run())


@pytest.mark.anyio
async def test_toolbox_acreate_works_inside_a_running_loop(
    tmp_path, monkeypatch
) -> None:
    """The whole point of acreate: ToolBox(config) raises here, this must not."""
    from ontocast.tool.llm import LLMTool

    async def fake_setup(self) -> None:
        self._llm = object()

    monkeypatch.setattr(LLMTool, "setup", fake_setup)
    tools = await ToolBox.acreate(_config(tmp_path, VectorStoreBackend.NONE))
    assert tools.triple_store_manager is not None
    await tools.aclose()


@pytest.mark.anyio
async def test_toolbox_is_an_async_context_manager(tmp_path, monkeypatch) -> None:
    from ontocast.tool.llm import LLMTool

    async def fake_setup(self) -> None:
        self._llm = object()

    monkeypatch.setattr(LLMTool, "setup", fake_setup)
    async with await ToolBox.acreate(
        _config(tmp_path, VectorStoreBackend.NONE)
    ) as tools:
        assert tools.config is not None


# -- vector store backend selection ----------------------------------------


def test_auto_backend_without_connection_settings_disables_retrieval(
    tmp_path,
) -> None:
    """AUTO must not silently give an unconfigured deployment a vector store."""
    config = _config(tmp_path, VectorStoreBackend.AUTO)
    assert ToolBox(config, llm=STUB_LLM).vector_store is None


def test_none_backend_disables_retrieval(tmp_path) -> None:
    tools = ToolBox(_config(tmp_path, VectorStoreBackend.NONE), llm=STUB_LLM)
    assert tools.vector_store is None
    assert tools.patch_retriever is None
