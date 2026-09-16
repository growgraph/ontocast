"""One ontology chapter per document instead of one per unit.

Per-unit retrieval gives each unit the smallest chapter -- and a different one,
so no two calls in the fan-out share a prompt prefix and a provider's prefix
cache can serve none of them. The document pays the chapter at full price once
per unit. These pin the alternative: resolve once, share it, and run one unit
first so the cache the others read has actually been written.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import (
    OntologyAssemblyMode,
    OntologyChapterFormat,
    OntologyContextMode,
    OntologyContextScope,
    Status,
)
from ontocast.onto.ontology_condense import TextCaps
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph import context_resolver, node_factories
from ontocast.stategraph.context_resolver import (
    UnitOntologyContext,
    build_unioned_document_ontology_context,
)
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.toolbox import ToolBox

pytestmark = [pytest.mark.anyio, pytest.mark.unit]

UNION_CALLS = "ctx/union_document_ontology.calls"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _unit_graph(term: str) -> RDFGraph:
    return RDFGraph._from_turtle_str(
        f"""
        @prefix ex: <https://example.org/onto#> .
        ex:{term} a ex:Class .
        """
    )


def _state(n_units: int) -> AgentState:
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
    )
    state.content_units = [
        ContentUnit(
            text=f"Unit {index} describes a concept.",
            index=index,
            doc_iri=URIRef("https://example.org/doc/1"),
        )
        for index in range(n_units)
    ]
    # No ontology stage ran, so there is nothing to merge -- the facts-only
    # shape in which every unit otherwise resolves its own context.
    state.reduced_ontology_artifacts = []
    return state


def _tools(*, scope: OntologyContextScope, warmup: int = 0) -> ToolBox:
    tool_config = SimpleNamespace(
        facts_validation=SimpleNamespace(context_from_units=False)
    )
    return cast(
        ToolBox,
        SimpleNamespace(
            config=SimpleNamespace(
                server=SimpleNamespace(
                    parallel_workers=4,
                    ontology_context_max_triples=4000,
                    ontology_chapter_format=OntologyChapterFormat.INHERIT,
                    ontology_text_caps=TextCaps(),
                    ontology_context_scope=scope,
                    fanout_warmup_units=warmup,
                ),
                get_tool_config=lambda: tool_config,
            ),
            shapes_prompt_contract=lambda: ("", (), False),
        ),
    )


def _install_per_unit_resolver(monkeypatch) -> None:
    """Each unit retrieves a distinct term, as vector search would."""

    async def fake_resolve(context, tools, unit, **kwargs):
        from ontocast.onto.ontology_snapshot import OntologySnapshot

        return UnitOntologyContext(
            snapshot=OntologySnapshot.from_graph(
                _unit_graph(f"Term{unit.index}"),
                source_iris=[f"https://example.org/onto/{unit.index}"],
                assembly_mode=OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE,
            ),
            writable_iris=[f"https://example.org/onto/{unit.index}"],
            confidence=1.0,
        )

    monkeypatch.setattr(context_resolver, "resolve_unit_ontology_context", fake_resolve)


async def test_union_contains_every_unit_context(monkeypatch) -> None:
    """Recall-safe by construction: no unit sees less than it would have.

    That is the whole licence for sharing one chapter. If the union could drop
    a term some unit's own retrieval selected, this would be a recall change
    dressed up as a cost change.
    """
    _install_per_unit_resolver(monkeypatch)
    state = _state(4)
    context = UnitLoopContext.from_agent_state(state)

    unioned = await build_unioned_document_ontology_context(
        context, _tools(scope=OntologyContextScope.DOCUMENT), state.content_units
    )

    assert unioned is not None
    terms = {str(term) for triple in unioned.snapshot.graph for term in triple}
    for index in range(4):
        assert f"https://example.org/onto#Term{index}" in terms
    assert context.budget_tracker.counters[UNION_CALLS] == 1


async def test_union_is_stable_across_calls(monkeypatch) -> None:
    """An unstable source order is an unstable prompt, which is the thing
    document scope exists to avoid."""
    _install_per_unit_resolver(monkeypatch)
    state = _state(4)
    tools = _tools(scope=OntologyContextScope.DOCUMENT)

    first = await build_unioned_document_ontology_context(
        UnitLoopContext.from_agent_state(state), tools, state.content_units
    )
    second = await build_unioned_document_ontology_context(
        UnitLoopContext.from_agent_state(state), tools, state.content_units
    )

    assert first is not None and second is not None
    assert first.snapshot.source_iris == second.snapshot.source_iris
    assert first.snapshot.graph.serialize_canonical_turtle() == (
        second.snapshot.graph.serialize_canonical_turtle()
    )


async def test_no_units_resolve_leaves_the_per_unit_path(monkeypatch) -> None:
    """An empty union must not be handed out as a context.

    Every unit receiving an empty snapshot is worse than every unit resolving
    its own: it silently removes the vocabulary instead of narrowing it.
    """

    async def resolve_nothing(context, tools, unit, **kwargs):
        from ontocast.onto.ontology_snapshot import OntologySnapshot

        return UnitOntologyContext(
            snapshot=OntologySnapshot.from_graph(
                RDFGraph(),
                source_iris=[],
                assembly_mode=OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE,
            )
        )

    monkeypatch.setattr(
        context_resolver, "resolve_unit_ontology_context", resolve_nothing
    )
    state = _state(3)

    unioned = await build_unioned_document_ontology_context(
        UnitLoopContext.from_agent_state(state),
        _tools(scope=OntologyContextScope.DOCUMENT),
        state.content_units,
    )

    assert unioned is None


def _install_recording_facts_loop(monkeypatch) -> list[dict]:
    order: list[dict] = []

    async def fake_facts_loop(state, tools, document_context, **kwargs):
        pre_resolved = kwargs.get("pre_resolved_context")
        if pre_resolved is not None:
            state.ontology_snapshot = pre_resolved.snapshot
        order.append(
            {
                "index": state.content_unit.index,
                "event": "start",
                "graph_id": id(state.ontology_snapshot.graph),
            }
        )
        await asyncio.sleep(0)
        order.append({"index": state.content_unit.index, "event": "end"})
        state.content_unit.graph = RDFGraph._from_turtle_str(
            f"""
            @prefix cd: <https://growgraph.dev/> .
            cd:unit{state.content_unit.index} a cd:Thing .
            """
        )
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(node_factories, "facts_loop", fake_facts_loop)
    return order


async def test_document_scope_hands_every_unit_the_same_graph(monkeypatch) -> None:
    _install_per_unit_resolver(monkeypatch)
    order = _install_recording_facts_loop(monkeypatch)
    state = _state(5)

    await node_factories.make_render_facts_node(
        _tools(scope=OntologyContextScope.DOCUMENT)
    )(state)

    graph_ids = {row["graph_id"] for row in order if row["event"] == "start"}
    assert len(graph_ids) == 1, "one chapter per document, not one per unit"


async def test_unit_scope_is_unchanged(monkeypatch) -> None:
    """The default must still resolve per unit -- this is opt-in."""
    _install_per_unit_resolver(monkeypatch)
    order = _install_recording_facts_loop(monkeypatch)
    state = _state(3)

    await node_factories.make_render_facts_node(
        _tools(scope=OntologyContextScope.UNIT)
    )(state)

    starts = [row for row in order if row["event"] == "start"]
    assert len(starts) == 3
    # Nothing was pre-resolved, so each unit carries its own empty seed graph.
    assert len({row["graph_id"] for row in starts}) == 3


async def test_warmup_runs_the_first_unit_before_the_rest_start(monkeypatch) -> None:
    """A prefix cache is populated by a request that has already completed.

    Without this the fan-out issues every call before any of them has written
    the entry the others would read, and all of them miss.
    """
    _install_per_unit_resolver(monkeypatch)
    order = _install_recording_facts_loop(monkeypatch)
    state = _state(4)

    await node_factories.make_render_facts_node(
        _tools(scope=OntologyContextScope.DOCUMENT, warmup=1)
    )(state)

    events = [(row["index"], row["event"]) for row in order]
    assert events[0] == (0, "start")
    assert events[1] == (0, "end"), "unit 0 must finish before any sibling starts"
    assert {index for index, event in events[2:] if event == "start"} == {1, 2, 3}


async def test_without_warmup_every_unit_starts_before_any_finishes(
    monkeypatch,
) -> None:
    """The default shape, kept: warm-up costs a serialized call, so it is opt-in."""
    _install_per_unit_resolver(monkeypatch)
    order = _install_recording_facts_loop(monkeypatch)
    state = _state(4)

    await node_factories.make_render_facts_node(
        _tools(scope=OntologyContextScope.DOCUMENT, warmup=0)
    )(state)

    first_end = next(i for i, row in enumerate(order) if row["event"] == "end")
    starts_before_first_end = sum(
        1 for row in order[:first_end] if row["event"] == "start"
    )
    assert starts_before_first_end == 4


async def test_every_unit_still_produces_output_with_warmup(monkeypatch) -> None:
    """The warm-up must not drop or duplicate a unit."""
    _install_per_unit_resolver(monkeypatch)
    _install_recording_facts_loop(monkeypatch)
    state = _state(6)

    await node_factories.make_render_facts_node(
        _tools(scope=OntologyContextScope.DOCUMENT, warmup=2)
    )(state)

    assert len(state.facts_units) == 6
    assert [unit.index for unit in state.facts_units] == list(range(6))


async def test_a_failing_unit_after_the_warmup_is_named_by_its_own_index(
    monkeypatch, caplog
) -> None:
    """A warm-up splits the fan-out into two gathers.

    Each gather sees its tasks positionally, so without an offset the second
    batch names its failures by position within the batch -- reporting unit 0
    for a unit that is not unit 0, which sends whoever reads the log to the
    wrong content unit.
    """
    _install_per_unit_resolver(monkeypatch)

    async def failing_facts_loop(state, tools, document_context, **kwargs):
        if state.content_unit.index == 3:
            raise RuntimeError("unit 3 blew up")
        state.content_unit.graph = RDFGraph._from_turtle_str(
            f"""
            @prefix cd: <https://growgraph.dev/> .
            cd:unit{state.content_unit.index} a cd:Thing .
            """
        )
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(node_factories, "facts_loop", failing_facts_loop)
    state = _state(5)

    with caplog.at_level("ERROR"):
        await node_factories.make_render_facts_node(
            _tools(scope=OntologyContextScope.DOCUMENT, warmup=1)
        )(state)

    assert any("Unit 3 raised" in record.message for record in caplog.records), [
        record.message for record in caplog.records
    ]
    assert len(state.facts_units) == 4
