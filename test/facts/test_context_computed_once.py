"""The facts fan-out must build its ontology context once, not once per unit.

The merged document ontology is a pure function of ``reduced_ontology_artifacts``,
which the ontology stage freezes upstream. Resolving it inside each unit task
meant N identical full rdflib merges plus 2N graph copies -- synchronous work on
the event loop, so it stalled every *other* unit's in-flight provider call and
collapsed the fan-out to roughly serial. These tests pin the fixed shape.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import (
    OntologyChapterFormat,
    OntologyContextMode,
    RetrievalMetric,
    Status,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_condense import TextCaps
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph import node_factories
from ontocast.toolbox import ToolBox

pytestmark = [pytest.mark.anyio, pytest.mark.unit]

MERGE_CALLS = "ctx/merge_document_ontology.calls"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _ontology(iri: str, local: str) -> Ontology:
    return Ontology(
        iri=iri,
        graph=RDFGraph._from_turtle_str(
            f"""
            @prefix ex: <{iri}#> .
            ex:{local} a ex:Class .
            ex:{local} ex:label ex:Value .
            """
        ),
    )


def _state(n_units: int) -> AgentState:
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
    )
    state.content_units = [
        ContentUnit(
            text=f"Unit {index} describes a concept.",
            index=index,
            doc_iri=URIRef("https://example.org/doc/1"),
        )
        for index in range(n_units)
    ]
    state.reduced_ontology_artifacts = [
        _ontology("https://example.org/onto/a", "Alpha"),
        _ontology("https://example.org/onto/b", "Beta"),
    ]
    return state


def _tools(*, context_from_units: bool = False) -> ToolBox:
    tool_config = SimpleNamespace(
        facts_validation=SimpleNamespace(context_from_units=context_from_units)
    )
    return cast(
        ToolBox,
        SimpleNamespace(
            config=SimpleNamespace(
                server=SimpleNamespace(
                    parallel_workers=4,
                    max_critic_visits_per_node=None,
                    ontology_context_max_triples=4000,
                    ontology_chapter_format=OntologyChapterFormat.INHERIT,
                    ontology_text_caps=TextCaps(),
                ),
                get_tool_config=lambda: tool_config,
            ),
            shapes_prompt_contract=lambda: ("", (), False),
        ),
    )


def _install_recording_facts_loop(monkeypatch) -> list[dict]:
    """Replace facts_loop with a stub that records what each unit received."""
    seen: list[dict] = []

    async def fake_facts_loop(state, tools, document_context, **kwargs):
        pre_resolved = kwargs.get("pre_resolved_context")
        if pre_resolved is not None:
            state.ontology_snapshot = pre_resolved.snapshot
            state.ontology_patch_sources = list(pre_resolved.patch_sources)
        seen.append(
            {
                "snapshot_id": id(state.ontology_snapshot),
                "graph_id": id(state.ontology_snapshot.graph),
                "pre_resolved": pre_resolved,
            }
        )
        # A unit with an empty graph is treated as producing no usable output
        # and dropped, so emit one triple to exercise the normal reduce path.
        state.content_unit.graph = RDFGraph._from_turtle_str(
            f"""
            @prefix cd: <https://growgraph.dev/> .
            cd:unit{state.content_unit.index} a cd:Thing .
            """
        )
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(node_factories, "facts_loop", fake_facts_loop)
    return seen


@pytest.mark.parametrize("n_units", [1, 5])
async def test_merged_context_built_once_regardless_of_unit_count(
    monkeypatch, n_units: int
) -> None:
    _install_recording_facts_loop(monkeypatch)
    state = _state(n_units)

    await node_factories.make_render_facts_node(_tools())(state)

    # Equality across unit counts is the strong form: it fails for any per-unit
    # reintroduction whatever the constant happens to be.
    assert state.budget_tracker.counters[MERGE_CALLS] == 1
    assert len(state.facts_units) == n_units


async def test_every_unit_receives_the_same_snapshot_object(monkeypatch) -> None:
    seen = _install_recording_facts_loop(monkeypatch)
    state = _state(5)

    await node_factories.make_render_facts_node(_tools())(state)

    # Identity, not equality: a reintroduced deepcopy would still compare equal
    # while costing a full graph copy per unit.
    assert len({entry["snapshot_id"] for entry in seen}) == 1
    assert len({entry["graph_id"] for entry in seen}) == 1
    assert all(entry["pre_resolved"] is not None for entry in seen)


async def test_shared_snapshot_is_not_mutated_by_the_fan_out(monkeypatch) -> None:
    seen = _install_recording_facts_loop(monkeypatch)
    state = _state(4)

    await node_factories.make_render_facts_node(_tools())(state)

    shared = seen[0]["pre_resolved"].snapshot
    before_triples = set(shared.graph)
    before_namespaces = sorted(shared.graph.namespaces())

    # Sharing is only safe while the snapshot stays read-only; if a unit ever
    # writes to it, every sibling silently sees the change mid-flight.
    assert set(shared.graph) == before_triples
    assert sorted(shared.graph.namespaces()) == before_namespaces
    assert len(before_triples) > 0


async def test_merge_and_validate_reuse_the_fan_out_context(monkeypatch) -> None:
    _install_recording_facts_loop(monkeypatch)
    state = _state(3)

    await node_factories.make_render_facts_node(_tools())(state)
    assert len(state.facts_ontology_context) > 0

    # Downstream aggregation reads the cached graph rather than merging again.
    ontology_graph, _metadata = node_factories._facts_aggregation_inputs(state)
    assert ontology_graph is state.facts_ontology_context
    assert state.budget_tracker.counters[MERGE_CALLS] == 1


async def test_facts_only_run_still_resolves_per_unit(monkeypatch) -> None:
    """With no ontology stage there is nothing to merge; units resolve their own."""
    seen = _install_recording_facts_loop(monkeypatch)
    state = _state(3)
    state.reduced_ontology_artifacts = []

    await node_factories.make_render_facts_node(_tools())(state)

    assert all(entry["pre_resolved"] is None for entry in seen)
    assert len(state.facts_units) == 3


def _install_per_unit_resolving_facts_loop(monkeypatch, snapshot) -> list[int]:
    """Facts loop stub standing in for a facts-only run's per-unit resolution.

    Mirrors ``_apply_facts_ontology_context``: with no merged context handed
    down, each unit resolves its own and the result lands on the unit state as
    a distinct object with the same contributing catalog IRIs.
    """
    snapshot_ids: list[int] = []

    async def fake_facts_loop(state, tools, document_context, **kwargs):
        assert kwargs.get("pre_resolved_context") is None
        state.ontology_snapshot = snapshot.model_copy(deep=True)
        snapshot_ids.append(id(state.ontology_snapshot))
        state.content_unit.graph = RDFGraph._from_turtle_str(
            f"""
            @prefix cd: <https://growgraph.dev/> .
            cd:unit{state.content_unit.index} a cd:Thing .
            """
        )
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(node_factories, "facts_loop", fake_facts_loop)
    return snapshot_ids


def _catalog_snapshot() -> OntologySnapshot:
    return OntologySnapshot(
        graph=RDFGraph._from_turtle_str(
            """
            @prefix ex: <https://example.org/onto/catalog#> .
            ex:Sample a ex:Class .
            ex:hasValue a ex:Property .
            """
        ),
        source_iris=["https://example.org/onto/catalog"],
    )


async def test_facts_only_context_stays_empty_by_default(monkeypatch) -> None:
    """The flag is off by default, so today's empty-graph behaviour is kept."""
    _install_per_unit_resolving_facts_loop(monkeypatch, _catalog_snapshot())
    state = _state(3)
    state.reduced_ontology_artifacts = []

    await node_factories.make_render_facts_node(_tools())(state)

    assert len(state.facts_ontology_context) == 0
    ontology_graph, _metadata = node_factories._facts_aggregation_inputs(state)
    assert len(ontology_graph) == 0


async def test_facts_only_context_from_units_reaches_merge_and_validate(
    monkeypatch,
) -> None:
    """With the flag on, the context the units rendered against is handed down.

    Both consumers of ``_facts_aggregation_inputs`` are affected: the
    aggregator's guards read the type declarations, and the gate stops
    reporting every extracted term as outside the catalog.
    """
    snapshot_ids = _install_per_unit_resolving_facts_loop(
        monkeypatch, _catalog_snapshot()
    )
    state = _state(3)
    state.reduced_ontology_artifacts = []

    await node_factories.make_render_facts_node(_tools(context_from_units=True))(state)

    # Distinct snapshot objects per unit, one union downstream.
    assert len(set(snapshot_ids)) == 3
    assert len(state.facts_ontology_context) == 2
    ontology_graph, _metadata = node_factories._facts_aggregation_inputs(state)
    assert ontology_graph is state.facts_ontology_context
    assert state.retrieval_metrics[RetrievalMetric.ONTOLOGY_SNAPSHOT_TRIPLES] == 2


def test_union_deduplicates_snapshots_sharing_source_iris() -> None:
    """One catalog ontology resolved by N units must be merged once.

    Unioning unfiltered would pay a full rdflib merge per unit for a graph that
    is already present.
    """
    snapshot = _catalog_snapshot()
    results = [
        (
            index,
            SimpleNamespace(ontology_snapshot=snapshot.model_copy(deep=True)),
            "",
            [],
            None,
        )
        for index in range(4)
    ]
    union = node_factories._union_unit_ontology_context(cast(list, results))
    assert len(union) == len(snapshot.graph)


def test_union_keeps_distinct_sources() -> None:
    first = _catalog_snapshot()
    second = OntologySnapshot(
        graph=RDFGraph._from_turtle_str(
            """
            @prefix other: <https://example.org/onto/other#> .
            other:Thing a other:Class .
            """
        ),
        source_iris=["https://example.org/onto/other"],
    )
    results = [
        (0, SimpleNamespace(ontology_snapshot=first), "", [], None),
        (1, SimpleNamespace(ontology_snapshot=second), "", [], None),
    ]
    union = node_factories._union_unit_ontology_context(cast(list, results))
    assert len(union) == len(first.graph) + len(second.graph)


def test_union_keeps_namespace_bindings() -> None:
    """The gate reads declared namespaces off this graph, not just its triples.

    ``NON_CATALOG_VOCABULARY`` decides what counts as catalog vocabulary from
    the bindings, so a union that carried triples but dropped prefixes would
    silently change which terms are reported as outside the catalog.
    """
    snapshot = _catalog_snapshot()
    results = [(0, SimpleNamespace(ontology_snapshot=snapshot), "", [], None)]
    union = node_factories._union_unit_ontology_context(cast(list, results))
    assert dict(union.namespaces()).keys() >= {"ex"}


def test_union_of_nothing_is_empty() -> None:
    assert len(node_factories._union_unit_ontology_context([])) == 0
