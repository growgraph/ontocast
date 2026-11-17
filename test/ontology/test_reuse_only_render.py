"""A document whose plan mints nothing makes no ontology update call.

The plan tells every render of the document which terms it may mint. When that
list is empty the render can only restate existing terms, so it is answered
with the empty patch such a render returns.
"""

import importlib
import typing
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDFS, Literal, URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.model import (
    GraphUpdateRenderReport,
    OntologyPlan,
    OntologyPlanTerm,
)
from ontocast.onto.prefixes import build_prefix_table
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.tool.atomic import AtomicToolBox
from test.snapshot_helpers import empty_snapshot

render_module = importlib.import_module("ontocast.agent.render_ontology")

pytestmark = pytest.mark.unit

_CLASS = URIRef("https://example.com/onto#Sample")


def _term(
    decision: typing.Literal["mint", "reuse"], local_name: str
) -> OntologyPlanTerm:
    namespace = (
        "https://example.com/onto#"
        if decision == "mint"
        else "https://example.com/ext#"
    )
    return OntologyPlanTerm(
        kind="class",
        label=local_name,
        local_name=local_name,
        iri=f"{namespace}{local_name}",
        decision=decision,
    )


def _state(plan: OntologyPlan | None) -> UnitOntologyState:
    graph = RDFGraph()
    graph.add((_CLASS, RDFS.label, Literal("Sample")))
    state = UnitOntologyState(
        content_unit=ContentUnit(
            text="a sample of perovskite",
            index=0,
            doc_iri=URIRef("https://example.com/doc/d1"),
            graph=RDFGraph(),
        ),
        ontology_snapshot=empty_snapshot(),
        ontology_plan=plan,
    )
    state.working_graph = graph
    return state


async def _llm_tool(_budget_tracker) -> SimpleNamespace:
    return SimpleNamespace()


def _tools() -> AtomicToolBox:
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            get_llm_tool=_llm_tool,
            prefix_table=lambda: build_prefix_table([]),
            web_grounding_enabled_for_node=lambda _node: False,
            ontology_insert_only=False,
        ),
    )


def _count_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    async def fake_call_llm_with_retry(*args, **kwargs):
        calls.append(1)
        return GraphUpdateRenderReport()

    monkeypatch.setattr(render_module, "call_llm_with_retry", fake_call_llm_with_retry)
    return calls


def test_reuse_only_means_terms_and_none_minted() -> None:
    assert render_module.plan_is_reuse_only(OntologyPlan(terms=[_term("reuse", "Ext")]))
    assert not render_module.plan_is_reuse_only(
        OntologyPlan(terms=[_term("reuse", "Ext"), _term("mint", "New")])
    )
    # No terms at all: the render runs without a plan block and may mint.
    assert not render_module.plan_is_reuse_only(OntologyPlan())
    assert not render_module.plan_is_reuse_only(None)


@pytest.mark.anyio
async def test_a_reuse_only_plan_makes_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_calls(monkeypatch)
    state = _state(OntologyPlan(terms=[_term("reuse", "Ext")]))

    result = await render_module.render_ontology_update(state, _tools())

    assert calls == []
    assert result.status == Status.SUCCESS
    # The same state an empty render leaves: one empty update, graph unchanged.
    assert len(result.all_updates) == 1
    assert result.all_updates[0].count_total_triples()[1] == 0
    assert (_CLASS, RDFS.label, Literal("Sample")) in result.working_graph
    assert result.budget_tracker.counters[render_module.RENDER_SKIPPED_REUSE_ONLY] == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "plan",
    [
        OntologyPlan(terms=[_term("reuse", "Ext"), _term("mint", "New")]),
        OntologyPlan(),
        None,
    ],
    ids=["mints-a-term", "no-terms", "no-plan"],
)
async def test_other_plans_render_as_before(
    monkeypatch: pytest.MonkeyPatch, plan: OntologyPlan | None
) -> None:
    calls = _count_calls(monkeypatch)

    result = await render_module.render_ontology_update(_state(plan), _tools())

    assert calls == [1]
    assert result.status == Status.SUCCESS
    assert render_module.RENDER_SKIPPED_REUSE_ONLY not in result.budget_tracker.counters
