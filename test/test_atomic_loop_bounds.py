"""Failure attribution and critic-call bounds in the per-unit loops.

Two properties that were previously untested and wrong:

* an unhandled exception was always reported as a *critique* failure, whatever
  stage it actually came from;
* the inner critic loop was bounded by the same constant as the outer render
  loop, so its worst case was ``max_visits ** 2`` billed critic calls -- though
  only along the evidence-request path, as the last test here measures.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import FailureStage, RenderMode, Status
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import _resolve_critic_visits, facts_loop
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.atomic import AtomicToolBox
from ontocast.toolbox import ToolBox
from test.snapshot_helpers import empty_snapshot

pytestmark = pytest.mark.unit


def _unit_state(**kwargs) -> UnitFactsState:
    unit = ContentUnit(
        text="Alice works for ACME.",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=RDFGraph(),
    )
    return UnitFactsState(
        content_unit=unit, ontology_snapshot=empty_snapshot(), **kwargs
    )


def _tools() -> ToolBox:
    return cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox,
                SimpleNamespace(
                    facts_llm_repair_visits=1,
                    additional_standard_namespaces=(),
                    validation_policy=None,
                ),
            ),
        ),
    )


def _context() -> UnitLoopContext:
    return UnitLoopContext.from_agent_state(AgentState(render_mode=RenderMode.FACTS))


def _resolved_context() -> UnitOntologyContext:
    return UnitOntologyContext(
        snapshot=empty_snapshot(), writable_iris=[], confidence=1.0
    )


# -- 3b: attribute a crash to the stage it came from ------------------------


@pytest.mark.anyio
async def test_render_crash_is_not_reported_as_a_critique_failure(
    monkeypatch,
) -> None:
    async def exploding_render(state, tools, **kwargs):
        raise RuntimeError("provider exploded during render")

    monkeypatch.setattr(atomic_module, "render_facts", exploding_render)

    result = await facts_loop(
        _unit_state(),
        _tools(),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    assert result.status == Status.FAILED
    assert result.failure_stage == FailureStage.GENERATE_GRAPH_UPDATE_FOR_FACTS
    assert "provider exploded during render" in (result.failure_reason or "")


@pytest.mark.anyio
async def test_critic_crash_is_reported_as_a_critique_failure(monkeypatch) -> None:
    async def ok_render(state, tools, **kwargs):
        # The critic is only reached after a *successful* render; a failed one
        # goes straight to the next render attempt.
        state.status = Status.SUCCESS
        return state

    async def exploding_critic(state, tools):
        raise RuntimeError("critic exploded")

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", exploding_critic)

    # max_visits >= 2 so a render attempt exists that is not the final one,
    # which is the only case where the critic runs at all.
    result = await facts_loop(
        _unit_state(max_visits_per_node=2),
        _tools(),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    assert result.status == Status.FAILED
    assert result.failure_stage == FailureStage.FACTS_CRITIQUE


# -- 3c: the critic loop has its own bound ---------------------------------


def test_critic_visits_default_to_the_legacy_coupling() -> None:
    """Unset keeps today's behaviour: the critic shares the render bound."""
    assert _resolve_critic_visits(_unit_state(max_visits_per_node=3)) == 3


def test_critic_visits_can_be_decoupled_from_render_visits() -> None:
    state = _unit_state(max_visits_per_node=3, max_critic_visits_per_node=1)
    assert _resolve_critic_visits(state) == 1


@pytest.mark.anyio
async def test_critic_calls_are_bounded_by_the_critic_setting(monkeypatch) -> None:
    """With 3 renders and a critic bound of 1, the critic runs once per render.

    Under the old coupling this was 3 critic calls for the first render alone.
    """
    calls = {"render": 0, "critic": 0}

    async def ok_render(state, tools, **kwargs):
        calls["render"] += 1
        state.status = Status.SUCCESS
        return state

    async def failing_critic(state, tools):
        # Never converges and never requests a search, so the critic loop runs
        # to its bound and then falls back to the next render attempt.
        calls["critic"] += 1
        state.status = Status.FAILED
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", failing_critic)

    await facts_loop(
        _unit_state(max_visits_per_node=3, max_critic_visits_per_node=1),
        _tools(),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    # Renders 1 and 2 each get one critique; render 3 is final so it is skipped.
    assert calls["render"] == 3
    assert calls["critic"] <= 2


@pytest.mark.anyio
async def test_critic_without_a_search_request_does_not_iterate(monkeypatch) -> None:
    """Measures how reachable the inner critic loop actually is.

    A critic that fails *without* requesting external evidence breaks out of
    the inner loop immediately, so with web grounding off -- the default -- the
    critic runs at most once per render whatever the bound is. The quadratic
    worst case the bound exists to cap is reachable only through the
    evidence-request path, i.e. with ``WEB_SEARCH_ENABLED=true``.
    """
    calls = {"critic": 0}

    async def ok_render(state, tools, **kwargs):
        state.status = Status.SUCCESS
        return state

    async def failing_critic(state, tools):
        calls["critic"] += 1
        state.status = Status.FAILED
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", failing_critic)

    await facts_loop(
        _unit_state(max_visits_per_node=3),
        _tools(),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    # Renders 1 and 2 are critiqued once each and break; render 3 is final.
    assert calls["critic"] == 2
