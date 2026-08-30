"""``--max-visits`` must observably reach the unit loops.

A run comparing ``--max-visits 1`` against
``--max-visits 2``, but LLM-call accounting later showed the critic never ran
in the second arm — the two runs were an A/A comparison, and nothing recorded
the effective setting. These tests pin the two ends of the chain that make a
future arm auditable: the batch entry path writes the flag into
``AgentState``, and a unit loop at ``max_visits=2`` actually spends a critic
call.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.api.process_helpers import expand_input_to_states
from ontocast.config import Config
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyContextMode, RenderMode, Status
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import facts_loop
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.atomic import AtomicToolBox
from ontocast.toolbox import ToolBox
from test.snapshot_helpers import empty_snapshot

pytestmark = pytest.mark.unit


def test_batch_entry_path_writes_max_visits_onto_state(
    tmp_path: pathlib.Path,
) -> None:
    document = tmp_path / "doc.txt"
    document.write_text("Alice works for ACME.")
    states = expand_input_to_states(
        document,
        config=Config(),
        head_chunks=None,
        ontology_context_mode_value=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        tenant=None,
        project=None,
        max_visits=2,
    )
    assert [state.max_visits for state in states] == [2]


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


def _tools(repair_visits: int = 0) -> ToolBox:
    return cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox,
                SimpleNamespace(
                    facts_llm_repair_visits=repair_visits,
                    additional_standard_namespaces=(),
                    validation_policy=None,
                    acceptance_policy=None,
                ),
            ),
        ),
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("max_visits", "expected_critic_calls"),
    [(1, 0), (2, 1)],
)
async def test_critic_spends_a_call_exactly_when_max_visits_allows(
    monkeypatch, max_visits: int, expected_critic_calls: int
) -> None:
    """With no repair budget, the critic runs only when a render slot is spare.

    ``FACTS_LLM_REPAIR_VISITS=0`` is the one configuration where a verdict has
    nowhere to go: no repair render to feed, and at ``max_visits=1`` no second
    extraction either. See the companion test below for the default case, where
    the critic runs at 1 visit because the repair lane can act on it.

    This is billing-visible in production (``criticise_facts`` is a provider
    call), which is how the silent A/A run was eventually detected — so the
    loop-level guarantee is asserted on call count, not on log output.
    """
    critic_calls = 0

    async def ok_render(state, tools, **kwargs):
        state.status = Status.SUCCESS
        return state

    async def converging_critic(state, tools):
        nonlocal critic_calls
        critic_calls += 1
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", converging_critic)

    context = UnitLoopContext.from_agent_state(AgentState(render_mode=RenderMode.FACTS))
    resolved = UnitOntologyContext(
        snapshot=empty_snapshot(), writable_iris=[], confidence=1.0
    )
    result = await facts_loop(
        _unit_state(max_visits_per_node=max_visits),
        _tools(),
        context,
        pre_resolved_context=resolved,
    )

    assert result.status == Status.SUCCESS
    assert critic_calls == expected_critic_calls


@pytest.mark.anyio
@pytest.mark.parametrize("max_visits", [1, 2])
async def test_critic_runs_at_one_visit_when_a_repair_budget_exists(
    monkeypatch, max_visits: int
) -> None:
    """The critic no longer needs a spare render slot to be worth calling.

    It used to be skipped whenever the current render was the last allowed, on
    the reasoning that a critique which cannot drive a retry is wasted. That is
    spent: a verdict now feeds the tiered repair lane, which compiles
    mechanical fixes with no LLM call and sends the rest to a bounded repair
    render. At the shipped default (``MAX_VISITS=1``,
    ``FACTS_LLM_REPAIR_VISITS=1``) the critic therefore runs — which is what
    makes it measurable without paying for a second full extraction.
    """
    critic_calls = 0

    async def ok_render(state, tools, **kwargs):
        state.status = Status.SUCCESS
        return state

    async def converging_critic(state, tools):
        nonlocal critic_calls
        critic_calls += 1
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", converging_critic)

    context = UnitLoopContext.from_agent_state(AgentState(render_mode=RenderMode.FACTS))
    resolved = UnitOntologyContext(
        snapshot=empty_snapshot(), writable_iris=[], confidence=1.0
    )
    result = await facts_loop(
        _unit_state(max_visits_per_node=max_visits),
        _tools(repair_visits=1),
        context,
        pre_resolved_context=resolved,
    )

    assert result.status == Status.SUCCESS
    assert critic_calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize("max_visits", [1, 2, 3])
async def test_a_converging_critic_costs_two_calls_at_any_bound(
    monkeypatch, max_visits: int
) -> None:
    """Raising ``MAX_VISITS`` buys nothing when the critic accepts the render.

    The loop returns as soon as a critique succeeds, so the *happy path* costs
    one render plus one critique however high the bound is set — and exactly
    one call at a bound of 1, where the critic is skipped outright. So the
    production cost of ``MAX_VISITS=3`` is not a fixed multiple: it is driven
    entirely by how often the critic rejects, which makes it a property of the
    corpus rather than of the setting.
    """
    calls: list[str] = []

    async def ok_render(state, tools, **kwargs):
        calls.append("render")
        state.status = Status.SUCCESS
        return state

    async def converging_critic(state, tools):
        calls.append("critic")
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", converging_critic)

    context = UnitLoopContext.from_agent_state(AgentState(render_mode=RenderMode.FACTS))
    result = await facts_loop(
        _unit_state(max_visits_per_node=max_visits),
        _tools(),
        context,
        pre_resolved_context=UnitOntologyContext(
            snapshot=empty_snapshot(), writable_iris=[], confidence=1.0
        ),
    )

    assert result.status == Status.SUCCESS
    expected = ["render"] if max_visits == 1 else ["render", "critic"]
    assert calls == expected


@pytest.mark.anyio
@pytest.mark.parametrize("max_visits", [1, 2, 3, 5])
async def test_a_rejecting_critic_costs_the_same_at_any_bound(
    monkeypatch, max_visits: int
) -> None:
    """Worst-case per-unit calls no longer grow with MAX_VISITS.

    The ledger used to be ``2 * max_visits - 1``: a rejecting critic fell
    through to the next ``render_attempt``, which re-extracted the whole unit.
    So the answer to "this one term is wrong" was another full render, and
    raising the bound bought more of them. Rejection now routes the critic's
    fixes into the same bounded rewrite-in-place repair the deterministic
    findings use, and the outer loop retries only on *render failure*.

    At ``max_visits=1`` the critic is skipped entirely (there is no second
    render for it to inform), so that arm costs one render. Above 1 the cost is
    one render plus one critique, flat.

    Web grounding is off here (the default): a critic that rejects *without*
    requesting evidence breaks the inner loop immediately, so the nominal
    ``max_visits ** 2`` worst case is unreachable on this path. The repair pass
    is free in this fixture (``facts_llm_repair_visits=0``).
    """
    calls: list[str] = []

    async def ok_render(state, tools, **kwargs):
        calls.append("render")
        state.status = Status.SUCCESS
        return state

    async def rejecting_critic(state, tools):
        calls.append("critic")
        state.status = Status.FAILED
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", rejecting_critic)

    context = UnitLoopContext.from_agent_state(AgentState(render_mode=RenderMode.FACTS))
    await facts_loop(
        _unit_state(max_visits_per_node=max_visits),
        _tools(),
        context,
        pre_resolved_context=UnitOntologyContext(
            snapshot=empty_snapshot(), writable_iris=[], confidence=1.0
        ),
    )

    expected_critics = 0 if max_visits == 1 else 1
    assert calls.count("render") == 1
    assert calls.count("critic") == expected_critics
    assert len(calls) == 1 + expected_critics
