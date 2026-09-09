"""The two per-unit budgets must observably reach the loop, and stay separate.

An arm comparing ``--max-visits 1`` against ``--max-visits 2`` was later shown
by call accounting to be an A/A comparison: the critic had never run in either,
because it was gated behind a spare render slot and nothing recorded the
effective setting. The budgets are now independent -- ``max_visits`` retries a
*failed* render, ``FACTS_CRITIC_PASSES`` buys review-and-patch passes -- and
these tests pin both ends of the chain that makes an arm auditable.
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
from ontocast.onto.model import Suggestions, TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import facts_loop
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import CriticPatchPolicy
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


def _tools(critic_passes: int = 1) -> ToolBox:
    return cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox,
                SimpleNamespace(
                    facts_critic_passes=critic_passes,
                    facts_patch_policy=CriticPatchPolicy(),
                    additional_standard_namespaces=(),
                    validation_policy=None,
                    acceptance_policy=None,
                    numeric_coverage_limit=30,
                    numeric_coverage_mandatory=False,
                    facts_critic_min_triples=0,
                    facts_completion_passes=0,
                    catalog_terms=lambda: set(),
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


@pytest.mark.anyio
@pytest.mark.parametrize("max_visits", [1, 2, 3])
@pytest.mark.parametrize("critic_passes", [0, 1, 2])
async def test_the_two_budgets_do_not_trade_against_each_other(
    monkeypatch, max_visits: int, critic_passes: int
) -> None:
    """One successful render, and exactly the passes that were paid for.

    The old loop entangled these: the critic ran only if a render attempt was
    left over, so "enable the critic" meant "authorise a second full
    extraction", and raising the render bound silently raised the critic bound
    too. Neither is true now.
    """
    calls = {"render": 0, "critic": 0}

    async def ok_render(state, tools, **kwargs):
        calls["render"] += 1
        state.status = Status.SUCCESS
        return state

    async def improving_critic(state, tools):
        calls["critic"] += 1
        state.suggestions = Suggestions(
            actionable_fixes=[
                TripleFix(
                    text_fragment="Alice",
                    action="ADD",
                    severity="important",
                    correct_value=(
                        f"<https://example.com/s{calls['critic']}> "
                        f'<https://example.com/p> "v" .'
                    ),
                    explanation="add one statement",
                )
            ]
        )
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", improving_critic)

    await facts_loop(
        _unit_state(max_visits_per_node=max_visits),
        _tools(critic_passes=critic_passes),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    assert calls == {"render": 1, "critic": critic_passes}


@pytest.mark.anyio
@pytest.mark.parametrize("max_visits", [1, 2, 3, 5])
async def test_a_rejecting_critic_costs_the_same_at_any_render_bound(
    monkeypatch, max_visits: int
) -> None:
    """A rejection buys a patch, never a re-extraction."""
    calls = {"render": 0, "critic": 0}

    async def ok_render(state, tools, **kwargs):
        calls["render"] += 1
        state.status = Status.SUCCESS
        return state

    async def rejecting_critic(state, tools):
        calls["critic"] += 1
        state.status = Status.FAILED
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", rejecting_critic)

    await facts_loop(
        _unit_state(max_visits_per_node=max_visits),
        _tools(critic_passes=1),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    assert calls == {"render": 1, "critic": 1}
