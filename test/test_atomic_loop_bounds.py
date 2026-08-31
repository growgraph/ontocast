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
from ontocast.onto.enum import FailureStage, RenderMode, Status, WorkflowNode
from ontocast.onto.model import Suggestions, TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import EmptyOntologyContextError
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import facts_loop, ontology_loop
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import CriticPatchPolicy
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


def _tools(critic_passes: int = 1) -> ToolBox:
    return cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox,
                SimpleNamespace(
                    facts_critic_passes=critic_passes,
                    facts_patch_policy=CriticPatchPolicy(),
                    ontology_critic_passes=critic_passes,
                    ontology_patch_policy=CriticPatchPolicy(),
                    additional_standard_namespaces=(),
                    validation_policy=None,
                    acceptance_policy=None,
                    ontology_acceptance_policy=None,
                    numeric_coverage_limit=30,
                    numeric_coverage_mandatory=False,
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

    result = await facts_loop(
        _unit_state(),
        _tools(),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    assert result.status == Status.FAILED
    assert result.failure_stage == FailureStage.FACTS_CRITIQUE


# -- 3c: the two budgets are independent -----------------------------------


@pytest.mark.anyio
async def test_a_rejecting_critic_does_not_escalate_to_another_render(
    monkeypatch,
) -> None:
    """Rejection buys a patch, not a re-extraction.

    A rejecting critic used to fall through to the next ``render_attempt``,
    re-extracting the unit from scratch -- so a unit's worst-case cost grew
    with MAX_VISITS and the answer to "this term is wrong" was "write the whole
    unit again". MAX_VISITS now bounds render *failures* only, so the render
    count is flat in it.
    """
    calls = {"render": 0, "critic": 0}

    async def ok_render(state, tools, **kwargs):
        calls["render"] += 1
        state.status = Status.SUCCESS
        return state

    async def failing_critic(state, tools):
        # Never converges and never requests a search, so the critic loop
        # breaks out and the unit goes to the repair pass.
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

    # One render, one critique, then the patch pass -- whatever the bound.
    assert calls["render"] == 1
    assert calls["critic"] == 1


@pytest.mark.anyio
async def test_the_critic_runs_once_per_configured_pass(monkeypatch) -> None:
    """The pass count is the critic's budget, and MAX_VISITS is not.

    The old loop reached the critic only when a render attempt was left over,
    and could then re-run it within one attempt via the external-evidence path,
    so its worst case was MAX_VISITS squared and its *best* case at the default
    was zero. Both are gone: passes are counted directly.
    """
    calls = {"render": 0, "critic": 0}

    async def ok_render(state, tools, **kwargs):
        calls["render"] += 1
        state.status = Status.SUCCESS
        return state

    async def improving_critic(state, tools):
        # Each pass must actually change something, or the loop converges: a
        # second critique of an unchanged graph is a second identical answer at
        # full price, which is exactly what convergence exists to avoid.
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
        _unit_state(max_visits_per_node=3),
        _tools(critic_passes=2),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    assert calls == {"render": 1, "critic": 2}


@pytest.mark.anyio
async def test_a_pass_that_changes_nothing_ends_the_loop(monkeypatch) -> None:
    """A second critique of an unchanged graph is the same answer, billed twice."""
    calls = {"critic": 0}

    async def ok_render(state, tools, **kwargs):
        state.status = Status.SUCCESS
        return state

    async def barren_critic(state, tools):
        calls["critic"] += 1
        state.status = Status.FAILED
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", barren_critic)

    await facts_loop(
        _unit_state(),
        _tools(critic_passes=5),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    assert calls["critic"] == 1


@pytest.mark.anyio
async def test_zero_passes_costs_exactly_one_call(monkeypatch) -> None:
    """Extraction only, with the deterministic findings still collected."""
    calls = {"render": 0, "critic": 0}

    async def ok_render(state, tools, **kwargs):
        calls["render"] += 1
        state.status = Status.SUCCESS
        return state

    async def critic(state, tools):
        calls["critic"] += 1
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", critic)

    result = await facts_loop(
        _unit_state(max_visits_per_node=3),
        _tools(critic_passes=0),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    assert calls == {"render": 1, "critic": 0}
    # The residual metric sums this field across units, so a unit that skipped
    # the critic must still report what the machine found.
    assert result.deterministic_findings is not None


@pytest.mark.anyio
async def test_max_visits_retries_only_a_failed_render(monkeypatch) -> None:
    calls = {"render": 0}

    async def failing_render(state, tools, **kwargs):
        calls["render"] += 1
        state.status = Status.FAILED
        return state

    monkeypatch.setattr(atomic_module, "render_facts", failing_render)

    await facts_loop(
        _unit_state(max_visits_per_node=3),
        _tools(),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    assert calls["render"] == 3


# -- a deployment fault is not a unit failure -------------------------------


@pytest.mark.anyio
async def test_an_unresolvable_ontology_context_stops_the_run(monkeypatch) -> None:
    """``ONTOLOGY_CONTEXT_REQUIRED`` promises the run stops; this is where.

    The blanket handler caught the error alongside genuine per-unit faults and
    recorded it as a render failure. Since the cause is the deployment, every
    sibling unit hit it too, so the fan-out finished, wrote a zero-triple
    manifest and exited successfully -- the vacuous pass the setting exists to
    prevent, with one traceback per unit burying the cause.
    """

    async def refuse(state, tools, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("render must not run without an ontology context")

    async def empty_context(unit_state, context, tools):
        raise EmptyOntologyContextError("catalog is empty")

    monkeypatch.setattr(atomic_module, "render_facts", refuse)
    monkeypatch.setattr(atomic_module, "_apply_facts_ontology_context", empty_context)

    with pytest.raises(EmptyOntologyContextError):
        await facts_loop(_unit_state(), _tools(), _context())


@pytest.mark.anyio
async def test_the_ontology_loop_reaches_the_renderer_with_an_empty_catalog(
    monkeypatch,
) -> None:
    """The bootstrap journey, at the seam where it used to die.

    An ontology unit resolves its own context, and with no catalog that context
    is empty -- which is the signal ``render_ontology`` reads to mint a fresh
    ontology, not a fault. Asserting the resolver is asked for the exemption
    pins the wiring: the resolver's own exemption is useless if the loop does
    not request it.
    """
    asked: list[bool] = []
    rendered: list[str] = []

    async def resolve(context, tools, unit, *, can_create_vocabulary=False):
        asked.append(can_create_vocabulary)
        return UnitOntologyContext(
            snapshot=empty_snapshot(), writable_iris=[], confidence=0.0
        )

    async def render(state, tools, **kwargs):
        rendered.append("ontology")
        state.set_node_status(WorkflowNode.TEXT_TO_ONTOLOGY, Status.SUCCESS)
        return state

    monkeypatch.setattr(atomic_module, "resolve_unit_ontology_context", resolve)
    monkeypatch.setattr(atomic_module, "render_ontology", render)

    state = UnitOntologyState(
        content_unit=ContentUnit(
            text="Alice works for ACME.",
            index=0,
            doc_iri=URIRef("https://example.com/doc/d1"),
            graph=RDFGraph(),
        ),
        ontology_snapshot=empty_snapshot(),
    )
    result = await ontology_loop(state, _tools(critic_passes=0), _context())

    assert asked == [True]
    assert rendered == ["ontology"]
    assert result.status != Status.FAILED
