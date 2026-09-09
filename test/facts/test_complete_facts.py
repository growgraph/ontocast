"""The insert-only facts completion pass.

Runs after the critic loop, only while the numeric-coverage inventory still
lists a measurement -- a number with its unit -- absent from the unit graph.
Each pass's inserts go through the same per-subject regression check a
critic fix goes through (see ``test_critic_patch_pass.py``), so this file
tests what is specific to the completion pass: it proposes ADD-only fixes,
counts recovered measurements, stops once nothing is missing, and never
touches the ontology loop or a deployment that left the pass budget at zero.
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import RenderMode, Status
from ontocast.onto.model import FactsUnitFinding, FactsUnitFindingKind, TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import (
    FACTS_PHASE,
    _run_completion_passes,
    facts_loop,
    ontology_loop,
)
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import CriticPatchPolicy
from ontocast.toolbox import ToolBox
from test.snapshot_helpers import empty_snapshot

pytestmark = pytest.mark.unit

_SUBJECT = URIRef("http://example.org/sample_1")
_POISON = URIRef("http://example.org/poison")
_MEASURED_VALUE = URIRef("http://example.org/hasNumericValue")
_XSD_DECIMAL = URIRef("http://www.w3.org/2001/XMLSchema#decimal")


def _unit_state(
    text: str = "a shift of 96 meV", graph: RDFGraph | None = None
) -> UnitFactsState:
    unit = ContentUnit(
        text=text,
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=graph if graph is not None else RDFGraph(),
    )
    state = UnitFactsState(content_unit=unit, ontology_snapshot=empty_snapshot())
    state.status = Status.SUCCESS
    return state


def _atomic(*, completion_passes: int = 1) -> AtomicToolBox:
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            facts_completion_passes=completion_passes,
            facts_patch_policy=CriticPatchPolicy(),
            additional_standard_namespaces=(),
            validation_policy=None,
            acceptance_policy=None,
            numeric_coverage_limit=30,
            numeric_coverage_mandatory="off",
            catalog_terms=lambda: set(),
        ),
    )


def _fix(correct: str) -> TripleFix:
    return TripleFix(
        text_fragment="96 meV",
        action="ADD",
        severity="minor",
        correct_value=correct,
        explanation="recovered measurement",
    )


def _no_findings(_state, _atomic) -> list[FactsUnitFinding]:
    return []


def _poison_sensitive_findings(state, _atomic) -> list[FactsUnitFinding]:
    """Mandatory finding iff the poison triple is in the graph -- lets a test
    control the regression check without depending on the real validator."""
    poisoned = any(p == _POISON for _, p, _ in state.content_unit.graph)
    if not poisoned:
        return []
    return [
        FactsUnitFinding(
            kind=FactsUnitFindingKind.UNKNOWN_TERM,
            mandatory=True,
            message="poisoned",
        )
    ]


# --- what one pass does to the graph -----------------------------------------


@pytest.mark.anyio
async def test_a_pass_produces_only_additions(monkeypatch) -> None:
    state = _unit_state()
    added = f'<{_SUBJECT}> <{_MEASURED_VALUE}> "96"^^<{_XSD_DECIMAL}> .'

    async def fake_complete(_state, _atomic, _inventory):
        return [_fix(added)]

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    phase = replace(FACTS_PHASE, collect_findings=_no_findings)

    await _run_completion_passes(state, _atomic(), phase, render_attempt=1)

    assert any(
        s == _SUBJECT and p == _MEASURED_VALUE for s, p, _ in state.content_unit.graph
    )
    attempt = state.attempt_log[-1]
    assert attempt.kind == "completion"
    assert attempt.n_fixes_applied == 1
    assert attempt.n_triples_inserted == 1
    assert attempt.n_triples_deleted == 0
    assert attempt.n_fixes_rolled_back == 0


@pytest.mark.anyio
async def test_a_regressing_insert_is_rolled_back(monkeypatch) -> None:
    """The pass reuses the critic patch's own regression check.

    A fix that manufactures a mandatory finding is undone on its own, the
    same as a critic fix would be -- see ``test_critic_patch_pass.py``.
    """
    state = _unit_state()
    poison = f'<{_SUBJECT}> <{_POISON}> "bad" .'

    async def fake_complete(_state, _atomic, _inventory):
        return [_fix(poison)]

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    phase = replace(FACTS_PHASE, collect_findings=_poison_sensitive_findings)

    await _run_completion_passes(state, _atomic(), phase, render_attempt=1)

    assert not any(
        s == _SUBJECT and p == _POISON for s, p, _ in state.content_unit.graph
    )
    attempt = state.attempt_log[-1]
    assert attempt.n_fixes_applied == 0
    assert attempt.n_fixes_rolled_back == 1
    assert attempt.rolled_back_fixes[0].reason == "new_mandatory"
    assert attempt.n_measurements_recovered == 0


@pytest.mark.anyio
async def test_only_add_fixes_are_kept(monkeypatch) -> None:
    """A non-ADD fix is dropped before it ever reaches the patch machinery.

    Defence in depth: ``complete_facts`` itself already filters this, but the
    unit loop must not trust it -- the pass is insert-only by contract.
    """
    state = _unit_state()

    async def fake_complete(_state, _atomic, _inventory):
        return [
            _fix(f'<{_SUBJECT}> <{_MEASURED_VALUE}> "96"^^<{_XSD_DECIMAL}> .'),
            TripleFix(
                text_fragment="x",
                action="REMOVE",
                severity="minor",
                triple_ids=[1],
                explanation="should be ignored: not addressable by id here",
            ),
        ]

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    phase = replace(FACTS_PHASE, collect_findings=_no_findings)

    await _run_completion_passes(state, _atomic(), phase, render_attempt=1)

    # Only the ADD landed; nothing was deleted (there was nothing to delete
    # by id since no index was handed to the compiler).
    assert state.attempt_log[-1].n_fixes_applied == 1
    assert state.attempt_log[-1].n_triples_deleted == 0


@pytest.mark.anyio
async def test_n_measurements_recovered_counts_precisely(monkeypatch) -> None:
    """One of two missing measurements is addressed; the count reflects that."""
    state = _unit_state(text="a shift of 96 meV and a gap of 12 nm")
    only_the_first = f'<{_SUBJECT}> <{_MEASURED_VALUE}> "96"^^<{_XSD_DECIMAL}> .'

    async def fake_complete(_state, _atomic, _inventory):
        return [_fix(only_the_first)]

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    phase = replace(FACTS_PHASE, collect_findings=_no_findings)

    await _run_completion_passes(
        state, _atomic(completion_passes=1), phase, render_attempt=1
    )

    assert state.attempt_log[-1].n_measurements_recovered == 1


@pytest.mark.anyio
async def test_it_stops_early_once_measurements_are_covered(monkeypatch) -> None:
    """A budget of two passes is not spent once one pass covers everything."""
    state = _unit_state()
    calls = {"n": 0}

    async def fake_complete(_state, _atomic, _inventory):
        calls["n"] += 1
        return [_fix(f'<{_SUBJECT}> <{_MEASURED_VALUE}> "96"^^<{_XSD_DECIMAL}> .')]

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    phase = replace(FACTS_PHASE, collect_findings=_no_findings)

    await _run_completion_passes(
        state, _atomic(completion_passes=2), phase, render_attempt=1
    )

    assert calls["n"] == 1
    assert len([a for a in state.attempt_log if a.kind == "completion"]) == 1


# --- when the pass must not run at all ---------------------------------------


def _facts_tools(*, completion_passes: int) -> object:
    atomic = cast(
        AtomicToolBox,
        SimpleNamespace(
            facts_critic_passes=0,
            facts_patch_policy=CriticPatchPolicy(),
            additional_standard_namespaces=(),
            validation_policy=None,
            acceptance_policy=None,
            numeric_coverage_limit=30,
            numeric_coverage_mandatory="off",
            facts_critic_min_triples=0,
            facts_completion_passes=completion_passes,
            catalog_terms=lambda: set(),
        ),
    )
    return SimpleNamespace(get_atomic_tools=lambda: atomic)


def _context() -> UnitLoopContext:
    return UnitLoopContext.from_agent_state(AgentState(render_mode=RenderMode.FACTS))


def _resolved_context() -> UnitOntologyContext:
    return UnitOntologyContext(
        snapshot=empty_snapshot(), writable_iris=[], confidence=1.0
    )


@pytest.mark.anyio
async def test_the_pass_never_runs_when_the_budget_is_zero(monkeypatch) -> None:
    async def ok_render(state, tools, **kwargs):
        state.status = Status.SUCCESS
        return state

    async def refuse(*_args, **_kwargs):  # pragma: no cover - never reached
        raise AssertionError("completion must not run when the budget is zero")

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "complete_facts", refuse)

    result = await facts_loop(
        _unit_state(),
        cast(ToolBox, _facts_tools(completion_passes=0)),
        _context(),
        pre_resolved_context=_resolved_context(),
    )

    assert not [a for a in result.attempt_log if a.kind == "completion"]


@pytest.mark.anyio
async def test_the_pass_does_not_run_on_the_ontology_loop(monkeypatch) -> None:
    async def render(state, tools, **kwargs):
        state.status = Status.SUCCESS
        return state

    async def resolve(context, tools, unit, *, can_create_vocabulary=False):
        return _resolved_context()

    async def refuse(*_args, **_kwargs):  # pragma: no cover - never reached
        raise AssertionError("completion is a facts-loop mechanism only")

    monkeypatch.setattr(atomic_module, "render_ontology", render)
    monkeypatch.setattr(atomic_module, "resolve_unit_ontology_context", resolve)
    monkeypatch.setattr(atomic_module, "complete_facts", refuse)

    state = UnitOntologyState(
        content_unit=ContentUnit(
            text="a shift of 96 meV",
            index=0,
            doc_iri=URIRef("https://example.com/doc/d1"),
            graph=RDFGraph(),
        ),
        ontology_snapshot=empty_snapshot(),
    )
    tools = cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox,
                SimpleNamespace(
                    ontology_critic_passes=0,
                    ontology_patch_policy=CriticPatchPolicy(),
                    additional_standard_namespaces=(),
                    validation_policy=None,
                    ontology_acceptance_policy=None,
                    # Deliberately no facts_completion_passes attribute: the
                    # ontology phase must never read it (`phase.name == "facts"`
                    # short-circuits first), so its absence proves the guard.
                ),
            )
        ),
    )

    result = await ontology_loop(state, tools, _context())

    assert result.status != Status.FAILED
    assert not [a for a in result.attempt_log if a.kind == "completion"]
