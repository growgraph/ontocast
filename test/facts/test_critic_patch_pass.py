"""A critic pass is undone when it leaves the unit worse than it found it.

The rollback rules were learned from a repair render that answered "this term is
wrong" by deleting the statement: the finding went away because the data went
away. A compiled patch can do the same thing, so the same three signals apply --
plus one the render path could not have: a pass is now transparent enough that
*creating* mandatory findings is detectable and undone.
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import Literal, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.model import FactsUnitFinding, FactsUnitFindingKind, TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.triple_index import build_triple_index
from ontocast.onto.unit_states import UnitFactsState
from ontocast.stategraph.atomic import FACTS_PHASE, _apply_critic_patch
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import CriticPatchPolicy

pytestmark = pytest.mark.unit

_EX_PREDICATE = URIRef("http://example.org/redShiftContribution")
_LABEL = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
_SUBJECT = URIRef(f"{DEFAULT_IRI}sample_1")
_VALUE = URIRef(f"{DEFAULT_IRI}value_1")


def _unit_state() -> UnitFactsState:
    graph = RDFGraph()
    graph.add((_SUBJECT, _EX_PREDICATE, _VALUE))
    graph.add((_SUBJECT, _LABEL, Literal("sample")))
    graph.add((_VALUE, _LABEL, Literal("96 meV")))
    unit = ContentUnit(
        text="a shift of 96 meV",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=graph,
    )
    state = UnitFactsState(content_unit=unit)
    state.status = Status.SUCCESS
    return state


def _atomic() -> AtomicToolBox:
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            facts_critic_passes=1,
            facts_patch_policy=CriticPatchPolicy(),
            additional_standard_namespaces=(),
            validation_policy=None,
            acceptance_policy=None,
        ),
    )


def _fix(action, *, triple_ids=None, correct="") -> TripleFix:
    return TripleFix(
        text_fragment="96 meV",
        action=action,
        severity="critical",
        triple_ids=triple_ids or [],
        correct_value=correct,
        explanation="test fix",
    )


def _run(state, findings=None, *, mandatory_before=0, phase=FACTS_PHASE, atomic=None):
    tools = atomic or _atomic()
    collected = findings if findings is not None else []
    phase = replace(phase, collect_findings=lambda _state, _tools: list(collected))
    return _apply_critic_patch(
        state,
        tools,
        phase,
        render_attempt=1,
        pass_index=1,
        mandatory_before=mandatory_before,
    )


def _finding(mandatory: bool = True) -> FactsUnitFinding:
    return FactsUnitFinding(
        kind=FactsUnitFindingKind.UNKNOWN_TERM,
        mandatory=mandatory,
        message="unknown term",
    )


def test_a_pass_that_only_deletes_is_rolled_back() -> None:
    """The statement is gone, so the finding is gone. Nothing was repaired."""
    state = _unit_state()
    index = build_triple_index(state.content_unit.graph)
    state.prompt_triple_index = index
    doomed = [tid for tid, (_, p, _) in index.by_id.items() if p == _EX_PREDICATE]
    state.suggestions.actionable_fixes = [_fix("REMOVE", triple_ids=doomed)]

    outcome = _run(state, [_finding()], mandatory_before=1)

    assert outcome.rolled_back is True
    assert (_SUBJECT, _EX_PREDICATE, _VALUE) in state.content_unit.graph
    assert state.critic_fixes_applied == 0


def test_a_pass_that_rewrites_in_place_is_kept() -> None:
    """Shrinking is only a problem when nothing was written back."""
    state = _unit_state()
    index = build_triple_index(state.content_unit.graph)
    state.prompt_triple_index = index
    target = [tid for tid, (_, p, _) in index.by_id.items() if p == _EX_PREDICATE]
    state.suggestions.actionable_fixes = [
        _fix(
            "REPLACE",
            triple_ids=target,
            correct=f"<{_SUBJECT}> <http://example.org/shift> <{_VALUE}> .",
        )
    ]

    outcome = _run(state, [], mandatory_before=1)

    assert outcome.rolled_back is False
    assert outcome.applied == 1
    assert (_SUBJECT, URIRef("http://example.org/shift"), _VALUE) in (
        state.content_unit.graph
    )


def test_a_pass_that_creates_mandatory_findings_is_rolled_back() -> None:
    """Undone whole, however much else it fixed.

    The render path could not see this: it compared finding counts and graph
    size, and a pass that fixed two things while breaking three scored as
    progress.
    """
    state = _unit_state()
    before = set(state.content_unit.graph)
    state.suggestions.actionable_fixes = [
        _fix("ADD", correct=f'<{_SUBJECT}> <http://example.org/new> "x" .')
    ]

    outcome = _run(state, [_finding(), _finding()], mandatory_before=1)

    assert outcome.rolled_back is True
    assert set(state.content_unit.graph) == before


def test_a_growing_pass_is_never_flagged() -> None:
    state = _unit_state()
    state.suggestions.actionable_fixes = [
        _fix("ADD", correct=f'<{_SUBJECT}> <http://example.org/new> "x" .')
    ]

    outcome = _run(state, [], mandatory_before=0)

    assert outcome.rolled_back is False
    assert outcome.applied == 1


def test_an_applied_fix_stops_being_a_request() -> None:
    """Otherwise the next pass is asked to redo what this one already did."""
    state = _unit_state()
    applied = _fix("ADD", correct=f'<{_SUBJECT}> <http://example.org/new> "x" .')
    unusable = _fix("REMOVE")
    state.suggestions.actionable_fixes = [applied, unusable]

    _run(state, [])

    assert state.suggestions.actionable_fixes == [unusable]


def test_the_index_is_cleared_so_a_later_pass_cannot_reuse_it() -> None:
    """Ids are paired with the critique they were issued for."""
    state = _unit_state()
    state.prompt_triple_index = build_triple_index(state.content_unit.graph)

    _run(state, [])

    assert state.prompt_triple_index is None


def test_a_pass_with_nothing_applicable_reports_no_change() -> None:
    state = _unit_state()
    state.suggestions.actionable_fixes = [_fix("REMOVE")]

    outcome = _run(state, [])

    assert outcome.applied == 0
    assert outcome.residual == 1
    assert outcome.converged is True


def test_a_patch_that_clears_every_defect_flips_the_unit_to_success() -> None:
    """The critic verdict predates its own patch.

    Acceptance was decided on the pre-patch findings, so a unit whose patch
    then resolved everything still left the loop FAILED -- and the reduce
    counted it as salvaged from a non-converged loop. Post-patch, status must
    reflect the graph that actually ships.
    """
    state = _unit_state()
    state.set_failure(FACTS_PHASE.critic_stage, "pre-patch rejection")
    state.suggestions.actionable_fixes = [
        _fix("ADD", correct=f'<{_SUBJECT}> <http://example.org/new> "x" .')
    ]

    # Post-patch findings are clean.
    _run(state, [], mandatory_before=1)

    assert state.status == Status.SUCCESS
    assert state.failure_stage is None
    assert state.failure_reason is None


def test_residual_mandatory_findings_keep_the_unit_failed() -> None:
    state = _unit_state()
    state.suggestions.actionable_fixes = [
        _fix("ADD", correct=f'<{_SUBJECT}> <http://example.org/new> "x" .')
    ]

    _run(state, [_finding()], mandatory_before=1)

    assert state.status == Status.FAILED
    assert state.failure_stage == FACTS_PHASE.critic_stage
    assert "material defect" in (state.failure_reason or "")


def test_a_rolled_back_patch_keeps_the_pre_patch_verdict() -> None:
    """Rollback restores the graph, so the pre-patch acceptance is what the
    re-evaluation recomputes -- an accepted unit stays accepted."""
    state = _unit_state()
    index = build_triple_index(state.content_unit.graph)
    state.prompt_triple_index = index
    doomed = [tid for tid, (_, p, _) in index.by_id.items() if p == _EX_PREDICATE]
    state.suggestions.actionable_fixes = [_fix("REMOVE", triple_ids=doomed)]

    outcome = _run(state, [], mandatory_before=0)

    assert outcome.rolled_back is True
    assert state.status == Status.SUCCESS


def test_an_empty_no_update_pass_reevaluates_status_too() -> None:
    """The no-update branch is the common path for an accepted render whose
    critique compiled to nothing; it must not preserve a stale FAILED."""
    state = _unit_state()
    state.set_failure(FACTS_PHASE.critic_stage, "pre-patch rejection")
    state.suggestions.actionable_fixes = []

    _run(state, [])

    assert state.status == Status.SUCCESS
    assert state.failure_stage is None
