"""Acceptance is decided by verifiable defects, not by the critic's score.

The loop used to accept a render on ``critique.success or critique.score > 90``
while ignoring ``deterministic_findings`` entirely -- the machine-derived,
pointable evidence it had already computed and put in the critic's own prompt.
That is an inversion: the expensive action (a full re-render) hung on an
uncalibrated LLM number, and the cheap one (a rewrite-in-place repair) on the
reliable signal.

The two assertions that matter here are the two halves of that inversion:
a clean unit with a *low* score is accepted, and a unit with a mandatory
finding is rejected however enthusiastic the critic was.
"""

from typing import Literal

import pytest

from ontocast.onto.model import FactsUnitFinding, FactsUnitFindingKind, TripleFix
from ontocast.tool.facts_validation import (
    FactsAcceptancePolicy,
    accept_reason,
    material_defects,
)

pytestmark = pytest.mark.unit


def _finding(*, mandatory: bool) -> FactsUnitFinding:
    return FactsUnitFinding(
        kind=FactsUnitFindingKind.UNKNOWN_TERM,
        mandatory=mandatory,
        message="ex:redShift does not exist in its ontology",
    )


def _fix(
    severity: Literal["critical", "important", "minor"],
    *,
    action: Literal["ADD", "REMOVE", "REPLACE"] = "REPLACE",
) -> TripleFix:
    return TripleFix(
        text_fragment="a shift of 96 meV",
        action=action,
        severity=severity,
        explanation="use the canonical scalar property",
    )


def test_a_clean_unit_is_accepted_however_low_the_score() -> None:
    """No verifiable defect means accept. The score is not consulted at all."""
    assert material_defects([], []) == []
    assert accept_reason([]) == "clean"


def test_a_mandatory_finding_rejects_however_high_the_score() -> None:
    """The single most important assertion in this revision.

    A unit carrying a mandatory ``UNKNOWN_TERM`` used to be accepted outright
    if the model returned ``success=True``.
    """
    defects = material_defects([_finding(mandatory=True)], [])

    assert len(defects) == 1
    assert defects[0].source == "finding"
    assert accept_reason(defects) == "mandatory_findings"


def test_an_advisory_finding_does_not_block() -> None:
    """NUMERIC_COVERAGE fires on nearly every unit of numeric prose."""
    assert material_defects([_finding(mandatory=False)], []) == []


@pytest.mark.parametrize(
    ("cut", "severity", "blocks"),
    [
        ("critical", "critical", True),
        ("critical", "important", False),
        ("critical", "minor", False),
        ("important", "critical", True),
        ("important", "important", True),
        ("important", "minor", False),
        ("never", "critical", False),
        ("never", "important", False),
    ],
)
def test_the_severity_cut_is_configurable(
    cut: Literal["critical", "important", "never"],
    severity: Literal["critical", "important", "minor"],
    blocks: bool,
) -> None:
    policy = FactsAcceptancePolicy(blocking_fix_severity=cut)
    defects = material_defects([], [_fix(severity)], policy)

    assert bool(defects) is blocks


def test_a_remove_fix_never_blocks_whatever_its_severity() -> None:
    """A mandatory REMOVE would contradict the prompt block it lands in.

    ``format_findings_for_prompt`` states that a finding is never resolved by
    deleting the statement. Rendering a mandatory REMOVE alongside that
    instruction is the contradiction class whose fallout the CHANGELOG records
    as 25 of 58 repair responses deleting valid values.
    """
    policy = FactsAcceptancePolicy(blocking_fix_severity="important")

    assert material_defects([], [_fix("critical", action="REMOVE")], policy) == []


def test_blocking_kinds_can_silence_one_lane_without_silencing_others() -> None:
    """The escape hatch for a lane found to emit false positives.

    Binding acceptance to findings means a systematically unfixable finding
    becomes a permanent per-unit tax, and this codebase has shipped one -- a
    false mandatory ``qudt:numericValue`` UNKNOWN_TERM that ordered renders to
    destroy correct values. Switching a lane off must not need a release.
    """
    quarantined = FactsUnitFinding(
        kind=FactsUnitFindingKind.QUARANTINED_LITERAL,
        mandatory=True,
        message="invalid xsd:double literal",
    )
    policy = FactsAcceptancePolicy(
        blocking_finding_kinds=frozenset({FactsUnitFindingKind.QUARANTINED_LITERAL})
    )

    assert material_defects([_finding(mandatory=True)], [], policy) == []
    assert len(material_defects([quarantined], [], policy)) == 1


def test_findings_are_reported_before_critic_fixes() -> None:
    """Deterministic evidence first: it is the half that can be pointed at."""
    defects = material_defects([_finding(mandatory=True)], [_fix("critical")])

    assert [defect.source for defect in defects] == ["finding", "critic_fix"]
    assert accept_reason(defects) == "mandatory_findings"
