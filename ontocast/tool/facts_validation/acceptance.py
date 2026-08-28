"""What counts as a blocking defect in a rendered unit graph.

The facts loop used to accept a render on ``critique.success or
critique.score > 90`` -- an LLM-assigned 0-100 score compared against a
threshold the model is never shown, from a prompt that never mentions scoring at
all. Measured over the critic calls of the 2026-08 matsci runs, that gate
rejected 28 of 34 renders, with scores min 55 / median 79 / max 98 and 62% of
calls landing in a 70-85 "good, with suggestions" band. An LLM asked to propose
improvements proposes some every time -- it did in 26 of 26 parseable calls --
so the gate was very nearly unconditional.

Worse, it was inverted. ``deterministic_findings`` -- machine-derived,
verifiable, carrying an explicit ``mandatory`` flag -- was computed before every
critic call and injected into the critic's prompt, then played no part in the
decision. A unit with twelve mandatory ``UNKNOWN_TERM`` findings was accepted if
the model said so; a unit with none was rejected if the model said 85. On the
same corpus only 7 of 34 units had any mandatory finding when the critic ran, so
79% were sent for a full re-render on the strength of a number alone. The
expensive action was bound to the unreliable signal and the cheap one to the
reliable signal.

Acceptance is therefore decided here, from defects that can be pointed at.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from ontocast.onto.model import FactsUnitFinding, FactsUnitFindingKind, TripleFix

#: Severity cut for critic-proposed fixes. ``never`` means the critic's
#: severity label is not trusted to block at all and only deterministic
#: findings gate -- the correct setting whenever the label carries no signal.
BlockingFixSeverity = Literal["critical", "important", "never"]


class MaterialDefect(BaseModel):
    """One reason a rendered unit is not acceptable as it stands."""

    source: Literal["finding", "critic_fix"]
    kind: str = Field(description="Finding kind, or the fix's action for a critic fix.")
    message: str


class FactsAcceptancePolicy(BaseModel):
    """Which defects block a rendered unit from leaving the loop.

    Attributes:
        blocking_finding_kinds: Finding kinds that block. ``None`` -- the
            default -- blocks on every finding carrying ``mandatory=True``,
            which is the deterministic validator's own judgement. The explicit
            set exists so an ablation can silence one lane without silencing its
            telemetry, and so a lane found to emit false positives can be
            switched off without a release. That escape hatch is not optional:
            binding acceptance to findings means a systematically unfixable
            finding becomes a permanent per-unit tax, and this codebase has
            already shipped one (a false mandatory ``qudt:numericValue``
            ``UNKNOWN_TERM`` that ordered renders to destroy correct values).
        blocking_fix_severity: The cut applied to critic-proposed fixes.
            ``critical`` is the default because it is the only severity with
            measured discrimination: on the matsci corpus the critic emitted 27
            ``critical`` fixes against 99 ``important``, so gating on
            ``important`` accepts 3 of 26 renders -- worse than the score gate
            it replaces.
    """

    blocking_finding_kinds: frozenset[FactsUnitFindingKind] | None = None
    blocking_fix_severity: BlockingFixSeverity = "critical"

    def blocks_finding(self, finding: FactsUnitFinding) -> bool:
        """True when this deterministic finding must be repaired before exit."""
        if self.blocking_finding_kinds is None:
            return finding.mandatory
        return finding.kind in self.blocking_finding_kinds

    def blocks_fix(self, fix: TripleFix) -> bool:
        """True when this critic-proposed fix must be applied before exit.

        A ``REMOVE`` fix never blocks, whatever its severity. The repair prompt
        it would be rendered into states that a finding is never resolved by
        deleting the statement, so a mandatory REMOVE would contradict the block
        it sits in -- the same shape of contradiction that
        ``shacl_catalog_contradictions`` exists to catch, and the one whose
        fallout the CHANGELOG records as 25 of 58 repair responses deleting
        valid values.
        """
        if self.blocking_fix_severity == "never":
            return False
        if fix.action == "REMOVE":
            return False
        if self.blocking_fix_severity == "critical":
            return fix.severity == "critical"
        return fix.severity in ("critical", "important")


def material_defects(
    findings: Sequence[FactsUnitFinding],
    fixes: Sequence[TripleFix],
    policy: FactsAcceptancePolicy | None = None,
) -> list[MaterialDefect]:
    """Every reason the unit is not acceptable, deterministic evidence first.

    Args:
        findings: Deterministic findings collected against the current graph.
        fixes: Fixes the LLM critic proposed, if it ran. Empty is normal --
            at ``MAX_VISITS=1`` the critic never runs and acceptance rests
            entirely on the findings.
        policy: The deployment's cut. ``None`` uses the defaults.

    Returns:
        Material defects; empty means accept. The list is returned rather than
        a bool so the caller can record *why* a unit was rejected, which the
        score gate never made recordable.
    """
    active = policy if policy is not None else FactsAcceptancePolicy()
    defects = [
        MaterialDefect(
            source="finding", kind=str(finding.kind), message=finding.message
        )
        for finding in findings
        if active.blocks_finding(finding)
    ]
    defects.extend(
        MaterialDefect(
            source="critic_fix",
            kind=fix.action,
            message=fix.explanation,
        )
        for fix in fixes
        if active.blocks_fix(fix)
    )
    return defects


def accept_reason(defects: Sequence[MaterialDefect]) -> str:
    """A short, aggregatable label for why the unit was accepted or not."""
    if not defects:
        return "clean"
    if any(defect.source == "finding" for defect in defects):
        return "mandatory_findings"
    return "critic_critical"
