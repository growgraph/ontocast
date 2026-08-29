"""Convert critic-proposed fixes into the finding shape the repair pass reads.

The loop had two repair channels that never met. Deterministic findings went to
``_run_finding_driven_repair``, a bounded rewrite-in-place pass costing one
patch render. Critic fixes went somewhere else entirely: they were stashed on
``state.suggestions`` and consumed by the *next full render*, which re-extracted
the unit from scratch under a prompt that also invited it to "proactively
identify and fix additional problems not mentioned in the critique".

So the cheap, contract-bound channel carried the reliable evidence and the
expensive, open-ended one carried the model's opinion. Routing both through the
same findings pipeline is what makes a rejection cost a repair rather than a
re-extraction.
"""

from collections.abc import Sequence

from ontocast.onto.model import FactsUnitFinding, FactsUnitFindingKind, TripleFix
from ontocast.tool.facts_validation.acceptance import FactsAcceptancePolicy


def critic_fixes_to_findings(
    fixes: Sequence[TripleFix],
    policy: FactsAcceptancePolicy | None = None,
) -> list[FactsUnitFinding]:
    """Render critic fixes as findings, blocking ones marked mandatory.

    ``mandatory`` follows the policy's severity cut, with one exception that is
    not configurable: an ``action="REMOVE"`` fix is **never** mandatory. The
    findings block it would be rendered into states that a finding is never
    resolved by deleting the statement, so a mandatory REMOVE would contradict
    the instruction printed directly above it -- the same contradiction shape
    ``shacl_catalog_contradictions`` exists to catch, and one that has already
    caused repair renders to delete valid values wholesale.

    Args:
        fixes: Fixes from the critique report, in the order proposed.
        policy: The deployment's severity cut. ``None`` uses the defaults.

    Returns:
        One finding per fix, advisory unless the policy blocks on it.
    """
    active = policy if policy is not None else FactsAcceptancePolicy()
    findings: list[FactsUnitFinding] = []
    for fix in fixes:
        message = f"{fix.action}: {fix.explanation}".strip()
        if fix.incorrect_value:
            message = f"{message} (currently: {fix.incorrect_value})"
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.CRITIC_FIX,
                mandatory=active.blocks_fix(fix),
                message=message,
                value=fix.incorrect_value or "",
                suggestions=[fix.correct_value] if fix.correct_value else [],
            )
        )
    return findings
