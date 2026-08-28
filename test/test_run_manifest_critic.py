"""The run manifest must carry the evidence for the critic's own decisions.

The facts loop accepts a render on ``critique.success or critique.score > 90``,
a score the model is asked for with no rubric and no statement of the threshold.
Nothing recorded that score, so establishing that the gate was miscalibrated
(median 79, 82% rejected, 62% landing in 70-85) required mining the LLM disk
cache -- which only worked because caching happened to be enabled.
"""

import pytest

from ontocast.onto.model import FactsLoopAttempt
from ontocast.onto.run_manifest import summarize_facts_loop

pytestmark = pytest.mark.unit


def _critic(score: float, *, success: bool, **severities: int) -> FactsLoopAttempt:
    return FactsLoopAttempt(
        kind="critic",
        score=score,
        success=success,
        severity_counts=dict(severities),
        n_actionable_fixes=sum(severities.values()),
    )


def test_no_critic_calls_summarizes_to_zero() -> None:
    """The MAX_VISITS=1 default: the critic never runs, and that is not an error."""
    summary = summarize_facts_loop(
        {0: [FactsLoopAttempt(kind="render"), FactsLoopAttempt(kind="llm_repair")]}
    )

    assert summary.calls == 0
    assert summary.accepted == 0
    assert summary.score_median is None
    assert summary.score_histogram == {}


def test_scores_and_severities_are_summarized_across_units() -> None:
    summary = summarize_facts_loop(
        {
            0: [
                FactsLoopAttempt(kind="render"),
                _critic(55, success=False, critical=2, minor=1),
            ],
            1: [_critic(85, success=False, important=4)],
            2: [_critic(95, success=True, minor=1)],
        }
    )

    assert summary.calls == 3
    assert summary.accepted == 1
    assert summary.score_min == 55
    assert summary.score_median == 85
    assert summary.score_max == 95
    assert summary.score_histogram == {"50-59": 1, "80-89": 1, "90-99": 1}
    assert summary.fix_severity_histogram == {
        "critical": 2,
        "minor": 2,
        "important": 4,
    }


def test_render_and_repair_attempts_are_not_counted_as_critic_calls() -> None:
    """Only ``kind == "critic"`` is a critic call; the ledger must not inflate."""
    summary = summarize_facts_loop(
        {
            0: [
                FactsLoopAttempt(kind="render", success=True),
                FactsLoopAttempt(kind="llm_repair", success=True),
                _critic(70, success=False),
            ]
        }
    )

    assert summary.calls == 1
    assert summary.accepted == 0


def test_a_critic_call_with_no_score_still_counts_as_a_call() -> None:
    """A response whose score failed to parse is a billed call, not an absence."""
    summary = summarize_facts_loop(
        {0: [FactsLoopAttempt(kind="critic", score=None, success=False)]}
    )

    assert summary.calls == 1
    assert summary.score_median is None
    assert summary.score_histogram == {}
