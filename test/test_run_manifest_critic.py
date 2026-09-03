"""The run manifest must carry the evidence for the critic's own decisions.

The facts loop accepts a render on ``critique.success or critique.score > 90``,
a score the model is asked for with no rubric and no statement of the threshold.
Nothing recorded that score, so establishing that the gate was miscalibrated
(median 79, 82% rejected, 62% landing in 70-85) required mining the LLM disk
cache -- which only worked because caching happened to be enabled.
"""

import pytest

from ontocast.onto.model import LoopAttempt
from ontocast.onto.run_manifest import summarize_loop

pytestmark = pytest.mark.unit


def _critic(score: float, *, success: bool, **severities: int) -> LoopAttempt:
    return LoopAttempt(
        kind="critic",
        score=score,
        success=success,
        severity_counts=dict(severities),
        n_actionable_fixes=sum(severities.values()),
    )


def test_no_critic_calls_summarizes_to_zero() -> None:
    """The MAX_VISITS=1 default: the critic never runs, and that is not an error."""
    summary = summarize_loop(
        {0: [LoopAttempt(kind="render"), LoopAttempt(kind="llm_repair")]}
    )

    assert summary.calls == 0
    assert summary.accepted == 0
    assert summary.score_median is None
    assert summary.score_histogram == {}


def test_scores_and_severities_are_summarized_across_units() -> None:
    summary = summarize_loop(
        {
            0: [
                LoopAttempt(kind="render"),
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
    summary = summarize_loop(
        {
            0: [
                LoopAttempt(kind="render", success=True),
                LoopAttempt(kind="llm_repair", success=True),
                _critic(70, success=False),
            ]
        }
    )

    assert summary.calls == 1
    assert summary.accepted == 0


def test_a_critic_call_with_no_score_still_counts_as_a_call() -> None:
    """A response whose score failed to parse is a billed call, not an absence."""
    summary = summarize_loop(
        {0: [LoopAttempt(kind="critic", score=None, success=False)]}
    )

    assert summary.calls == 1
    assert summary.score_median is None
    assert summary.score_histogram == {}


def test_batch_state_merge_carries_the_critic_telemetry() -> None:
    """The case10 failure shape: manifests said `critic: {calls: 0}` while
    their own retrieval_metrics recorded 20 facts-critic and 26
    ontology-critic calls. The batch path merges astream dict chunks through
    an explicit copy list, and the telemetry fields were not on it -- so
    everything summarize_loop reads arrived empty at manifest time.
    """
    from ontocast.api.process_helpers import _merge_workflow_state_into_agent_state
    from ontocast.onto.state import AgentState

    chunk = {
        "facts_loop_telemetry": {0: [LoopAttempt(kind="critic", success=True)]},
        "ontology_loop_telemetry": {0: [LoopAttempt(kind="critic", score=85.0)]},
        "ontology_reduce_metrics": {"minted_duplicates": 2},
    }
    state = _merge_workflow_state_into_agent_state(AgentState(), chunk)

    assert summarize_loop(state.facts_loop_telemetry).calls == 1
    assert summarize_loop(state.ontology_loop_telemetry).calls == 1
    assert state.ontology_reduce_metrics["minted_duplicates"] == 2


def test_batch_state_merge_carries_the_selection_census_and_unit_ledgers() -> None:
    """The selection census and the per-unit ledgers ride the same copy list.

    ``labeled_units`` / ``section_label_histogram`` read ``content_units``;
    the validation dump reads ``facts_repairs_applied`` and ``unit_failures``.
    None were copied, so every batch manifest reported an empty census -- a
    section filter that never acted was unrecordable -- and every predicate
    the machine rewrote in a render was logged and dropped.
    """
    from rdflib import URIRef

    from ontocast.api.process_helpers import (
        _merge_workflow_state_into_agent_state,
        _selection_manifest,
    )
    from ontocast.config import Config
    from ontocast.onto.content_unit import ContentUnit
    from ontocast.onto.model import (
        FactsUnitFindingKind,
        GraphRepairRecord,
        UnitFailure,
    )
    from ontocast.onto.state import AgentState

    doc = URIRef("https://example.com/doc")
    chunk = {
        "content_units": [
            ContentUnit(text="a", index=0, doc_iri=doc, section_label="methods"),
            ContentUnit(text="b", index=1, doc_iri=doc, section_label="methods"),
            ContentUnit(text="c", index=2, doc_iri=doc),
        ],
        "unit_failures": [UnitFailure(unit_index=2, phase="facts", stage="render")],
        "facts_repairs_applied": {
            0: [
                GraphRepairRecord(
                    kind=FactsUnitFindingKind.PROPERTY_ALIAS,
                    source="ex:hasASite",
                    target="ex:hasBSite",
                )
            ]
        },
        "aggregation_clusters": {"ex:final": ["ex:a", "ex:b"]},
        "aggregation_key_clusters": ["ex:final"],
    }
    state = _merge_workflow_state_into_agent_state(AgentState(), chunk)

    selection = _selection_manifest(state, Config())
    assert selection.labeled_units == 2
    assert selection.unlabeled_units == 1
    assert selection.section_label_histogram == {"methods": 2, "(unlabeled)": 1}
    assert [failure.unit_index for failure in state.unit_failures] == [2]
    assert state.facts_repairs_applied[0][0].target == "ex:hasBSite"
    assert state.aggregation_clusters == {"ex:final": ["ex:a", "ex:b"]}
    assert state.aggregation_key_clusters == ["ex:final"]


def test_selection_manifest_records_the_summary_cap_only_when_summarizing() -> None:
    """``summary_max_sentences`` has no effect without ``summarize_sections``.

    Writing its default beside an empty section list reads as a setting the
    run used; the manifest must tell a knob that acted from one that merely
    had a value.
    """
    from ontocast.api.process_helpers import _selection_manifest
    from ontocast.config import Config
    from ontocast.onto.state import AgentState

    off = _selection_manifest(AgentState(), Config())
    assert off.summary_max_sentences is None

    on = _selection_manifest(
        AgentState(summarize_sections=["introduction"], summary_max_sentences=3),
        Config(),
    )
    assert on.summary_max_sentences == 3


def test_unreviewed_and_skipped_units_are_counted_apart_from_calls() -> None:
    """A critic that did not answer is a billed call and an unreviewed unit; a
    skipped pass is neither a call nor a review, and must not inflate either."""
    summary = summarize_loop(
        {
            0: [
                LoopAttempt(kind="render", success=True),
                LoopAttempt(
                    kind="critic", success=False, accept_reason="critic_unavailable"
                ),
            ],
            1: [
                LoopAttempt(kind="render", success=True),
                LoopAttempt(kind="critic_skipped", accept_reason="empty_render"),
            ],
            2: [_critic(90, success=True)],
        }
    )

    assert summary.calls == 2
    assert summary.accepted == 1
    assert summary.units_unreviewed == 1
    assert summary.units_skipped == 1
    assert summary.accept_reason_histogram["critic_unavailable"] == 1


def test_per_fix_outcomes_are_summed_over_patch_passes() -> None:
    summary = summarize_loop(
        {
            0: [
                LoopAttempt(
                    kind="critic_patch",
                    n_fixes_applied=2,
                    n_fixes_rolled_back=1,
                    patch_rolled_back=True,
                    n_fixes_junk_refused=3,
                    n_fixes_unresolved_prefix=1,
                ),
                LoopAttempt(kind="critic_patch", n_fixes_junk_refused=1),
            ]
        }
    )

    assert summary.patches_rolled_back == 1
    assert summary.fixes_rolled_back == 1
    assert summary.fixes_junk_refused == 4
    assert summary.fixes_unresolved_prefix == 1


def test_completion_summary_reads_its_own_attempt_kind() -> None:
    from ontocast.onto.run_manifest import summarize_completion

    summary = summarize_completion(
        {
            0: [
                LoopAttempt(kind="critic"),
                LoopAttempt(
                    kind="completion",
                    n_fixes_applied=2,
                    n_fixes_rolled_back=1,
                    n_triples_inserted=7,
                    n_measurements_recovered=2,
                ),
            ],
            1: [LoopAttempt(kind="render")],
        }
    )

    assert summary.calls == 1
    assert summary.units == 1
    assert summary.subjects_inserted == 2
    assert summary.subjects_rolled_back == 1
    assert summary.triples_inserted == 7
    assert summary.measurements_recovered == 2
