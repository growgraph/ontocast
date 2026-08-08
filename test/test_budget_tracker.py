"""Tests for BudgetTracker usage and merge behavior."""

from ontocast.onto.state import UNIT_SUM_SUFFIX, BudgetTracker


def test_add_usage_records_chars_and_tokens() -> None:
    tracker = BudgetTracker()
    tracker.add_usage(100, 50, input_tokens=10, output_tokens=5)
    assert tracker.chars_sent == 100
    assert tracker.chars_received == 50
    assert tracker.calls_count == 1
    assert tracker.input_tokens == 10
    assert tracker.output_tokens == 5


def test_add_usage_without_tokens_leaves_token_counters_zero() -> None:
    tracker = BudgetTracker()
    tracker.add_usage(100, 50)
    assert tracker.input_tokens == 0
    assert tracker.output_tokens == 0


def test_merge_from_accumulates_tokens() -> None:
    left = BudgetTracker()
    left.add_usage(10, 5, input_tokens=3, output_tokens=1)
    right = BudgetTracker()
    right.add_usage(20, 15, input_tokens=7, output_tokens=4)
    left.merge_from(right)
    assert left.input_tokens == 10
    assert left.output_tokens == 5
    assert left.chars_sent == 30


def test_add_cache_hit_does_not_increment_calls_count() -> None:
    tracker = BudgetTracker()
    tracker.add_cache_hit(100, 50)
    assert tracker.cache_hits == 1
    assert tracker.calls_count == 0
    assert tracker.chars_sent == 100
    assert tracker.chars_received == 50


def test_merge_from_accumulates_cache_hits() -> None:
    left = BudgetTracker()
    left.add_cache_hit(10, 5)
    right = BudgetTracker()
    right.add_cache_hit(20, 15)
    left.merge_from(right)
    assert left.cache_hits == 2


def test_get_summary_includes_tokens_when_present() -> None:
    tracker = BudgetTracker()
    tracker.add_usage(100, 50, input_tokens=1000, output_tokens=250)
    summary = tracker.get_summary()
    assert "1,000 in / 250 out tokens" in summary


def test_get_summary_omits_tokens_when_zero() -> None:
    tracker = BudgetTracker()
    tracker.add_usage(100, 50)
    summary = tracker.get_summary()
    assert "tokens" not in summary


def test_add_duration_accumulates_per_name() -> None:
    tracker = BudgetTracker()
    tracker.add_duration("Chunk Text", 1.5)
    tracker.add_duration("Chunk Text", 0.5)
    tracker.add_duration("Serialize", 0.25)
    assert tracker.node_durations == {"Chunk Text": 2.0, "Serialize": 0.25}


def test_merge_from_sums_node_durations() -> None:
    left = BudgetTracker()
    left.add_duration("unit facts loop", 2.0)
    right = BudgetTracker()
    right.add_duration("unit facts loop", 3.0)
    right.add_duration("unit ontology loop", 1.0)
    left.merge_from(right)
    assert left.node_durations == {
        "unit facts loop": 5.0,
        "unit ontology loop": 1.0,
    }


def test_get_duration_summary_ranks_slowest_first() -> None:
    tracker = BudgetTracker()
    assert tracker.get_duration_summary() == ""
    tracker.add_duration("fast", 0.5)
    tracker.add_duration("slow", 2.0)
    summary = tracker.get_duration_summary()
    assert summary.startswith("Durations: slow 2.0s")
    assert "fast 0.5s" in summary


def test_incr_accumulates_and_merges() -> None:
    left = BudgetTracker()
    left.incr("ctx/merge_document_ontology.calls")
    left.incr("ctx/merge_document_ontology.calls", 2)
    assert left.counters == {"ctx/merge_document_ontology.calls": 3}

    right = BudgetTracker()
    right.incr("ctx/merge_document_ontology.calls", 4)
    right.incr("llm/calls_timed")
    left.merge_from(right)
    assert left.counters == {
        "ctx/merge_document_ontology.calls": 7,
        "llm/calls_timed": 1,
    }


def test_peak_keys_take_max_instead_of_summing() -> None:
    # Two workers each observing a 0.4s stall saw one 0.4s stall between them,
    # not a 0.8s one; summing would invent a pause that never happened.
    tracker = BudgetTracker()
    tracker.add_duration("Render Facts/loop_lag_max", 0.4)
    tracker.add_duration("Render Facts/loop_lag_max", 0.1)
    assert tracker.node_durations["Render Facts/loop_lag_max"] == 0.4

    other = BudgetTracker()
    other.add_duration("Render Facts/loop_lag_max", 0.9)
    other.add_duration("Render Facts/loop_lag_total", 1.0)
    tracker.add_duration("Render Facts/loop_lag_total", 2.0)
    tracker.merge_from(other)
    assert tracker.node_durations["Render Facts/loop_lag_max"] == 0.9
    assert tracker.node_durations["Render Facts/loop_lag_total"] == 3.0


def test_parallel_efficiency_needs_both_wall_and_unit_sum() -> None:
    tracker = BudgetTracker()
    assert tracker.parallel_efficiency("Render Facts") is None

    tracker.add_duration(f"Render Facts{UNIT_SUM_SUFFIX}", 40.0)
    assert tracker.parallel_efficiency("Render Facts") is None, (
        "unit_sum without a wall-clock entry cannot yield a ratio"
    )

    tracker.add_duration("Render Facts", 10.0)
    assert tracker.parallel_efficiency("Render Facts") == 4.0


def test_unit_sum_does_not_pollute_the_wall_clock_entry() -> None:
    # The two keys must stay separate: adding worker time to the node's wall
    # clock is exactly the confusion the suffix convention exists to prevent.
    tracker = BudgetTracker()
    tracker.add_duration("Render Facts", 10.0)
    tracker.add_duration(f"Render Facts{UNIT_SUM_SUFFIX}", 40.0)
    tracker.add_duration("Render Facts/worker_wait", 5.0)
    assert tracker.node_durations["Render Facts"] == 10.0


def test_get_parallelism_summary_reports_effective_workers() -> None:
    tracker = BudgetTracker()
    assert tracker.get_parallelism_summary() == ""

    tracker.add_duration("Render Facts", 10.0)
    tracker.add_duration(f"Render Facts{UNIT_SUM_SUFFIX}", 40.0)
    tracker.add_duration("Render Facts/loop_lag_total", 8.0)
    summary = tracker.get_parallelism_summary()
    assert "Render Facts 4.0x" in summary
    assert "loop lag 8.0s" in summary
