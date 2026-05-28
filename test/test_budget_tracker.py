"""Tests for BudgetTracker usage and merge behavior."""

from ontocast.onto.state import BudgetTracker


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
