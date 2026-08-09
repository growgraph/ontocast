"""Per-unit LLM budget attribution under concurrency.

The LLM tool is a ToolBox singleton. Binding a per-unit budget tracker to an
instance attribute meant concurrent unit workers overwrote each other, so
whichever bound last collected every in-flight call's usage.
"""

from __future__ import annotations

import asyncio

import pytest

from ontocast.onto.state import BudgetTracker
from ontocast.tool.llm import _active_budget_tracker, use_budget_tracker


def test_scope_restores_previous_tracker() -> None:
    outer = BudgetTracker()
    with use_budget_tracker(outer):
        assert _active_budget_tracker.get() is outer
    assert _active_budget_tracker.get() is None


@pytest.mark.anyio
async def test_concurrent_tasks_do_not_share_the_tracker() -> None:
    """Each task must see its own tracker across an await boundary."""
    observed: dict[int, BudgetTracker | None] = {}
    trackers = {index: BudgetTracker() for index in range(4)}

    async def worker(index: int) -> None:
        with use_budget_tracker(trackers[index]):
            # Yield so every task interleaves inside its own scope.
            await asyncio.sleep(0.01 * (4 - index))
            observed[index] = _active_budget_tracker.get()

    await asyncio.gather(*(worker(i) for i in range(4)))

    assert observed == trackers


@pytest.mark.anyio
async def test_usage_is_charged_to_the_scoped_tracker() -> None:
    """Charging goes to the task's tracker, not to a shared instance slot."""
    from ontocast.config import LLMConfig
    from ontocast.tool.llm import LLMTool

    instance_default = BudgetTracker()
    tool = LLMTool(config=LLMConfig(), budget_tracker=instance_default)

    scoped = BudgetTracker()
    with use_budget_tracker(scoped):
        tool._record_cache_hit("prompt", "response", None)

    assert scoped.cache_hits == 1
    assert instance_default.cache_hits == 0


def test_falls_back_to_the_instance_tracker_outside_a_scope() -> None:
    """Direct library use of a single LLMTool keeps working."""
    from ontocast.config import LLMConfig
    from ontocast.tool.llm import LLMTool

    tracker = BudgetTracker()
    tool = LLMTool(config=LLMConfig(), budget_tracker=tracker)

    tool._record_cache_hit("prompt", "response", None)

    assert tracker.cache_hits == 1
