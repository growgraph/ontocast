"""The per-run token meter: arithmetic, limits, enforcement and propagation."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from ontocast.config import LLMConfig
from ontocast.config.settings import ServerConfig
from ontocast.onto.enum import WorkflowNode
from ontocast.onto.run_meter import (
    BudgetExceededError,
    RunMeter,
    RunMeterSnapshot,
    TokenTotals,
    active_run_meter,
    run_meter_scope,
    use_run_meter,
)
from ontocast.onto.state import AgentState, BudgetTracker
from ontocast.onto.token_usage import TokenUsage
from ontocast.stategraph.node_factories import _gather_units
from ontocast.tool.cache import Cacher
from ontocast.tool.llm import LLMTool

pytestmark = pytest.mark.unit


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


_USAGE = {
    "input_tokens": 1000,
    "output_tokens": 300,
    "total_tokens": 1300,
    "input_token_details": {"cache_read": 600, "cache_creation": 100},
    "output_token_details": {"reasoning": 200},
}


class _FakeModel:
    """Provider stand-in that counts calls and reports fixed usage."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls = 0
        self.delay = delay

    async def ainvoke(self, *args, **kwds) -> AIMessage:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return AIMessage(content="ok", usage_metadata=_USAGE)  # type: ignore[arg-type]


def _tool(model: _FakeModel, *, cache_dir=None) -> LLMTool:
    config = LLMConfig(cache_enabled=cache_dir is not None)
    tool = LLMTool(
        config=config,
        cache=Cacher(cache_dir=cache_dir) if cache_dir is not None else Cacher(),
        budget_tracker=BudgetTracker(),
    )
    tool._llm = model
    return tool


# Counting and limits


def test_usage_is_kept_per_model_and_summed() -> None:
    meter = RunMeter()
    meter.record("a", TokenTotals(input=10, calls=1))
    meter.record("b", TokenTotals(input=5, calls=1))
    meter.record("a", TokenTotals(replayed_input=7, replays=1))

    assert meter.usage_by_model["a"].input == 10
    assert meter.usage_by_model["a"].replays == 1
    totals = meter.totals()
    assert (totals.input, totals.calls, totals.replayed_input) == (15, 2, 7)


def test_the_limit_policy_sees_counts_per_model_and_refuses_with_its_reason() -> None:
    seen: list[dict[str, int]] = []

    def policy(usage) -> str | None:
        seen.append({model: totals.output for model, totals in usage.items()})
        return "over budget" if sum(seen[-1].values()) >= 100 else None

    meter = RunMeter(limit=policy)
    meter.record("a", TokenTotals(output=60))
    meter.record("b", TokenTotals(output=39))
    meter.check()
    assert seen[-1] == {"a": 60, "b": 39}

    meter.record("b", TokenTotals(output=1))
    with pytest.raises(BudgetExceededError, match="over budget") as info:
        meter.check()
    assert info.value.snapshot["exceeded"] == "over budget"
    assert meter.exceeded == "over budget"


def test_a_refusal_sticks_without_asking_the_policy_again() -> None:
    asked: list[int] = []

    def policy(usage) -> str | None:
        asked.append(1)
        return "stop"

    meter = RunMeter(limit=policy)
    for _ in range(3):
        with pytest.raises(BudgetExceededError):
            meter.check()
    assert len(asked) == 1


def test_the_policy_gets_a_copy_and_may_read_the_meter() -> None:
    meter = RunMeter()

    def policy(usage) -> str | None:
        usage["m"].input = 10**9  # a copy: the meter is unchanged
        assert meter.totals().input == 5  # no deadlock: called outside the lock
        return None

    meter.limit = policy
    meter.record("m", TokenTotals(input=5))
    meter.check()
    assert meter.totals().input == 5


def test_an_exception_in_the_policy_propagates() -> None:
    def policy(usage) -> str | None:
        raise ZeroDivisionError("policy bug")

    meter = RunMeter(limit=policy)
    with pytest.raises(ZeroDivisionError):
        meter.check()
    assert meter.exceeded is None


def test_token_caps_count_provider_tokens_only() -> None:
    meter = RunMeter.from_token_limits(max_input_tokens=100, max_output_tokens=50)
    assert meter is not None
    meter.record("m", TokenTotals(replayed_input=10_000, replayed_output=10_000))
    meter.check()

    meter.record("m", TokenTotals(output=50))
    with pytest.raises(BudgetExceededError, match="output"):
        meter.check()


def test_token_caps_are_checked_before_the_policy() -> None:
    asked: list[int] = []

    def policy(usage) -> str | None:
        asked.append(1)
        return None

    meter = RunMeter(limit=policy, token_caps={"input": 50})
    meter.record("m", TokenTotals(input=50))
    with pytest.raises(BudgetExceededError, match="cap"):
        meter.check()
    assert asked == []


def test_no_limits_means_no_meter() -> None:
    assert RunMeter.from_token_limits() is None


def test_unknown_cap_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        RunMeter(token_caps={"tokens": 1})


def test_snapshot_is_json_serialisable() -> None:
    import json

    meter = RunMeter(token_caps={"input": 10})
    meter.record("m", TokenTotals(input=4, calls=1))
    snapshot = meter.snapshot()
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert snapshot["usage_by_model"]["m"]["input"] == 4
    assert snapshot["totals"]["calls"] == 1
    assert set(snapshot) == {"usage_by_model", "totals", "token_caps", "exceeded"}


# on_usage throttling


def test_on_usage_is_throttled_and_flush_always_reports() -> None:
    now = [0.0]
    seen: list[RunMeterSnapshot] = []
    meter = RunMeter(on_usage=seen.append, min_interval_s=10.0, clock=lambda: now[0])

    meter.record("m", TokenTotals(input=1))  # first record reports
    meter.record("m", TokenTotals(input=1))  # inside the interval
    now[0] = 5.0
    meter.record("m", TokenTotals(input=1))  # still inside
    assert [s["totals"]["input"] for s in seen] == [1]

    now[0] = 10.0
    meter.record("m", TokenTotals(input=1))
    assert [s["totals"]["input"] for s in seen] == [1, 4]

    meter.flush()
    assert [s["totals"]["input"] for s in seen] == [1, 4, 4]


def test_a_failing_callback_does_not_fail_the_run() -> None:
    def boom(snapshot: RunMeterSnapshot) -> None:
        raise RuntimeError("sink down")

    meter = RunMeter(on_usage=boom)
    meter.record("m", TokenTotals(input=1))
    meter.flush()
    assert meter.totals().input == 1


def test_use_run_meter_flushes_on_exit() -> None:
    seen: list[RunMeterSnapshot] = []
    meter = RunMeter(on_usage=seen.append, min_interval_s=3600.0)
    with use_run_meter(meter):
        meter.record("m", TokenTotals(input=1))
        meter.record("m", TokenTotals(input=2))
    assert seen[-1]["totals"]["input"] == 3


# Binding


def test_use_run_meter_raises_on_exit_when_a_refusal_was_swallowed() -> None:
    meter = RunMeter(token_caps={"input": 1})
    meter.record("m", TokenTotals(input=1))
    with pytest.raises(BudgetExceededError):
        with use_run_meter(meter):
            try:
                meter.check()
            except BudgetExceededError:
                pass  # a best-effort step that swallowed it
    assert active_run_meter() is None


def test_run_meter_scope_defers_to_a_bound_meter() -> None:
    outer = RunMeter()
    with use_run_meter(outer):
        with run_meter_scope(max_input_tokens=5) as scoped:
            assert scoped is outer
            assert outer.token_caps == {}
    with run_meter_scope() as none:
        assert none is None
        assert active_run_meter() is None
    with run_meter_scope(max_output_tokens=5) as capped:
        assert capped is not None
        assert active_run_meter() is capped
        assert capped.token_caps == {"output": 5}


def test_document_caps_are_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("MAX_INPUT_TOKENS_PER_DOCUMENT", "1000")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS_PER_DOCUMENT", "200")
    server = ServerConfig()
    assert server.max_input_tokens_per_document == 1000
    assert server.max_output_tokens_per_document == 200


def test_document_caps_default_to_unlimited() -> None:
    server = ServerConfig()
    assert server.max_input_tokens_per_document is None
    assert server.max_output_tokens_per_document is None


# LLMTool enforcement and recording


@pytest.mark.anyio
async def test_provider_call_is_metered_by_model() -> None:
    model = _FakeModel()
    tool = _tool(model)
    meter = RunMeter()
    with use_run_meter(meter):
        await tool("prompt")

    usage = meter.usage_by_model[str(tool.config.model_name)]
    assert usage == TokenTotals(
        input=1000,
        cache_read=600,
        cache_write=100,
        output=300,
        reasoning=200,
        calls=1,
    )


@pytest.mark.anyio
async def test_a_refused_call_never_reaches_the_provider() -> None:
    model = _FakeModel()
    tool = _tool(model)
    meter = RunMeter(token_caps={"input": 10})
    meter.record("m", TokenTotals(input=10))

    with pytest.raises(BudgetExceededError):
        with use_run_meter(meter):
            await tool("prompt")
    assert model.calls == 0


@pytest.mark.anyio
async def test_a_replay_is_metered_as_replayed_and_is_never_refused(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    await _tool(_FakeModel(), cache_dir=cache_dir)("prompt")

    model = _FakeModel()
    replaying = _tool(model, cache_dir=cache_dir)
    # The cap is already reached, but a replay spends nothing and is not refused.
    meter = RunMeter(token_caps={"input": 1})
    meter.record("m", TokenTotals(input=1))
    with use_run_meter(meter):
        await replaying("prompt")
    assert model.calls == 0
    usage = meter.usage_by_model[str(replaying.config.model_name)]
    assert (usage.replayed_input, usage.replayed_output, usage.replays) == (
        1000,
        300,
        1,
    )
    assert (usage.input, usage.calls) == (0, 0)


# Fan-out


@pytest.mark.anyio
async def test_a_budget_stop_drains_the_fanout_and_raises_once() -> None:
    """Every unit is awaited before the single re-raise; nothing is orphaned."""
    model = _FakeModel(delay=0.01)
    tool = _tool(model)
    meter = RunMeter(token_caps={"input": 1000})
    finished: list[int] = []
    tasks_before = asyncio.all_tasks()

    async def unit(index: int) -> int:
        try:
            for _ in range(3):
                await tool(f"unit {index}")
            return index
        finally:
            finished.append(index)

    state = AgentState()
    with pytest.raises(BudgetExceededError):
        with use_run_meter(meter):
            await _gather_units(
                WorkflowNode.RENDER_FACTS, state, [unit(i) for i in range(4)]
            )

    assert sorted(finished) == [0, 1, 2, 3]
    assert asyncio.all_tasks() - tasks_before == set()
    # Each unit's first call was already admitted; every later one was refused.
    assert model.calls == 4
    assert meter.totals().input == 4000


# Context propagation through LangGraph


class _GraphState(BaseModel):
    seen: list[bool] = []


@pytest.mark.anyio
async def test_the_bound_meter_reaches_langgraph_nodes_and_their_subtasks() -> None:
    from langgraph.graph import END, START, StateGraph

    async def node(state: _GraphState) -> dict[str, list[bool]]:
        async def sub() -> bool:
            meter = active_run_meter()
            if meter is None:
                return False
            meter.record("m", TokenTotals(input=1, calls=1))
            return True

        results = await asyncio.gather(sub(), sub())
        return {"seen": [*state.seen, *results]}

    builder = StateGraph(_GraphState)
    builder.add_node("a", node)
    builder.add_node("b", node)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile()

    meter = RunMeter()
    with use_run_meter(meter):
        final = None
        async for chunk in graph.astream(_GraphState(), stream_mode="values"):
            final = chunk

    assert final is not None and final["seen"] == [True] * 4
    assert meter.totals().calls == 4


# BudgetTracker paid / replay split


def test_tracker_keeps_replayed_detail_out_of_billed_fields() -> None:
    tracker = BudgetTracker()
    detail = TokenUsage(
        input_tokens=1000,
        output_tokens=400,
        reasoning_tokens=300,
        cache_read_input_tokens=600,
        cache_creation_input_tokens=50,
    )
    tracker.add_usage(10, 5, usage=detail)
    tracker.add_cache_hit(10, 5, usage=detail)

    assert tracker.reasoning_tokens == 300
    assert tracker.cache_read_input_tokens == 600
    assert tracker.cache_creation_input_tokens == 50
    assert tracker.replayed_reasoning_tokens == 300
    assert tracker.replayed_cache_read_input_tokens == 600
    assert tracker.replayed_cache_creation_input_tokens == 50
    # The ratios still describe the whole workload.
    assert tracker.prefix_cache_hit_rate == 0.6
    assert tracker.reasoning_share_of_output == 0.75

    merged = BudgetTracker()
    merged.merge_from(tracker)
    assert merged.replayed_reasoning_tokens == 300
    assert merged.reasoning_tokens == 300
