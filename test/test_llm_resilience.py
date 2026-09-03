"""Timeout and retry behaviour that protects fan-out throughput.

A hung or rate-limited provider call is a throughput problem, not just an error
path: it holds a unit-worker slot *and* an LLM_MAX_INFLIGHT slot, so a couple of
them permanently narrow the pipeline.
"""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import Field

from ontocast.agent import common
from ontocast.agent.common import _retry_backoff_seconds, call_llm_with_retry
from ontocast.onto.model import BasePydanticModel
from ontocast.tool.llm import LLMTool

pytestmark = [pytest.mark.anyio, pytest.mark.unit]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Answer(BasePydanticModel):
    value: str = Field(default="")


def _parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=_Answer)


def _prompt() -> PromptTemplate:
    return PromptTemplate.from_template("{format_instructions}\nGo.")


def _kwargs() -> dict:
    return {"format_instructions": ""}


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


def test_backoff_grows_and_is_bounded_and_jittered() -> None:
    first = [_retry_backoff_seconds(1) for _ in range(50)]
    second = [_retry_backoff_seconds(2) for _ in range(50)]

    assert max(first) <= common.RETRY_BACKOFF_BASE_SECONDS
    assert sum(second) / len(second) > sum(first) / len(first)
    assert len(set(first)) > 1, "identical delays would re-issue in lockstep"
    assert max(_retry_backoff_seconds(20) for _ in range(20)) <= (
        common.RETRY_BACKOFF_MAX_SECONDS
    )


async def test_parse_failure_is_retried_with_feedback(monkeypatch) -> None:
    monkeypatch.setattr(common, "_retry_backoff_seconds", lambda attempt: 0.0)
    seen_prompts: list[str] = []
    responses = ["not json at all", '{"value": "ok"}']

    async def fake_llm(prompt):
        seen_prompts.append(str(prompt))
        return _Response(responses[len(seen_prompts) - 1])

    result = await call_llm_with_retry(
        cast(LLMTool, fake_llm), _prompt(), _parser(), _kwargs(), max_retries=3
    )

    assert result.value == "ok"
    assert len(seen_prompts) == 2
    assert "failed to parse" in seen_prompts[1]


async def test_transport_failure_is_not_retried(monkeypatch) -> None:
    monkeypatch.setattr(common, "_retry_backoff_seconds", lambda attempt: 0.0)
    calls = 0

    async def fake_llm(prompt):
        nonlocal calls
        calls += 1
        raise RuntimeError("429 rate limit exceeded")

    with pytest.raises(RuntimeError, match="429"):
        await call_llm_with_retry(
            cast(LLMTool, fake_llm), _prompt(), _parser(), _kwargs(), max_retries=3
        )

    # Retrying here would triple the request rate precisely when the provider
    # is asking for less of it, and the parse-error feedback would be nonsense.
    assert calls == 1


async def test_repeated_syntax_error_is_abandoned_before_exhaustion(
    monkeypatch,
) -> None:
    """Two identical syntax errors are a fixed point; the third call is spend.

    Each of these is a full re-extraction (~4k output plus reasoning tokens on
    the render path), so the saved attempt is not a rounding error.
    """
    monkeypatch.setattr(common, "_retry_backoff_seconds", lambda attempt: 0.0)
    calls = 0

    async def fake_llm(prompt):
        nonlocal calls
        calls += 1
        return _Response("still not json")

    with pytest.raises(common.LLMJsonParseError):
        await call_llm_with_retry(
            cast(LLMTool, fake_llm), _prompt(), _parser(), _kwargs(), max_retries=3
        )

    assert calls == 2


async def test_schema_errors_keep_their_full_retry_budget(monkeypatch) -> None:
    """A ValidationError is not a fixed point -- those retries do converge."""
    monkeypatch.setattr(common, "_retry_backoff_seconds", lambda attempt: 0.0)
    calls = 0

    async def fake_llm(prompt):
        nonlocal calls
        calls += 1
        # Parses fine; fails the schema (value must be a string).
        return _Response(json.dumps({"value": {"not": "a string"}}))

    with pytest.raises(Exception):
        await call_llm_with_retry(
            cast(LLMTool, fake_llm), _prompt(), _parser(), _kwargs(), max_retries=3
        )

    assert calls == 3


async def test_differing_syntax_errors_keep_retrying(monkeypatch) -> None:
    monkeypatch.setattr(common, "_retry_backoff_seconds", lambda attempt: 0.0)
    bodies = ['{"value": ', '{"value": "a" "b"}', "not json at all"]
    calls = 0

    async def fake_llm(prompt):
        nonlocal calls
        body = bodies[calls]
        calls += 1
        return _Response(body)

    with pytest.raises(Exception):
        await call_llm_with_retry(
            cast(LLMTool, fake_llm), _prompt(), _parser(), _kwargs(), max_retries=3
        )

    assert calls == 3


async def test_retry_feedback_is_bounded_and_points_at_the_error(monkeypatch) -> None:
    monkeypatch.setattr(common, "_retry_backoff_seconds", lambda attempt: 0.0)
    seen: list[str] = []
    # Long, valid up to the tail -- the shape the models actually emit.
    bodies = [
        '{"pad": "' + "x" * 9000 + '", "value": "a" "b"}',
        json.dumps({"value": "ok"}),
    ]
    calls = 0

    async def fake_llm(prompt):
        nonlocal calls
        seen.append(prompt.to_string())
        body = bodies[calls]
        calls += 1
        return _Response(body)

    result = await call_llm_with_retry(
        cast(LLMTool, fake_llm), _prompt(), _parser(), _kwargs(), max_retries=3
    )

    assert result.value == "ok"
    retry_prompt = seen[1]
    # The whole 9k body used to be pasted back into every retry.
    assert len(retry_prompt) < 4000
    assert '"a" "b"' in retry_prompt


async def test_timeout_frees_the_inflight_slot() -> None:
    """A hung call must not hold its concurrency slot forever."""
    from ontocast.config import LLMConfig
    from ontocast.tool.cache import Cacher
    from ontocast.tool.llm import LLMRequestTimeoutError, LLMTool

    class _HangingModel:
        async def ainvoke(self, *args, **kwds):
            await asyncio.Event().wait()

    tool = LLMTool(
        config=LLMConfig(
            cache_enabled=False,
            llm_max_inflight=1,
            request_timeout_seconds=0.05,
        ),
        cache=Cacher(),
    )
    tool._llm = _HangingModel()

    with pytest.raises(LLMRequestTimeoutError):
        await tool("hello")

    # The slot is back: a second call reaches the provider rather than queueing
    # behind the abandoned one.
    with pytest.raises(LLMRequestTimeoutError):
        await asyncio.wait_for(tool("hello again"), timeout=2.0)


async def test_timeout_is_charged_to_the_budget() -> None:
    """A timed-out call must reach calls_count, not only llm/timeouts.

    The provider received and worked on the prompt; only the answer was
    abandoned. Recorded as nothing, a run of timeouts read as a run of few,
    cheap calls: the prompt characters it paid for were missing from
    chars_sent and calls_count did not cover the attempt.
    """
    from ontocast.config import LLMConfig
    from ontocast.onto.state import BudgetTracker
    from ontocast.tool.cache import Cacher
    from ontocast.tool.llm import LLMRequestTimeoutError, LLMTool

    class _HangingModel:
        async def ainvoke(self, *args, **kwds):
            await asyncio.Event().wait()

    tracker = BudgetTracker()
    tool = LLMTool(
        config=LLMConfig(cache_enabled=False, request_timeout_seconds=0.05),
        cache=Cacher(),
        budget_tracker=tracker,
    )
    tool._llm = _HangingModel()

    with pytest.raises(LLMRequestTimeoutError):
        await tool("hello")

    assert tracker.calls_count == 1
    assert tracker.chars_sent == len("hello")
    assert tracker.chars_received == 0
    assert tracker.counters["llm/timeouts"] == 1
    assert tracker.counters["llm/calls_failed"] == 1
    assert "llm/calls_timed" not in tracker.counters


async def test_a_provider_error_is_a_failed_call_but_not_usage() -> None:
    """A rejected or dropped request cost nothing: counted, not charged."""
    from ontocast.config import LLMConfig
    from ontocast.onto.state import BudgetTracker
    from ontocast.tool.cache import Cacher
    from ontocast.tool.llm import LLMTool

    class _FailingModel:
        async def ainvoke(self, *args, **kwds):
            raise RuntimeError("connection reset")

    tracker = BudgetTracker()
    tool = LLMTool(
        config=LLMConfig(cache_enabled=False),
        cache=Cacher(),
        budget_tracker=tracker,
    )
    tool._llm = _FailingModel()

    with pytest.raises(RuntimeError):
        await tool("hello")

    assert tracker.calls_count == 0
    assert tracker.counters["llm/calls_failed"] == 1
    assert "llm/timeouts" not in tracker.counters


def test_unescape_json_delimiters_repairs_escaped_string_delimiters() -> None:
    """The model escapes the quotes that should *delimit* a JSON string.

    Observed in gpt-5-mini critique output:
    ``"text_fragment": \\"quoted text\\",`` — invalid JSON that langchain's
    partial parser silently degraded to ``None``.
    """
    broken = '{"a": \\"quoted \\"inner\\" text\\", "b": 1}'
    repaired = common.unescape_json_delimiters(broken)
    assert json.loads(repaired) == {"a": 'quoted "inner" text', "b": 1}


def test_unescape_json_delimiters_repairs_escaped_token_whitespace() -> None:
    # Same responses escape the newline after the comma: ``\\",\\n  "action"``.
    broken = '{"a": \\"x\\",\\n  "action": "ADD"}'
    assert json.loads(common.unescape_json_delimiters(broken)) == {
        "a": "x",
        "action": "ADD",
    }


def test_unescape_json_delimiters_leaves_valid_json_untouched() -> None:
    valid = '{"note": "he said: \\"hi\\", then left", "n": [1, 2], "u": "a\\\\b"}'
    assert common.unescape_json_delimiters(valid) == valid


def test_parse_json_object_reports_position_on_structural_error() -> None:
    """A missing ``}`` must raise with line/column and a context window.

    Langchain's partial parser returned ``None`` for this shape, so the retry
    feedback was ``input_value=None`` — and the model repeated the identical
    malformation on retry. The feedback must name the broken spot instead.
    """
    broken = '{"graph_update": {"ops": [{"type": "insert", "x": 1]}}'
    with pytest.raises(ValueError) as exc_info:
        common.parse_json_object(broken)
    message = str(exc_info.value)
    assert "line 1" in message
    assert "insert" in message, "context window around the error is missing"


def test_parse_json_object_rejects_partial_recovery() -> None:
    # An unterminated string must not silently validate as a truncated object.
    broken = '{"success": true, "score": 90, "fixes": ["one", "two'
    with pytest.raises(ValueError):
        common.parse_json_object(broken)


def test_parse_json_object_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="not an object"):
        common.parse_json_object("null")


def test_parse_json_object_accepts_fenced_and_control_characters() -> None:
    fenced = '```json\n{"value": "line one\nline two"}\n```'
    assert common.parse_json_object(fenced) == {"value": "line one\nline two"}


async def test_timeout_is_retried_exactly_once(monkeypatch) -> None:
    """One identical re-issue for a timeout, then it propagates.

    A timeout is not a provider "send less" signal, and at low MAX_VISITS a
    lost call silently costs a unit its whole critique.
    """
    from ontocast.tool.llm import LLMRequestTimeoutError

    monkeypatch.setattr(common, "_retry_backoff_seconds", lambda attempt: 0.0)
    calls = 0

    async def timeout_once_llm(prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMRequestTimeoutError("LLM request exceeded 180.0s")
        return _Response('{"value": "ok"}')

    result = await call_llm_with_retry(
        cast(LLMTool, timeout_once_llm), _prompt(), _parser(), _kwargs()
    )
    assert result.value == "ok"
    assert calls == 2

    calls = 0

    async def always_timeout_llm(prompt):
        nonlocal calls
        calls += 1
        raise LLMRequestTimeoutError("LLM request exceeded 180.0s")

    with pytest.raises(LLMRequestTimeoutError):
        await call_llm_with_retry(
            cast(LLMTool, always_timeout_llm), _prompt(), _parser(), _kwargs()
        )
    assert calls == 2


async def test_timeout_is_not_a_cancellation() -> None:
    """The error must be catchable as Exception, not propagate as a cancel.

    The unit loops catch ``Exception`` to fail one unit gracefully; an
    ``asyncio.TimeoutError`` escaping ``gather`` would abort the whole fan-out.
    """
    from ontocast.tool.llm import LLMRequestTimeoutError

    assert issubclass(LLMRequestTimeoutError, Exception)
    assert not issubclass(LLMRequestTimeoutError, asyncio.CancelledError)


def test_repair_bracket_kinds_fixes_the_observed_swapped_tail() -> None:
    """The measured failure: `] }` emitted where `} ]` was due, at the tail.

    Bracket *counts* balance, so only the decoder notices.
    """
    broken = (
        '{"graph_update": {"triple_operations": [{"type": "insert", "graph": '
        '{"@context": {"ex": "http://example.org/"}, '
        '"@graph": [{"@id": "ex:a"}]}}]}}'
    ).replace('"@id": "ex:a"}]}}]}}', '"@id": "ex:a"}]}]}}}')

    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)

    repaired, fixes = common.repair_bracket_kinds(broken)
    assert fixes == 2
    parsed = json.loads(repaired)
    ops = parsed["graph_update"]["triple_operations"]
    assert len(ops) == 1
    assert ops[0]["type"] == "insert"


def test_repair_bracket_kinds_leaves_valid_json_untouched() -> None:
    text = json.dumps({"a": [1, 2, {"b": ["c"]}], "d": {"e": []}}, indent=2)
    repaired, fixes = common.repair_bracket_kinds(text)
    assert fixes == 0
    assert repaired == text


def test_repair_bracket_kinds_ignores_brackets_inside_strings() -> None:
    text = '{"a": "] } [ {", "b": ["x"]}'
    repaired, fixes = common.repair_bracket_kinds(text)
    assert fixes == 0
    assert repaired == text


def test_repair_bracket_kinds_abandons_truncated_json() -> None:
    """Frames still open at EOF are truncation; closing them fabricates output."""
    truncated = '{"a": {"b": [1, 2'
    repaired, fixes = common.repair_bracket_kinds(truncated)
    assert fixes == 0
    assert repaired == truncated


def test_repair_bracket_kinds_abandons_unopened_closer() -> None:
    text = '{"a": 1}}'
    repaired, fixes = common.repair_bracket_kinds(text)
    assert fixes == 0
    assert repaired == text


def test_parse_json_object_recovers_a_swapped_tail() -> None:
    broken = '{"outer": {"items": [{"k": {"v": [1]}}]}}'.replace("]}}]}}", "]}]}}}")
    assert common.parse_json_object(broken) == {"outer": {"items": [{"k": {"v": [1]}}]}}


def test_parse_json_object_raises_typed_error_with_position() -> None:
    with pytest.raises(common.LLMJsonParseError) as excinfo:
        common.parse_json_object('{"a": 1 "b": 2}')
    assert excinfo.value.pos > 0
    assert excinfo.value.msg
