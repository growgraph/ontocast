"""Timeout and retry behaviour that protects fan-out throughput.

A hung or rate-limited provider call is a throughput problem, not just an error
path: it holds a unit-worker slot *and* an LLM_MAX_INFLIGHT slot, so a couple of
them permanently narrow the pipeline.
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import Field

from ontocast.agent import common
from ontocast.agent.common import _retry_backoff_seconds, call_llm_with_retry
from ontocast.onto.model import BasePydanticModel
from ontocast.tool.llm import LLMTool

pytestmark = pytest.mark.anyio


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


async def test_parse_retries_are_exhausted_then_raised(monkeypatch) -> None:
    monkeypatch.setattr(common, "_retry_backoff_seconds", lambda attempt: 0.0)
    calls = 0

    async def fake_llm(prompt):
        nonlocal calls
        calls += 1
        return _Response("still not json")

    with pytest.raises(Exception):
        await call_llm_with_retry(
            cast(LLMTool, fake_llm), _prompt(), _parser(), _kwargs(), max_retries=3
        )

    assert calls == 3


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


async def test_timeout_is_not_a_cancellation() -> None:
    """The error must be catchable as Exception, not propagate as a cancel.

    The unit loops catch ``Exception`` to fail one unit gracefully; an
    ``asyncio.TimeoutError`` escaping ``gather`` would abort the whole fan-out.
    """
    from ontocast.tool.llm import LLMRequestTimeoutError

    assert issubclass(LLMRequestTimeoutError, Exception)
    assert not issubclass(LLMRequestTimeoutError, asyncio.CancelledError)
