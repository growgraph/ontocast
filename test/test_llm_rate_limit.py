"""Provider pacing and throttle visibility (mocked, no live API).

LLM_MAX_INFLIGHT caps concurrency; LLM_REQUESTS_PER_SECOND paces the
sustained rate underneath it; LLM_MAX_RETRIES tunes the provider SDK's own
backoff. A throttle that slips through anyway must be counted -- a 429 used
to surface as an ordinary failed render, indistinguishable from a model
failure in the telemetry.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.rate_limiters import InMemoryRateLimiter

from ontocast.config import LLMConfig, LLMProvider, OpenAIModel
from ontocast.tool.llm import LLMTool, _is_rate_limit_error

pytestmark = pytest.mark.unit


def _openai_config(**overrides) -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=OpenAIModel.GPT4_O_MINI,
        api_key="test-key",
        **overrides,
    )


def _setup_kwargs(config: LLMConfig, patch_target: str) -> dict:
    with patch(patch_target) as mock_cls:
        mock_cls.return_value = MagicMock()
        tool = LLMTool(config=config)
        asyncio.run(tool.setup())
        mock_cls.assert_called_once()
        return dict(mock_cls.call_args.kwargs)


def test_pacing_kwargs_absent_by_default() -> None:
    kwargs = _setup_kwargs(_openai_config(), "langchain_openai.ChatOpenAI")
    assert "rate_limiter" not in kwargs
    assert "max_retries" not in kwargs


def test_requests_per_second_builds_a_limiter() -> None:
    kwargs = _setup_kwargs(
        _openai_config(requests_per_second=2.0), "langchain_openai.ChatOpenAI"
    )
    limiter = kwargs["rate_limiter"]
    assert isinstance(limiter, InMemoryRateLimiter)
    assert limiter.requests_per_second == 2.0


def test_max_retries_is_forwarded() -> None:
    kwargs = _setup_kwargs(_openai_config(max_retries=5), "langchain_openai.ChatOpenAI")
    assert kwargs["max_retries"] == 5


def test_ollama_gets_pacing_but_no_retry_budget() -> None:
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="qwen3.6:27b",
        requests_per_second=1.0,
        max_retries=5,
    )
    kwargs = _setup_kwargs(config, "langchain_ollama.ChatOllama")
    assert isinstance(kwargs["rate_limiter"], InMemoryRateLimiter)
    # ChatOllama exposes no retry budget; the knob must not be forwarded.
    assert "max_retries" not in kwargs


def test_google_gets_pacing_and_retries() -> None:
    config = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-3.5-flash",
        api_key="test-key",
        requests_per_second=2.0,
        max_retries=4,
    )
    kwargs = _setup_kwargs(config, "langchain_google_genai.ChatGoogleGenerativeAI")
    assert isinstance(kwargs["rate_limiter"], InMemoryRateLimiter)
    assert kwargs["max_retries"] == 4


class RateLimitError(Exception):
    """Name-alike of the providers' throttle exceptions."""


def test_rate_limit_classifier() -> None:
    assert _is_rate_limit_error(RateLimitError("too many requests"))
    # Google's name for the same condition.
    exc = type("ResourceExhausted", (Exception,), {})("quota")
    assert _is_rate_limit_error(exc)
    # Message-shaped: a 429 mentioned alongside rate language.
    assert _is_rate_limit_error(
        RuntimeError("Error code: 429 - rate limit reached for gpt-5-mini")
    )
    # Cause chain is walked.
    wrapped = RuntimeError("provider call failed")
    wrapped.__cause__ = RateLimitError("slow down")
    assert _is_rate_limit_error(wrapped)
    # Ordinary failures are not throttles.
    assert not _is_rate_limit_error(ValueError("bad input"))
    assert not _is_rate_limit_error(RuntimeError("HTTP 500"))


def test_throttle_is_counted_and_propagates() -> None:
    """A rate-limit error increments llm/rate_limited and re-raises."""
    from ontocast.onto.state import BudgetTracker
    from ontocast.tool.llm import use_budget_tracker

    tool = LLMTool(config=_openai_config(cache_enabled=False))
    failing = MagicMock()

    async def _boom(*args, **kwargs):
        raise RateLimitError("Error code: 429")

    failing.ainvoke = _boom
    tool._llm = failing
    tracker = BudgetTracker()

    async def _call():
        with use_budget_tracker(tracker):
            await tool._invoke_cached("prompt")

    with pytest.raises(RateLimitError):
        asyncio.run(_call())
    assert tracker.counters.get("llm/rate_limited") == 1
