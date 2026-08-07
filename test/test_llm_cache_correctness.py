"""Tests for LLM cache-key completeness and provider content normalisation."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from ontocast.config import LLMConfig, LLMProvider, OllamaModel, OpenAIModel
from ontocast.onto.state import BudgetTracker
from ontocast.tool.cache import Cacher
from ontocast.tool.llm import LLMTool


async def _tool(config: LLMConfig, cache_dir, response, tracker=None) -> LLMTool:
    """An LLMTool whose provider returns ``response`` (or raises, if given one)."""
    with (
        patch("langchain_openai.ChatOpenAI") as mock_openai,
        patch("langchain_ollama.ChatOllama") as mock_ollama,
    ):
        mock_llm = MagicMock()
        if isinstance(response, BaseException):
            mock_llm.ainvoke = AsyncMock(side_effect=response)
        else:
            mock_llm.ainvoke = AsyncMock(return_value=response)
        mock_openai.return_value = mock_llm
        mock_ollama.return_value = mock_llm
        return await LLMTool.acreate(
            config=config,
            cache=Cacher(cache_dir=cache_dir),
            budget_tracker=tracker or BudgetTracker(),
        )


class Answer(BaseModel):
    """Minimal structured-output schema."""

    answer: int


def test_complete_normalises_provider_content_blocks(tmp_path) -> None:
    """Anthropic and Gemini return typed blocks, not a bare string.

    ``complete`` used to ``str()`` the list, returning a Python repr like
    ``"[{'type': 'text', ...}]"`` -- and then cache that repr, so the damage
    persisted across runs.
    """
    blocks: list[str | dict[Any, Any]] = [{"type": "text", "text": "Paris"}]
    config = LLMConfig(provider=LLMProvider.OPENAI, cache_enabled=True)

    async def run() -> None:
        tool = await _tool(config, tmp_path / "cache", AIMessage(content=blocks))
        assert await tool.complete("capital?") == "Paris"

        # And the cached value is the normalised string, not the repr.
        cached = tool.cache.get("capital?", config=tool._cache_config_dict())
        assert isinstance(cached, dict)
        assert cached["content"] == "Paris"

    asyncio.run(run())


def test_extract_parses_content_blocks(tmp_path) -> None:
    blocks: list[str | dict[Any, Any]] = [{"type": "text", "text": '{"answer": 42}'}]
    config = LLMConfig(provider=LLMProvider.OPENAI, cache_enabled=True)

    async def run() -> None:
        tool = await _tool(config, tmp_path / "cache", AIMessage(content=blocks))
        assert (await tool.extract("how many?", Answer)).answer == 42
        # Second call is served from cache and must parse identically.
        assert (await tool.extract("how many?", Answer)).answer == 42

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value", [("num_ctx", 32768), ("num_predict", 512), ("think", True)]
)
def test_ollama_generation_knobs_are_part_of_the_cache_key(
    tmp_path, field, value
) -> None:
    """These bound reasoning and output length, so they change the response."""
    tracker = BudgetTracker()
    cache_dir = tmp_path / "cache"
    base = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name=OllamaModel.QWEN3_6_LATEST,
        base_url="http://localhost:11434",
    )

    async def run() -> None:
        tool = await _tool(base, cache_dir, AIMessage(content="first"), tracker)
        await tool.complete("same prompt")
        assert tracker.calls_count == 1

        setattr(tool.config, field, value)
        await tool.complete("same prompt")
        assert tracker.calls_count == 2, f"{field} must discriminate cache entries"

    asyncio.run(run())


def test_cache_hit_replays_response_metadata(tmp_path) -> None:
    """A cached call must be behaviourally identical to a fresh one."""
    config = LLMConfig(provider=LLMProvider.OPENAI, cache_enabled=True)
    metadata = {"finish_reason": "length"}

    async def run() -> None:
        cache_dir = tmp_path / "cache"
        tool = await _tool(
            config, cache_dir, AIMessage(content="hi", response_metadata=metadata)
        )
        fresh = await tool("prompt")
        assert fresh.response_metadata["finish_reason"] == "length"

        cached_tool = await _tool(
            config, cache_dir, AssertionError("provider must not be called")
        )
        cached = await cached_tool("prompt")
        assert cached.response_metadata["finish_reason"] == "length"

    asyncio.run(run())


def test_inflight_semaphore_survives_a_second_event_loop(tmp_path) -> None:
    """asyncio.Semaphore binds to a loop on its first *contended* acquire.

    A single process-wide semaphore therefore raised "bound to a different
    event loop" on the second ``asyncio.run`` in a process as soon as calls
    overlapped.
    """
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=OpenAIModel.GPT4_O_MINI,
        cache_enabled=False,
    )
    config.llm_max_inflight = 1

    async def contend(index: int) -> None:
        async def slow(*args, **kwargs):
            await asyncio.sleep(0)
            return AIMessage(content="ok")

        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=slow)
            mock_cls.return_value = mock_llm
            tool = await LLMTool.acreate(
                config=config, cache=Cacher(cache_dir=tmp_path / "cache")
            )
        # Concurrent calls with max_inflight=1 force the semaphore to wait,
        # which is what binds it to the running loop.
        await asyncio.gather(*(tool.complete(f"p{index}-{n}") for n in range(4)))

    asyncio.run(contend(0))
    asyncio.run(contend(1))
