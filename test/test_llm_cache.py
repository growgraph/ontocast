"""Tests for LLMTool disk cache behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages.ai import AIMessage
from langchain_core.prompt_values import StringPromptValue

from ontocast.config import LLMConfig, LLMProvider, OpenAIModel
from ontocast.onto.state import BudgetTracker
from ontocast.tool.cache import Cacher
from ontocast.tool.llm import LLMTool


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def llm_config():
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=OpenAIModel.GPT4_O_MINI,
        temperature=0.0,
        cache_enabled=True,
        cache_read_only=False,
    )


async def _make_tool(
    llm_config: LLMConfig,
    cache_dir,
    budget_tracker: BudgetTracker | None = None,
) -> LLMTool:
    shared = Cacher(cache_dir=cache_dir)
    with patch("ontocast.tool.llm.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"answer": 42}'))
        mock_cls.return_value = mock_llm
        tool = await LLMTool.acreate(
            config=llm_config,
            cache=shared,
            budget_tracker=budget_tracker or BudgetTracker(),
        )
    return tool


def test_cache_hit_skips_provider_and_counts_cache_hit(llm_config, cache_dir) -> None:
    tracker = BudgetTracker()

    async def run() -> None:
        tool = await _make_tool(llm_config, cache_dir, tracker)
        await tool.complete("What is 2+2?")
        assert tracker.calls_count == 1
        assert tracker.cache_hits == 0

        await tool.complete("What is 2+2?")
        assert tracker.calls_count == 1
        assert tracker.cache_hits == 1
        assert tool.get_cache_stats()["cache_hits"] == 1

    asyncio.run(run())


def test_cache_read_only_does_not_write(llm_config, cache_dir) -> None:
    llm_config.cache_read_only = True

    async def run() -> None:
        tool = await _make_tool(llm_config, cache_dir)
        await tool.complete("unique prompt alpha")
        assert tool.get_cache_stats()["cache_misses"] == 1

        tool2 = await _make_tool(llm_config, cache_dir)
        await tool2.complete("unique prompt alpha")
        assert tool2.get_cache_stats()["cache_misses"] == 1

    asyncio.run(run())


def test_cache_disabled_always_calls_provider(llm_config, cache_dir) -> None:
    llm_config.cache_enabled = False

    async def run() -> None:
        tool = await _make_tool(llm_config, cache_dir)
        await tool.complete("no cache 1")
        await tool.complete("no cache 1")
        assert tool.get_cache_stats()["cache_misses"] == 2

    asyncio.run(run())


def test_prompt_value_uses_to_string_for_cache_key(llm_config, cache_dir) -> None:
    tracker = BudgetTracker()

    async def run() -> None:
        tool = await _make_tool(llm_config, cache_dir, tracker)
        prompt_text = "Explain ontologies briefly."
        prompt_value = StringPromptValue(text=prompt_text)

        await tool(prompt_value)
        assert tracker.calls_count == 1

        await tool(prompt_value)
        assert tracker.calls_count == 1
        assert tracker.cache_hits == 1

    asyncio.run(run())


def test_config_change_invalidates_cache(llm_config, cache_dir) -> None:
    tracker = BudgetTracker()

    async def run() -> None:
        tool = await _make_tool(llm_config, cache_dir, tracker)
        await tool.complete("same question")
        assert tracker.calls_count == 1

        tool.config.temperature = 0.5
        await tool.complete("same question")
        assert tracker.calls_count == 2

    asyncio.run(run())


def test_get_cache_stats_includes_disk(llm_config, cache_dir) -> None:
    async def run() -> None:
        tool = await _make_tool(llm_config, cache_dir)
        await tool.complete("stats probe")
        stats = tool.get_cache_stats()
        assert stats["cache_misses"] == 1
        assert "disk" in stats
        disk = stats["disk"]
        assert isinstance(disk, dict)
        total_files = disk.get("total_files")
        assert isinstance(total_files, int)
        assert total_files >= 1

    asyncio.run(run())
