"""Tests for OpenAI batch cache import helpers."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ontocast.config import LLMConfig, LLMProvider, OpenAIModel
from ontocast.onto.state import BudgetTracker
from ontocast.tool.cache import Cacher
from ontocast.tool.llm import LLMTool, llm_cache_config
from ontocast.tool.llm_batch import (
    import_openai_batch_output_jsonl,
    write_openai_chat_batch_jsonl,
)

pytestmark = pytest.mark.unit


def test_batch_import_is_readable_by_the_llm_tool(tmp_path) -> None:
    """A prewarmed entry must actually be hit by the server that reads it.

    Regression test: the batch importer built its own cache-key config and
    dropped ``base_url`` when it was None -- the default. Every imported entry
    hashed differently from what LLMTool looked up, so the whole prewarm
    feature wrote entries that were never read. Asserting through the real
    read path is the point; comparing the importer against itself passes even
    when the two disagree.
    """
    output_path = tmp_path / "batch_out.jsonl"
    cache_dir = tmp_path / "cache"
    prompt = "what is the capital of France?"

    output_path.write_text(
        json.dumps(
            {
                "custom_id": "req-1",
                "response": {"body": {"choices": [{"message": {"content": "Paris"}}]}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=OpenAIModel.GPT4_O_MINI,
        temperature=0.0,
    )
    assert llm_config.base_url is None, "the regression only shows with base_url unset"

    shared = Cacher(cache_dir=cache_dir)
    assert (
        import_openai_batch_output_jsonl(
            output_path,
            shared_cache=shared,
            llm_config=llm_config,
            custom_id_to_cache_key={"req-1": prompt},
        )
        == 1
    )

    tracker = BudgetTracker()

    async def run() -> None:
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(
                side_effect=AssertionError("provider called despite a prewarmed entry")
            )
            mock_cls.return_value = mock_llm
            tool = await LLMTool.acreate(
                config=llm_config,
                cache=Cacher(cache_dir=cache_dir),
                budget_tracker=tracker,
            )
        assert await tool.complete(prompt) == "Paris"
        assert tracker.cache_hits == 1
        assert tracker.calls_count == 0

    asyncio.run(run())


def test_write_and_import_openai_batch_jsonl(tmp_path) -> None:
    input_path = tmp_path / "batch_in.jsonl"
    output_path = tmp_path / "batch_out.jsonl"
    cache_dir = tmp_path / "cache"

    write_openai_chat_batch_jsonl(
        [
            {
                "custom_id": "req-1",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            }
        ],
        input_path,
    )
    assert input_path.exists()

    output_path.write_text(
        json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": "cached batch reply"}}]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=OpenAIModel.GPT4_O_MINI,
        temperature=0.0,
    )
    shared = Cacher(cache_dir=cache_dir)
    written = import_openai_batch_output_jsonl(
        output_path,
        shared_cache=shared,
        llm_config=llm_config,
        custom_id_to_cache_key={"req-1": "hello"},
    )
    assert written == 1
    tool_cache = shared.get(
        "hello", subdirectory="llm", config=llm_cache_config(llm_config)
    )
    assert isinstance(tool_cache, dict)
    assert tool_cache["content"] == "cached batch reply"
