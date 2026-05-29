"""Tests for OpenAI batch cache import helpers."""

import json

from ontocast.config import LLMConfig, LLMProvider, OpenAIModel
from ontocast.tool.cache import Cacher
from ontocast.tool.llm_batch import (
    _llm_cache_config_dict,
    import_openai_batch_output_jsonl,
    write_openai_chat_batch_jsonl,
)


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
        "hello", subdirectory="llm", config=_llm_cache_config_dict(llm_config)
    )
    assert isinstance(tool_cache, dict)
    assert tool_cache["content"] == "cached batch reply"
