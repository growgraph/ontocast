"""Tests for LLM token usage extraction from provider responses."""

from langchain_core.messages.ai import AIMessage

from ontocast.tool.llm import _usage_from_llm_result


def test_usage_from_usage_metadata() -> None:
    message = AIMessage(
        content="hello",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
        },
    )
    assert _usage_from_llm_result(message) == (100, 25)


def test_usage_from_legacy_token_usage() -> None:
    message = AIMessage(
        content="hello",
        response_metadata={
            "token_usage": {"prompt_tokens": 50, "completion_tokens": 10},
        },
    )
    assert _usage_from_llm_result(message) == (50, 10)


def test_usage_metadata_takes_priority_over_legacy() -> None:
    message = AIMessage(
        content="hello",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
        response_metadata={
            "token_usage": {"prompt_tokens": 99, "completion_tokens": 99},
        },
    )
    assert _usage_from_llm_result(message) == (10, 5)


def test_usage_returns_none_when_not_reported() -> None:
    message = AIMessage(content="hello")
    assert _usage_from_llm_result(message) == (None, None)


def test_usage_returns_none_for_non_message() -> None:
    assert _usage_from_llm_result("plain string") == (None, None)
