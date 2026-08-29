"""Tests for LLM token usage extraction from provider responses."""

import pytest
from langchain_core.messages.ai import AIMessage

from ontocast.onto.token_usage import TokenUsage
from ontocast.tool.llm import (
    _usage_from_llm_result,
    _usage_metadata_from,
    token_usage_from_openai_payload,
)

pytestmark = pytest.mark.unit


def test_usage_from_usage_metadata() -> None:
    message = AIMessage(
        content="hello",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
        },
    )
    usage = _usage_from_llm_result(message)
    assert (usage.input_tokens, usage.output_tokens) == (100, 25)
    assert usage.reasoning_tokens is None


def test_usage_from_legacy_token_usage() -> None:
    message = AIMessage(
        content="hello",
        response_metadata={
            "token_usage": {"prompt_tokens": 50, "completion_tokens": 10},
        },
    )
    usage = _usage_from_llm_result(message)
    assert (usage.input_tokens, usage.output_tokens) == (50, 10)


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
    usage = _usage_from_llm_result(message)
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)


def test_usage_returns_empty_when_not_reported() -> None:
    assert _usage_from_llm_result(AIMessage(content="hello")).is_empty()


def test_usage_returns_empty_for_non_message() -> None:
    assert _usage_from_llm_result("plain string").is_empty()


def test_usage_picks_up_reasoning_and_cache_detail() -> None:
    # A thinking model behind provider-side prompt caching: without the detail
    # keys, the reasoning tokens that dominate output cost are invisible, and
    # cache reads inflate input spend that was billed at a fraction of the rate.
    message = AIMessage(
        content="hello",
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 400,
            "total_tokens": 1400,
            "input_token_details": {"cache_read": 900, "cache_creation": 100},
            "output_token_details": {"reasoning": 350},
        },
    )
    usage = _usage_from_llm_result(message)
    assert usage.reasoning_tokens == 350
    assert usage.cache_read_input_tokens == 900
    assert usage.cache_creation_input_tokens == 100


def test_openai_payload_maps_details() -> None:
    usage = token_usage_from_openai_payload(
        {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 12},
            "prompt_tokens_details": {"cached_tokens": 64},
        }
    )
    assert usage.input_tokens == 80
    assert usage.reasoning_tokens == 12
    assert usage.cache_read_input_tokens == 64


def test_openai_payload_empty_without_totals() -> None:
    assert token_usage_from_openai_payload(None).is_empty()
    assert token_usage_from_openai_payload({"prompt_tokens": 5}).is_empty()


def test_usage_metadata_round_trips() -> None:
    # A cache hit rebuilds the AIMessage from this, so anything reading usage
    # off the message -- a LangChain-native tracer, say -- sees a replayed call
    # exactly as it saw the live one.
    original = TokenUsage(
        input_tokens=1000,
        output_tokens=400,
        reasoning_tokens=350,
        cache_read_input_tokens=900,
    )
    message = AIMessage(content="x", usage_metadata=_usage_metadata_from(original))
    assert _usage_from_llm_result(message) == original


def test_usage_metadata_is_none_without_totals() -> None:
    assert _usage_metadata_from(TokenUsage(reasoning_tokens=5)) is None
