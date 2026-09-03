"""Tests for LLMTool provider setup (mocked, no live API)."""

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

from ontocast.config import (
    ClaudeModel,
    GeminiModel,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    OpenAIModel,
)
from ontocast.tool.llm import LLMTool

pytestmark = pytest.mark.unit


def test_setup_openai() -> None:
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=OpenAIModel.GPT4_O_MINI,
        api_key="test-key",
    )
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        tool = LLMTool(config=config)
        asyncio.run(tool.setup())
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == OpenAIModel.GPT4_O_MINI


def test_setup_anthropic() -> None:
    config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name=ClaudeModel.CLAUDE_SONNET_4,
        api_key="test-key",
        base_url="https://api.example.com",
    )
    with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        tool = LLMTool(config=config)
        asyncio.run(tool.setup())
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == ClaudeModel.CLAUDE_SONNET_4
        assert kwargs["anthropic_api_url"] == "https://api.example.com"


def test_setup_google() -> None:
    config = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name=GeminiModel.GEMINI_2_0_FLASH,
        api_key="test-key",
    )
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        tool = LLMTool(config=config)
        asyncio.run(tool.setup())
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == GeminiModel.GEMINI_2_0_FLASH
        assert kwargs["google_api_key"] == "test-key"


def test_setup_ollama() -> None:
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name=OllamaModel.LLAMA3_1,
        base_url="http://localhost:11434",
    )
    with patch("langchain_ollama.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock()
        tool = LLMTool(config=config)
        asyncio.run(tool.setup())
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == OllamaModel.LLAMA3_1


def test_setup_openai_passes_reasoning_effort_to_the_client() -> None:
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=OpenAIModel.GPT4_O_MINI,
        api_key="test-key",
        reasoning_effort="low",
    )
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        asyncio.run(LLMTool(config=config).setup())
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["reasoning_effort"] == "low"
        assert "thinking_budget" not in kwargs


def test_setup_openai_leaves_reasoning_effort_to_the_client_when_unset() -> None:
    # The client's own default is the provider's; passing None would override
    # it with an explicit "no preference" the client may serialise.
    config = LLMConfig(
        provider=LLMProvider.OPENAI, model_name=OpenAIModel.GPT4_O_MINI, api_key="k"
    )
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        asyncio.run(LLMTool(config=config).setup())
        assert "reasoning_effort" not in mock_cls.call_args.kwargs


def test_setup_google_passes_thinking_budget_to_the_client() -> None:
    config = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name=GeminiModel.GEMINI_2_0_FLASH,
        api_key="test-key",
        thinking_budget=0,
    )
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        asyncio.run(LLMTool(config=config).setup())
        kwargs = mock_cls.call_args.kwargs
        # 0 is a value (thinking off), not an absence.
        assert kwargs["thinking_budget"] == 0
        assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize(
    ("provider", "model_name", "patched", "knob"),
    [
        (
            LLMProvider.ANTHROPIC,
            ClaudeModel.CLAUDE_SONNET_4,
            "langchain_anthropic.ChatAnthropic",
            {"reasoning_effort": "low"},
        ),
        (
            LLMProvider.OLLAMA,
            OllamaModel.LLAMA3_1,
            "langchain_ollama.ChatOllama",
            {"thinking_budget": 256},
        ),
        (
            LLMProvider.OPENAI,
            OpenAIModel.GPT4_O_MINI,
            "langchain_openai.ChatOpenAI",
            {"thinking_budget": 256},
        ),
        (
            LLMProvider.GOOGLE,
            GeminiModel.GEMINI_2_0_FLASH,
            "langchain_google_genai.ChatGoogleGenerativeAI",
            {"reasoning_effort": "low"},
        ),
    ],
)
def test_setup_warns_and_ignores_the_other_providers_reasoning_knob(
    provider, model_name, patched, knob, caplog
) -> None:
    """Each knob is one provider's spelling; the other spelling is a no-op.

    Silently, the run would bill full reasoning while its manifest recorded a
    budget that never applied.
    """
    config = LLMConfig(provider=provider, model_name=model_name, api_key="k", **knob)
    with patch(patched) as mock_cls:
        mock_cls.return_value = MagicMock()
        with caplog.at_level(logging.WARNING, logger="ontocast.tool.llm"):
            asyncio.run(LLMTool(config=config).setup())
        assert not (mock_cls.call_args.kwargs.keys() & knob.keys())
    assert "is ignored by the" in caplog.text
