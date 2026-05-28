"""Tests for LLMTool provider setup (mocked, no live API)."""

import asyncio
from unittest.mock import MagicMock, patch

from ontocast.config import (
    ClaudeModel,
    GeminiModel,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    OpenAIModel,
)
from ontocast.tool.llm import LLMTool


def test_setup_openai() -> None:
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=OpenAIModel.GPT4_O_MINI,
        api_key="test-key",
    )
    with patch("ontocast.tool.llm.ChatOpenAI") as mock_cls:
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
    with patch("ontocast.tool.llm.ChatAnthropic") as mock_cls:
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
    with patch("ontocast.tool.llm.ChatGoogleGenerativeAI") as mock_cls:
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
    with patch("ontocast.tool.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock()
        tool = LLMTool(config=config)
        asyncio.run(tool.setup())
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == OllamaModel.LLAMA3_1
