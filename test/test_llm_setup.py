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
        model_name=ClaudeModel.CLAUDE_SONNET_4_6,
        api_key="test-key",
        base_url="https://api.example.com",
    )
    with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        tool = LLMTool(config=config)
        asyncio.run(tool.setup())
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == ClaudeModel.CLAUDE_SONNET_4_6
        assert kwargs["anthropic_api_url"] == "https://api.example.com"


def test_setup_google() -> None:
    config = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name=GeminiModel.GEMINI_3_7_FLASH,
        api_key="test-key",
    )
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        tool = LLMTool(config=config)
        asyncio.run(tool.setup())
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == GeminiModel.GEMINI_3_7_FLASH
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
        model_name="gemini-2.5-flash",
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
            ClaudeModel.CLAUDE_SONNET_4_6,
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
    ],
)
def test_setup_warns_and_ignores_a_knob_the_provider_does_not_read(
    provider, model_name, patched, knob, caplog
) -> None:
    """A knob the configured provider does not read is a no-op.

    Silently, the run would bill full reasoning while its manifest recorded a
    budget that never applied.
    """
    config = LLMConfig(provider=provider, model_name=model_name, api_key="k", **knob)
    with patch(patched) as mock_cls:
        mock_cls.return_value = MagicMock()
        with caplog.at_level(logging.WARNING, logger="ontocast.tool.llm"):
            asyncio.run(LLMTool(config=config).setup())
        assert not (mock_cls.call_args.kwargs.keys() & knob.keys())
    assert "is ignored:" in caplog.text


def test_setup_google_passes_reasoning_effort_as_the_thinking_level() -> None:
    """Gemini 3+ reads the discrete level, which the client aliases.

    ``reasoning_effort`` on ChatGoogleGenerativeAI carries
    ``alias="thinking_level"`` and is routed into ThinkingConfig, so this is
    the same knob OpenAI reads under its own name -- not the integer budget.
    """
    config = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name=GeminiModel.GEMINI_3_5_FLASH,
        api_key="test-key",
        reasoning_effort="minimal",
    )
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        asyncio.run(LLMTool(config=config).setup())
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["reasoning_effort"] == "minimal"
        assert "thinking_budget" not in kwargs


def test_setup_drops_the_thinking_budget_on_a_gemini_3_model(caplog) -> None:
    """The integer budget is superseded from Gemini 3 on.

    Forwarding it would hand the API a parameter of the wrong generation while
    the warning claims it was ignored.
    """
    config = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name=GeminiModel.GEMINI_3_5_FLASH,
        api_key="test-key",
        thinking_budget=0,
    )
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        with caplog.at_level(logging.WARNING, logger="ontocast.tool.llm"):
            asyncio.run(LLMTool(config=config).setup())
        assert "thinking_budget" not in mock_cls.call_args.kwargs
    assert "LLM_REASONING_EFFORT" in caplog.text


def test_setup_keeps_the_thinking_budget_on_a_gemini_2_5_model() -> None:
    """The generation that still reads it is unaffected by the 3+ handling."""
    config = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-2.5-flash",
        api_key="test-key",
        thinking_budget=0,
    )
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        asyncio.run(LLMTool(config=config).setup())
        assert mock_cls.call_args.kwargs["thinking_budget"] == 0


@pytest.mark.parametrize(
    "model_name",
    [OpenAIModel.GPT5, OpenAIModel.GPT5_MINI, OpenAIModel.GPT5_NANO],
)
def test_setup_pins_temperature_for_the_gpt_5_series(model_name) -> None:
    """The series rejects any temperature but 1.0, so it is overridden."""
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=model_name,
        api_key="k",
        temperature=0.0,
    )
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        asyncio.run(LLMTool(config=config).setup())
        assert mock_cls.call_args.kwargs["temperature"] == 1.0


@pytest.mark.parametrize(
    "model_name",
    [OpenAIModel.GPT5_4, OpenAIModel.GPT5_4_MINI, OpenAIModel.GPT5_4_PRO],
)
def test_setup_leaves_temperature_alone_for_later_families(model_name) -> None:
    """A name that merely starts like the pinned series is not the series.

    The override is anchored, so gpt-5.4 keeps the configured temperature.
    Forcing it to 1.0 would silently unpin decoding on every arm that asks
    for deterministic output -- and the mutation also reaches the cache key
    and the run manifest, so the substitution would not be visible after
    the fact.
    """
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=model_name,
        api_key="k",
        temperature=0.0,
    )
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        asyncio.run(LLMTool(config=config).setup())
        assert mock_cls.call_args.kwargs["temperature"] == 0.0
        assert config.temperature == 0.0


@pytest.mark.parametrize(
    "model_name",
    [
        OpenAIModel.GPT5_6_LUNA,
        OpenAIModel.GPT5_6_TERRA,
        OpenAIModel.GPT6_LUNA,
        OpenAIModel.GPT6_ASTRA,
        "gpt-7-nova",
    ],
)
def test_setup_omits_temperature_while_a_newer_openai_model_reasons(
    model_name,
) -> None:
    """GPT-5.5 onward rejects a temperature unless reasoning is off."""
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=model_name,
        api_key="k",
        temperature=0.0,
    )
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        asyncio.run(LLMTool(config=config).setup())
        assert mock_cls.call_args.kwargs["temperature"] is None
        # The configured value stays what the cache key and manifest record.
        assert config.temperature == 0.0


@pytest.mark.parametrize("model_name", [OpenAIModel.GPT5_6_LUNA, OpenAIModel.GPT6_SOL])
def test_setup_keeps_temperature_at_reasoning_effort_none(model_name) -> None:
    """With reasoning off the model samples, so a pinned temperature applies."""
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=model_name,
        api_key="k",
        temperature=0.0,
        reasoning_effort="none",
    )
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        asyncio.run(LLMTool(config=config).setup())
        assert mock_cls.call_args.kwargs["temperature"] == 0.0


@pytest.mark.parametrize(
    ("model_name", "sent"),
    [
        (ClaudeModel.CLAUDE_OPUS_5_5, False),
        (ClaudeModel.CLAUDE_OPUS_4_7, False),
        (ClaudeModel.CLAUDE_SONNET_5_5, False),
        (ClaudeModel.CLAUDE_FABLE_5_1, False),
        (ClaudeModel.CLAUDE_OPUS_4_6, True),
        (ClaudeModel.CLAUDE_SONNET_4_6, True),
        (ClaudeModel.CLAUDE_HAIKU_4_5, True),
        # A dated snapshot's date is not a minor version.
        ("claude-opus-4-20250514", True),
    ],
)
def test_setup_sends_temperature_only_to_claude_models_that_accept_it(
    model_name, sent: bool
) -> None:
    config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name=model_name,
        api_key="k",
        temperature=0.0,
    )
    with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        asyncio.run(LLMTool(config=config).setup())
        temperature = mock_cls.call_args.kwargs["temperature"]
        assert temperature == (0.0 if sent else None)


def test_default_model_is_a_preset() -> None:
    assert LLMConfig().model_name == OpenAIModel.GPT5_6_LUNA
