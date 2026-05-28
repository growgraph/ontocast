"""Tests for LLM provider/model configuration validation."""

import pytest

from ontocast.config import (
    ClaudeModel,
    Config,
    GeminiModel,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    OpenAIModel,
)


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        (LLMProvider.OPENAI, OpenAIModel.GPT4_O_MINI),
        (LLMProvider.OLLAMA, OllamaModel.LLAMA3_1),
        (LLMProvider.ANTHROPIC, ClaudeModel.CLAUDE_SONNET_4),
        (LLMProvider.GOOGLE, GeminiModel.GEMINI_2_0_FLASH),
    ],
)
def test_llm_config_accepts_matching_provider_and_model(
    provider: LLMProvider, model_name
) -> None:
    config = LLMConfig(provider=provider, model_name=model_name)
    assert config.provider == provider
    assert config.model_name == model_name


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        (LLMProvider.OPENAI, OllamaModel.LLAMA3_1),
        (LLMProvider.OLLAMA, OpenAIModel.GPT4_O_MINI),
        (LLMProvider.ANTHROPIC, GeminiModel.GEMINI_2_0_FLASH),
        (LLMProvider.GOOGLE, ClaudeModel.CLAUDE_SONNET_4),
    ],
)
def test_llm_config_rejects_mismatched_provider_and_model(
    provider: LLMProvider, model_name
) -> None:
    with pytest.raises(ValueError, match="not compatible"):
        LLMConfig(provider=provider, model_name=model_name)


@pytest.mark.parametrize(
    "provider",
    [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GOOGLE],
)
def test_validate_llm_config_requires_api_key(provider: LLMProvider) -> None:
    config = Config()
    config.tool_config.llm_config = LLMConfig(
        provider=provider,
        model_name=_default_model_for(provider),
        api_key=None,
    )
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        config.validate_llm_config()


def test_validate_llm_config_ollama_does_not_require_api_key() -> None:
    config = Config()
    config.tool_config.llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name=OllamaModel.LLAMA3_1,
        api_key=None,
        base_url="http://localhost:11434",
    )
    config.validate_llm_config()


def _default_model_for(provider: LLMProvider):
    if provider == LLMProvider.OPENAI:
        return OpenAIModel.GPT4_O_MINI
    if provider == LLMProvider.ANTHROPIC:
        return ClaudeModel.CLAUDE_SONNET_4
    return GeminiModel.GEMINI_2_0_FLASH
