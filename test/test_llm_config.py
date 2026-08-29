"""Tests for LLM provider/model configuration validation."""

import logging

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

pytestmark = pytest.mark.unit


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
def test_llm_config_warns_but_accepts_mismatched_provider_and_model(
    provider: LLMProvider, model_name, caplog
) -> None:
    # Not an error: a cross-provider name is legitimate behind an
    # OpenAI-compatible base_url, and indistinguishable from a typo here.
    with caplog.at_level(logging.WARNING, logger="ontocast.config.settings"):
        config = LLMConfig(provider=provider, model_name=model_name)
    assert config.model_name == model_name
    assert "is not a known" in caplog.text


@pytest.mark.parametrize(
    "model_name",
    ["kimi-k3", "qwen3-max", "some-model-released-next-week"],
)
def test_llm_config_accepts_arbitrary_model_names(model_name: str, caplog) -> None:
    # The presets are not a whitelist. Hosted Qwen/Kimi and anything newer than
    # this package must reach the provider through base_url without a release.
    with caplog.at_level(logging.WARNING, logger="ontocast.config.settings"):
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name=model_name,
            base_url="https://api.moonshot.ai/v1",
        )
    assert config.model_name == model_name
    assert "is not a known" in caplog.text


def test_llm_config_does_not_warn_for_a_known_preset(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="ontocast.config.settings"):
        LLMConfig(provider=LLMProvider.OPENAI, model_name=OpenAIModel.GPT4_O_MINI)
    assert caplog.text == ""


def test_a_preset_named_as_a_plain_string_normalises_without_warning(caplog) -> None:
    # LLM_MODEL_NAME always arrives as a string, and with `str` in the union
    # pydantic has no reason to prefer the enum -- so without normalisation
    # every env-configured run warned about a model that is a known preset.
    with caplog.at_level(logging.WARNING, logger="ontocast.config.settings"):
        config = LLMConfig(provider=LLMProvider.OLLAMA, model_name="kimi-k3")
    assert config.model_name is OllamaModel.KIMI_K3
    assert caplog.text == ""


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
