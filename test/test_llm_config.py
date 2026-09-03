"""Tests for LLM provider/model configuration validation."""

import logging
from typing import Literal

import pytest
from pydantic import ValidationError

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


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_llm_config_accepts_each_reasoning_effort(
    effort: Literal["minimal", "low", "medium", "high"],
) -> None:
    config = LLMConfig(provider=LLMProvider.OPENAI, reasoning_effort=effort)
    assert config.reasoning_effort == effort


def test_llm_config_rejects_an_unknown_reasoning_effort() -> None:
    # A typo would otherwise reach the provider as a request error on the
    # first call, after the ontology sync has already been paid for.
    with pytest.raises(ValidationError):
        LLMConfig.model_validate(
            {"provider": LLMProvider.OPENAI, "reasoning_effort": "max"}
        )


def test_llm_config_thinking_budget_allows_off_and_model_chosen() -> None:
    # 0 turns thinking off where the model allows it; -1 hands the choice to
    # the model. Anything lower is not a value the provider defines.
    assert LLMConfig(thinking_budget=0).thinking_budget == 0
    assert LLMConfig(thinking_budget=-1).thinking_budget == -1
    with pytest.raises(ValidationError):
        LLMConfig(thinking_budget=-2)


def test_reasoning_knobs_are_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("LLM_THINKING_BUDGET", "1024")
    config = LLMConfig()
    assert config.reasoning_effort == "low"
    assert config.thinking_budget == 1024


def test_workers_above_inflight_warn_at_construction(caplog) -> None:
    """PARALLEL_WORKERS past LLM_MAX_INFLIGHT only queue; say so up front.

    A unit worker never issues two calls at once, so the provider concurrency
    a document reaches is min(workers, inflight). Each surplus worker holds a
    unit slot and its memory while it waits -- a cost with no return that
    nothing at runtime names beyond a growing llm/inflight_wait.
    """
    from ontocast.toolbox import warn_if_workers_exceed_inflight

    config = Config()
    config.server.parallel_workers = 16
    config.tool_config.llm_config.llm_max_inflight = 8
    with caplog.at_level(logging.WARNING, logger="ontocast.toolbox"):
        assert warn_if_workers_exceed_inflight(config) is True
    assert "PARALLEL_WORKERS=16 exceeds LLM_MAX_INFLIGHT=8" in caplog.text

    caplog.clear()
    config.server.parallel_workers = 8
    with caplog.at_level(logging.WARNING, logger="ontocast.toolbox"):
        assert warn_if_workers_exceed_inflight(config) is False
    assert caplog.text == ""
