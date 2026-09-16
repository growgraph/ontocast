"""LLM_JSON_MODE: the setting, and the prompt precondition it depends on."""

from __future__ import annotations

import pytest

from ontocast.config.settings import LLMConfig
from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.model import (
    FactsCritiqueReport,
    FactsRenderReport,
    GraphUpdateRenderReport,
    OntologyCritiqueReport,
    OntologySelectorReport,
)
from ontocast.prompt.llm_json_schema import format_instructions_for_model

pytestmark = pytest.mark.unit

_REPORT_MODELS = [
    FactsRenderReport,
    GraphUpdateRenderReport,
    FactsCritiqueReport,
    OntologyCritiqueReport,
    OntologySelectorReport,
]


def test_json_mode_is_off_by_default() -> None:
    """Enabling it is a deployment decision, not a silent default change."""
    assert LLMConfig().json_mode is False


@pytest.mark.parametrize("model", _REPORT_MODELS, ids=lambda m: m.__name__)
@pytest.mark.parametrize("fmt", list(LLMGraphFormat), ids=lambda f: f.value)
def test_prompt_json_mode_precondition(model, fmt) -> None:
    """Every format instruction must name JSON, in both graph formats.

    OpenAI rejects a ``response_format=json_object`` request unless the word
    "JSON" appears somewhere in the prompt. That precondition is a property of
    the prompt set, so it belongs in CI rather than being discovered by a 400
    from the provider. Turtle is included deliberately: it only changes the
    graph *fields*, the envelope stays JSON.
    """
    instructions = format_instructions_for_model(model, fmt)
    assert "JSON" in instructions
