"""The LLM wire-format default is declared in four places; keep them agreeing.

``ServerConfig``, ``AgentState``, ``UnitState`` and the ``llm_graph_format_ctx``
ContextVar each carry their own default. Whichever one the entry point happens
to consult decides how the LLM is prompted and how its graph payloads are
parsed, so a partial edit does not fail loudly -- it silently makes the
effective default depend on whether you came in via the HTTP API, the batch
CLI, ``run_unit_pipeline``, or a bare ``model_validate``. That is precisely how
the flip to JSON-LD could have gone wrong.

These assert the *declared* defaults rather than instantiating, because
``ServerConfig`` is a ``BaseSettings``: instantiating it reads the ambient
environment and would make the test pass or fail based on the developer's
shell.
"""

import pytest
from pydantic import BaseModel

from ontocast.config.settings import ServerConfig
from ontocast.onto.enum import LLMGraphFormat, OntologyContextMode, RenderMode
from ontocast.onto.llm_graph_payload import llm_graph_format_ctx
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitState

pytestmark = pytest.mark.unit


def _declared_default(model: type[BaseModel], field: str) -> object:
    return model.model_fields[field].default


def test_wire_format_default_is_jsonld_everywhere() -> None:
    for model in (ServerConfig, AgentState, UnitState):
        assert _declared_default(model, "llm_graph_format") is LLMGraphFormat.JSONLD, (
            f"{model.__name__} disagrees on the default LLM wire format"
        )
    assert llm_graph_format_ctx.get() is LLMGraphFormat.JSONLD


def test_mode_selector_defaults_agree_between_config_and_state() -> None:
    """Same drift risk for the two pipeline mode selectors."""
    for model in (ServerConfig, AgentState):
        assert _declared_default(model, "render_mode") is RenderMode.ONTOLOGY_AND_FACTS
        assert (
            _declared_default(model, "ontology_context_mode")
            is OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
        )
