"""What a first-time user sees for a missing key or an unsupported file."""

import click
import pytest

from ontocast.agent.convert_document import convert_document
from ontocast.cli.server import _bootstrap_tools
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState, InputDocument
from ontocast.tool.converter import ConverterTool
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


def test_a_missing_api_key_is_a_cli_error_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(click.ClickException, match="LLM_API_KEY"):
        _bootstrap_tools(tenant=None, project=None, wipe_vector_store=None)


def test_an_unsupported_file_names_what_is_accepted() -> None:
    state = AgentState(
        input_document=InputDocument(filename="slides.key", data=b"binary")
    )
    tools = ToolBox.__new__(ToolBox)
    tools.converter = ConverterTool.model_construct(supported_extensions={".pdf"})

    result = convert_document(state, tools)

    assert result.status == Status.FAILED
    reason = result.failure_reason or ""
    assert "slides.key" in reason
    assert ".pdf" in reason
    assert ".txt" in reason
    assert "doc-processing" in reason
