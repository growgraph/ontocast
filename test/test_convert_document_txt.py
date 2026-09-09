"""convert_document handles .txt uploads as plain text (regression for #67)."""

import pytest

from ontocast.agent.convert_document import convert_document
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState
from ontocast.tool.converter import ConverterTool
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


def test_convert_document_txt_is_plain_text_not_json() -> None:
    # A plain-text .txt payload is not valid JSON; it must not be json.loads'd.
    state = AgentState(
        raw_input={"note.txt": b"Hello world. This is plain text, not JSON."}
    )
    tools = ToolBox.__new__(ToolBox)
    tools.converter = ConverterTool.model_construct(supported_extensions=set())

    result = convert_document(state, tools)

    assert result.status == Status.SUCCESS
    assert result.docling_doc is not None
