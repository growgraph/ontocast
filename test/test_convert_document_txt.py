"""convert_document handles .txt uploads as plain text (regression for #67)."""

from types import SimpleNamespace

from ontocast.agent.convert_document import convert_document
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox


def test_convert_document_txt_is_plain_text_not_json() -> None:
    # A plain-text .txt payload is not valid JSON; it must not be json.loads'd.
    state = AgentState(
        raw_input={"note.txt": b"Hello world. This is plain text, not JSON."}
    )
    tools = ToolBox.__new__(ToolBox)
    tools.converter = SimpleNamespace(supported_extensions=())

    result = convert_document(state, tools)

    assert result.status == Status.SUCCESS
    assert result.docling_doc is not None
