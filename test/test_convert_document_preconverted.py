"""A state that already carries a converted document is not converted again."""

import pytest

from ontocast.agent.convert_document import convert_document
from ontocast.onto.docling_helpers import plain_text_to_docling_doc
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


class _NoConverter:
    supported_extensions = {".pdf"}

    def __call__(self, content: bytes, *, filename: str | None = None):
        raise AssertionError("a converted document must not be converted again")


def _tools() -> ToolBox:
    tools = ToolBox.__new__(ToolBox)
    tools.converter = _NoConverter()  # ty: ignore[invalid-assignment]
    return tools


def test_kept_document_skips_conversion_and_keeps_its_namespace() -> None:
    doc = plain_text_to_docling_doc("Films were annealed at 100 C.", "paper.pdf")
    first = AgentState()
    first.set_docling_doc(doc)

    kept = doc.model_dump_json()
    rerun = AgentState(docling_doc=type(doc).model_validate_json(kept))
    result = convert_document(rerun, _tools())

    assert result.status == Status.SUCCESS
    assert result.doc_hid == first.doc_hid
    assert result.doc_namespace == first.doc_namespace
