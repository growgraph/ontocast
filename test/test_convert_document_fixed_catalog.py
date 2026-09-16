"""convert_document validates fixed-single-ontology catalog id when applicable."""

import pytest

from ontocast.agent.convert_document import convert_document
from ontocast.onto.enum import OntologyContextMode, Status
from ontocast.onto.state import AgentState
from ontocast.tool.converter import ConverterTool
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


def test_convert_document_fails_when_fixed_mode_missing_catalog_id() -> None:
    state = AgentState(
        raw_input={"input.json": b'{"text": "hello"}'},
        ontology_context_mode=OntologyContextMode.FIXED_SINGLE_ONTOLOGY,
        ontology_context_fixed_ontology_id="",
    )
    tools = ToolBox.__new__(ToolBox)
    tools.converter = ConverterTool.model_construct(supported_extensions=set())
    result = convert_document(state, tools)
    assert result.status == Status.FAILED
    assert "ontology_context_fixed_ontology_id" in (result.failure_reason or "")
