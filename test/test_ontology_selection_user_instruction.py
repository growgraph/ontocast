from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from rdflib import URIRef

from ontocast.agent.convert_document import convert_document
from ontocast.agent.select_ontology_catalog import select_catalog_ontology_for_excerpt
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph import context_resolver as cr
from ontocast.stategraph.context_resolver import resolve_unit_ontology_context
from ontocast.tool.llm import LLMTool
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.toolbox import ToolBox


class _FakeOntologyManager(OntologyManager):
    def __init__(self, ontologies: list[Ontology]) -> None:
        super().__init__()
        self._ontologies = ontologies

    @property
    def has_ontologies(self) -> bool:
        return bool(self._ontologies)

    @property
    def ontologies(self) -> list[Ontology]:
        return self._ontologies


def test_convert_document_sets_ontology_selection_user_instruction_from_json() -> None:
    payload = {
        "text": "Hello world",
        "ontology_selection_user_instruction": "Prefer legal ontologies",
    }
    state = AgentState(raw_input={"input.json": json.dumps(payload).encode("utf-8")})
    tools = ToolBox.__new__(ToolBox)
    tools.converter = SimpleNamespace(supported_extensions=())

    out = convert_document(state, tools)

    assert out.docling_doc is not None
    assert out.docling_doc.export_to_markdown().strip() == "Hello world"
    assert out.ontology_selection_user_instruction == "Prefer legal ontologies"


@pytest.mark.anyio
async def test_selector_passes_selection_instruction_into_prompt_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    selected = Ontology(
        iri="https://example.org/ontology/selected",
        graph=RDFGraph._from_turtle_str(
            "@prefix ex: <https://example.org/onto#> . ex:A ex:relatedTo ex:B ."
        ),
    )

    async def _fake_call_llm_with_retry(*, llm_tool, prompt, parser, prompt_kwargs):
        _ = (llm_tool, prompt, parser)
        captured.update(prompt_kwargs)
        return SimpleNamespace(answer_index=1)

    monkeypatch.setattr(
        "ontocast.agent.select_ontology_catalog.call_llm_with_retry",
        _fake_call_llm_with_retry,
    )
    manager = _FakeOntologyManager([selected])
    llm_tool = LLMTool.__new__(LLMTool)

    result = await select_catalog_ontology_for_excerpt(
        ontology_manager=manager,
        llm_tool=llm_tool,
        excerpt="Excerpt text",
        ontology_selection_user_instruction="Prioritize legal/compliance context",
    )

    assert result.iri == selected.iri
    assert (
        captured["ontology_selection_user_instruction"]
        == "Prioritize legal/compliance context"
    )


@pytest.mark.anyio
async def test_context_resolver_forwards_selection_instruction(monkeypatch) -> None:
    captured_instruction: dict[str, str] = {}
    selected = Ontology(
        iri="https://example.org/ontology/selected",
        graph=RDFGraph._from_turtle_str(
            "@prefix ex: <https://example.org/onto#> . ex:A ex:relatedTo ex:B ."
        ),
    )

    async def _select(*_args, **_kwargs) -> Ontology:
        captured_instruction["value"] = _args[3]
        return selected

    monkeypatch.setattr(cr, "select_catalog_ontology_for_excerpt", _select)
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        ontology_selection_user_instruction="Prefer healthcare ontologies",
    )
    tools = ToolBox.__new__(ToolBox)
    tools.ontology_manager = OntologyManager.__new__(OntologyManager)
    tools.llm = LLMTool.__new__(LLMTool)
    unit = ContentUnit(
        text="Alpha is a concept",
        index=0,
        doc_iri=URIRef("https://example.org/doc/1"),
    )

    result = await resolve_unit_ontology_context(state, tools, unit)

    assert result.anchor_iri == selected.iri
    assert captured_instruction["value"] == "Prefer healthcare ontologies"
