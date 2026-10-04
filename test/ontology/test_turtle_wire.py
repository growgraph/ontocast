"""The ontology loop on the default wire: Turtle, compact layout.

Its chapter follows the wire (``inherit``), so a unit state built with no
format reads and writes Turtle in both the update render and the critic.
"""

import importlib
import json
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.config import WebSearchConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import LLMGraphFormat, LLMOutputLayout
from ontocast.onto.model import GraphUpdateRenderReport, OntologyCritiqueReport
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.llm import LLMTool
from test.snapshot_helpers import snapshot_from_ontology

pytestmark = pytest.mark.unit

render_ontology_module = importlib.import_module("ontocast.agent.render_ontology")
criticise_ontology_module = importlib.import_module("ontocast.agent.criticise_ontology")


class _FakeLLMProvider:
    async def get_llm_tool(self, budget_tracker: object) -> LLMTool:
        return cast(LLMTool, object())


def _tools() -> AtomicToolBox:
    return AtomicToolBox(
        llm_provider=_FakeLLMProvider(),
        web_search_config=WebSearchConfig(enabled=False),
    )


def _state() -> UnitOntologyState:
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix onto: <https://example.com/onto#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <https://example.com/onto> a owl:Ontology .
        onto:Company a owl:Class ; rdfs:label "Company" .
        """,
        format="turtle",
    )
    ontology = Ontology(iri="https://example.com/onto", graph=graph)
    return UnitOntologyState(
        content_unit=ContentUnit(
            text="Alice works for ACME.",
            index=0,
            doc_iri=URIRef("https://example.com/doc/d1"),
        ),
        ontology_snapshot=snapshot_from_ontology(ontology),
    )


def _schema(format_instructions: str) -> dict:
    body = format_instructions.split("```\n", 1)[1].split("\n```", 1)[0]
    return json.loads(body)


def test_a_default_unit_state_is_on_the_turtle_compact_wire() -> None:
    state = _state()
    assert state.llm_graph_format is LLMGraphFormat.TURTLE
    assert state.llm_output_layout is LLMOutputLayout.COMPACT


@pytest.mark.anyio
async def test_update_render_reads_and_writes_turtle(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_call(**kwargs):
        captured.update(kwargs["prompt_kwargs"])
        return GraphUpdateRenderReport()

    monkeypatch.setattr(render_ontology_module, "call_llm_with_retry", fake_call)
    await render_ontology_module.render_ontology_update(_state(), tools=_tools())

    combined = "\n".join(str(value) for value in captured.values())
    assert "single Turtle string" in combined
    assert "```ttl" in combined
    assert "Inside a Turtle string" in combined
    format_instructions = next(
        str(value) for value in captured.values() if "JSON schema" in str(value)
    )
    properties = _schema(format_instructions)["properties"]
    assert properties["insert_graph"]["type"] == "string"
    assert properties["delete_graph"]["type"] == "string"


@pytest.mark.anyio
async def test_critic_reads_an_inline_id_turtle_chapter(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_call(**kwargs):
        captured.update(kwargs["prompt_kwargs"])
        return OntologyCritiqueReport(
            success=True,
            score=95,
            systemic_critique_summary="Looks good.",
            actionable_ontology_fixes=[],
        )

    monkeypatch.setattr(criticise_ontology_module, "call_llm_with_retry", fake_call)
    await criticise_ontology_module.criticise_ontology(_state(), tools=_tools())

    combined = "\n".join(str(value) for value in captured.values())
    assert "```ttl" in combined
    assert "TRIPLE INDEX (id |" not in combined
    assert "LLM_GRAPH_FORMAT=turtle" in combined
