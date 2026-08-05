"""Tests for the LangChain tool wrappers."""

import json
from typing import cast

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ontocast.config import Config, ToolConfig
from ontocast.config.settings import PathConfig
from ontocast.integrations.langchain import (
    ALL_TOOL_NAMES,
    MUTATING_TOOLS,
    OPT_IN_TOOLS,
    READ_TOOLS,
    _reject_update_query,
    ontocast_tool_diagnostics,
    ontocast_tool_names,
    ontocast_tools,
)
from ontocast.integrations.serialize import (
    graph_to_llm_text,
    models_to_llm_text,
)
from ontocast.onto.enum import VectorStoreBackend
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.llm import LLMTool
from ontocast.tool.vector_store.in_memory import InMemoryVectorStoreManager
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

#: Stands in for a real LLMTool. None of these tests reach the model, and
#: building one would need provider credentials.
STUB_LLM = cast(LLMTool, object())

ONTOLOGY_TTL = """
@prefix ex: <http://example.org/onto#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://example.org/onto> a owl:Ontology .
ex:Widget a owl:Class ; rdfs:label "Widget" .
ex:partOf a owl:ObjectProperty ; rdfs:label "part of" .
"""


@pytest.fixture
def toolbox(tmp_path) -> ToolBox:
    """A light-core ToolBox: in-memory triple store, no vector store."""
    # in_memory() pins the stores so the fixture does not pick up a Fuseki or
    # Qdrant URI from the developer's .env and start asserting against live data.
    config = Config.in_memory(
        tool_config=ToolConfig(path_config=PathConfig(working_directory=tmp_path))
    )
    config.tool_config.vector_store.backend = VectorStoreBackend.NONE
    # Bypass provider setup: none of these tests reach the model, and building
    # a real LLMTool would need provider credentials.
    return ToolBox(config, llm=STUB_LLM)


def test_tool_names_are_stable_and_ordered(toolbox: ToolBox) -> None:
    names = ontocast_tool_names(toolbox)
    assert names == [n for n in ALL_TOOL_NAMES if n in names]


def test_default_set_excludes_mutating_and_opt_in(toolbox: ToolBox) -> None:
    names = set(ontocast_tool_names(toolbox))
    assert not names & set(MUTATING_TOOLS)
    assert not names & set(OPT_IN_TOOLS)


def test_mutating_flag_adds_write_tools(toolbox: ToolBox) -> None:
    without = set(ontocast_tool_names(toolbox))
    with_writes = set(ontocast_tool_names(toolbox, mutating=True))
    added = with_writes - without
    assert added
    assert added <= set(MUTATING_TOOLS)


def test_every_tool_is_a_basetool_with_docs_and_schema(toolbox: ToolBox) -> None:
    for tool in ontocast_tools(toolbox):
        assert isinstance(tool, BaseTool)
        assert tool.description.strip(), f"{tool.name} has no description"
        assert tool.args_schema is not None, f"{tool.name} has no args_schema"


def test_args_schemas_render_as_valid_json_schema(toolbox: ToolBox) -> None:
    """Guards against an RDFGraph-typed field leaking into a tool spec.

    Providers reject a tool whose parameter schema is not plain JSON Schema, so
    a non-primitive field here breaks tool calling at runtime rather than here.
    """
    for tool in ontocast_tools(toolbox, mutating=True, include=ALL_TOOL_NAMES):
        schema = tool.args_schema
        assert isinstance(schema, type) and issubclass(schema, BaseModel)
        rendered = schema.model_json_schema()
        json.dumps(rendered)  # must be serializable
        for field, spec in rendered.get("properties", {}).items():
            assert spec, f"{tool.name}.{field} rendered an empty schema"


def test_include_and_exclude_filter(toolbox: ToolBox) -> None:
    only = ontocast_tools(toolbox, include=["ontocast_list_ontologies"])
    assert [t.name for t in only] == ["ontocast_list_ontologies"]

    dropped = ontocast_tools(toolbox, exclude=["ontocast_list_ontologies"])
    assert "ontocast_list_ontologies" not in {t.name for t in dropped}


def test_unknown_tool_name_is_rejected(toolbox: ToolBox) -> None:
    with pytest.raises(ValueError, match="Unknown tool name"):
        ontocast_tools(toolbox, include=["ontocast_nope"])


def test_sparql_tools_present_for_in_memory_store(toolbox: ToolBox) -> None:
    """The in-memory triple store is a full SPARQL engine, not a degraded one."""
    names = ontocast_tool_names(toolbox)
    assert "ontocast_sparql_select" in names
    assert "ontocast_sparql_construct" in names


def test_vector_tools_gated_off_without_vector_store(tmp_path) -> None:
    """VECTOR_STORE_BACKEND=none must remove the retrieval tools entirely."""
    config = Config.in_memory(
        tool_config=ToolConfig(path_config=PathConfig(working_directory=tmp_path))
    )
    config.tool_config.vector_store.backend = VectorStoreBackend.NONE
    tools = ToolBox(config, llm=STUB_LLM)

    assert tools.vector_store is None
    names = ontocast_tool_names(tools)
    assert "ontocast_search_ontology_terms" not in names
    assert "ontocast_retrieve_ontology_context" not in names


def test_vector_tools_appear_with_in_memory_backend(tmp_path) -> None:
    """The in-memory backend needs no external service, so the tools show up."""
    config = Config.in_memory(
        tool_config=ToolConfig(path_config=PathConfig(working_directory=tmp_path))
    )
    tools = ToolBox(config, llm=STUB_LLM)

    assert isinstance(tools.vector_store, InMemoryVectorStoreManager)
    names = ontocast_tool_names(tools)
    assert "ontocast_search_ontology_terms" in names
    assert "ontocast_retrieve_ontology_context" in names


def test_diagnostics_explain_every_omission(toolbox: ToolBox) -> None:
    available = set(ontocast_tool_names(toolbox, mutating=True))
    reasons = ontocast_tool_diagnostics(toolbox)
    for name in READ_TOOLS + MUTATING_TOOLS:
        if name not in available:
            assert name in reasons, f"{name} omitted with no explanation"
            assert reasons[name].strip()


@pytest.mark.parametrize(
    "query",
    [
        "DELETE WHERE { ?s ?p ?o }",
        "INSERT DATA { <a:b> <a:c> <a:d> }",
        "DROP GRAPH <urn:x>",
        "CLEAR ALL",
    ],
)
def test_update_queries_are_refused(query: str) -> None:
    """The SPARQL tools are read-only; an agent must not be able to wipe a store."""
    with pytest.raises(ValueError, match="read-only"):
        _reject_update_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT ?s WHERE { ?s ?p ?o }",
        "# delete this comment line\nSELECT ?s WHERE { ?s ?p ?o }",
        "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
    ],
)
def test_read_queries_are_allowed(query: str) -> None:
    _reject_update_query(query)


@pytest.mark.anyio
async def test_list_ontologies_returns_json(toolbox: ToolBox) -> None:
    tool = ontocast_tools(toolbox, include=["ontocast_list_ontologies"])[0]
    result = await tool.ainvoke({})
    assert json.loads(result) == []


@pytest.mark.anyio
async def test_apply_graph_update_round_trips(toolbox: ToolBox) -> None:
    tool = ontocast_tools(
        toolbox, include=["ontocast_apply_graph_update"], mutating=True
    )[0]
    result = await tool.ainvoke(
        {
            "insert_ttl": ONTOLOGY_TTL,
            "base_ttl": "",
            "persist": False,
        }
    )
    assert "ex:Widget" in result or "Widget" in result
    summary = json.loads(result.split("# --- result ---")[1])
    assert summary["applied"] is True
    assert summary["persisted"] is False
    assert summary["triples_after"] > summary["triples_before"]


@pytest.mark.anyio
async def test_apply_graph_update_requires_a_patch(toolbox: ToolBox) -> None:
    tool = ontocast_tools(
        toolbox, include=["ontocast_apply_graph_update"], mutating=True
    )[0]
    with pytest.raises(ValueError, match="at least one"):
        await tool.ainvoke({"insert_ttl": "", "delete_ttl": ""})


def test_graph_serialization_marks_truncation() -> None:
    graph = RDFGraph()
    graph.parse(data=ONTOLOGY_TTL, format="turtle")
    text = graph_to_llm_text(graph, max_chars=40)
    assert "TRUNCATED" in text
    assert str(len(graph)) in text


def test_graph_serialization_emits_sources_header() -> None:
    graph = RDFGraph()
    graph.parse(data=ONTOLOGY_TTL, format="turtle")
    text = graph_to_llm_text(
        graph, max_chars=10_000, sources=["http://example.org/onto"]
    )
    assert text.startswith("# sources: http://example.org/onto")


def test_empty_graph_says_so_rather_than_returning_blank() -> None:
    assert "no triples" in graph_to_llm_text(RDFGraph(), max_chars=100)


def test_model_list_truncates_by_item_and_stays_valid_json() -> None:
    class Row(BaseModel):
        name: str

    rows = [Row(name=f"item-{i:03d}") for i in range(50)]
    text = models_to_llm_text(rows, max_chars=200)
    payload, _, note = text.partition("\n// showing")
    assert note
    json.loads(payload)
