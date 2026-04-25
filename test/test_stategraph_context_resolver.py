from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from rdflib import URIRef

from ontocast.config import QdrantConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode, OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import OntologyContextConfigError
from ontocast.onto.state import AgentState
from ontocast.stategraph import context_resolver as cr
from ontocast.stategraph.context_resolver import resolve_unit_ontology_context
from ontocast.toolbox import ToolBox


class _StubPatchRetriever:
    def __init__(self, graph: RDFGraph, sources: list[str]) -> None:
        self._graph = graph
        self._sources = sources

    async def aretrieve_ensemble(self, **kwargs) -> tuple[RDFGraph, list[str]]:
        _ = kwargs
        return self._graph, self._sources


def _build_unit() -> ContentUnit:
    return ContentUnit(
        text="Alpha is a concept. Beta is another concept.",
        index=0,
        doc_iri=URIRef("https://example.org/doc/1"),
    )


def _build_tools(
    *,
    patch_retriever: _StubPatchRetriever | None,
    vector_store: object | None,
    ontology_manager: object,
    llm: object | None = None,
) -> ToolBox:
    qdrant = QdrantConfig(top_k=3, proposition_retrieval_enabled=False)
    return cast(
        ToolBox,
        SimpleNamespace(
            patch_retriever=patch_retriever,
            vector_store=vector_store,
            ontology_manager=ontology_manager,
            llm=llm,
            config=SimpleNamespace(tool_config=SimpleNamespace(qdrant=qdrant)),
        ),
    )


def test_resolver_vector_retrieval_prefers_ensemble() -> None:
    graph = RDFGraph._from_turtle_str(
        "@prefix ex: <https://example.org/o#> . ex:A ex:relatedTo ex:B ."
    )
    ontology_iri = "https://example.org/finance"
    tools = _build_tools(
        patch_retriever=_StubPatchRetriever(graph=graph, sources=[ontology_iri]),
        vector_store=object(),
        ontology_manager=SimpleNamespace(),
    )
    state = AgentState(ontology_context_mode=OntologyContextMode.VECTOR_RETRIEVAL)

    result = asyncio.run(resolve_unit_ontology_context(state, tools, _build_unit()))

    assert result.anchor_iri == ontology_iri
    assert len(result.ontology_snapshot.graph) > 0
    assert result.assembly_mode == OntologyAssemblyMode.ENSEMBLE_STITCHED


def test_resolver_vector_retrieval_raises_when_vector_stack_missing() -> None:
    state = AgentState(ontology_context_mode=OntologyContextMode.VECTOR_RETRIEVAL)
    tools = _build_tools(
        patch_retriever=None,
        vector_store=None,
        ontology_manager=SimpleNamespace(),
    )
    with pytest.raises(OntologyContextConfigError):
        asyncio.run(resolve_unit_ontology_context(state, tools, _build_unit()))


def test_resolver_full_ttl_uses_mocked_llm_selection(monkeypatch) -> None:
    finance_iri = "https://example.org/finance"
    finance_ontology = Ontology(
        iri=finance_iri,
        graph=RDFGraph._from_turtle_str(
            "@prefix ex: <https://example.org/f#> . ex:F ex:has ex:X ."
        ),
    )

    async def _select(*_a, **_k) -> Ontology:
        return finance_ontology

    monkeypatch.setattr(
        cr,
        "select_catalog_ontology_for_excerpt",
        _select,
    )
    tools = _build_tools(
        patch_retriever=None,
        vector_store=None,
        ontology_manager=SimpleNamespace(),
        llm=AsyncMock(),
    )
    state = AgentState(ontology_context_mode=OntologyContextMode.FULL_TTL)
    result = asyncio.run(resolve_unit_ontology_context(state, tools, _build_unit()))
    assert result.assembly_mode == OntologyAssemblyMode.LLM_SELECTED_UNIT_ONTOLOGY
    assert result.anchor_iri == finance_iri
    assert result.ontology_snapshot.iri == finance_iri
