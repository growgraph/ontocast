from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from rdflib import URIRef

from ontocast.config import VectorStoreConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode, OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import OntologyContextConfigError
from ontocast.onto.state import AgentState
from ontocast.stategraph import context_resolver as cr
from ontocast.stategraph.context_resolver import (
    build_merged_document_ontology_context,
    resolve_unit_ontology_context,
)
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


class _StubPatchRetriever:
    def __init__(self, graph: RDFGraph, sources: list[str]) -> None:
        self._graph = graph
        self._sources = sources
        self.last_retrieval_metrics: dict = {}

    async def aretrieve_ensemble(self, **kwargs) -> tuple[RDFGraph, list[str]]:
        _ = kwargs
        return self._graph, self._sources


def _build_unit() -> ContentUnit:
    return ContentUnit(
        text="Alpha is a concept. Beta is another concept.",
        index=0,
        doc_iri=URIRef("https://example.org/doc/1"),
    )


def _stub_ontology_manager(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(preferred_namespace_prefixes={}, **kwargs)


def _build_tools(
    *,
    patch_retriever: _StubPatchRetriever | None,
    vector_store: object | None,
    ontology_manager: object,
    llm: object | None = None,
) -> ToolBox:
    vcfg = VectorStoreConfig(top_k=3, proposition_retrieval_enabled=False)
    return cast(
        ToolBox,
        SimpleNamespace(
            patch_retriever=patch_retriever,
            vector_store=vector_store,
            is_vector_store_ready=lambda: (
                patch_retriever is not None and vector_store is not None
            ),
            vector_store_last_error=None,
            ontology_manager=ontology_manager,
            llm=llm,
            config=SimpleNamespace(tool_config=SimpleNamespace(vector_store=vcfg)),
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
        ontology_manager=_stub_ontology_manager(),
    )
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
    )

    result = asyncio.run(
        resolve_unit_ontology_context(
            UnitLoopContext.from_agent_state(state), tools, _build_unit()
        )
    )

    assert result.primary_writable_iri == ontology_iri
    assert len(result.snapshot.graph) > 0
    assert result.assembly_mode == OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE


def test_resolver_vector_retrieval_raises_when_vector_stack_missing() -> None:
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
    )
    tools = _build_tools(
        patch_retriever=None,
        vector_store=None,
        ontology_manager=_stub_ontology_manager(),
    )
    with pytest.raises(OntologyContextConfigError):
        asyncio.run(
            resolve_unit_ontology_context(
                UnitLoopContext.from_agent_state(state), tools, _build_unit()
            )
        )


def test_resolver_selected_single_ontology_uses_mocked_llm_selection(
    monkeypatch,
) -> None:
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
        ontology_manager=_stub_ontology_manager(),
        llm=AsyncMock(),
    )
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
    )
    result = asyncio.run(
        resolve_unit_ontology_context(
            UnitLoopContext.from_agent_state(state), tools, _build_unit()
        )
    )
    assert result.assembly_mode == OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM
    assert result.primary_writable_iri == finance_iri
    assert result.snapshot.source_iris == [finance_iri]


def test_build_merged_document_ontology_context_merges_sorted_artifacts() -> None:
    state = AgentState()
    first = Ontology(
        iri="https://example.org/onto/b",
        graph=RDFGraph._from_turtle_str(
            """
            @prefix exb: <https://example.org/onto/b#> .
            exb:ClassB exb:label exb:ValueB .
            """
        ),
    )
    second = Ontology(
        iri="https://example.org/onto/a",
        graph=RDFGraph._from_turtle_str(
            """
            @prefix exa: <https://example.org/onto/a#> .
            exa:ClassA exa:label exa:ValueA .
            """
        ),
    )
    state.reduced_ontology_artifacts = [first, second]

    context = build_merged_document_ontology_context(
        UnitLoopContext.from_agent_state(state)
    )

    assert context is not None
    assert context.patch_sources == [
        "https://example.org/onto/a",
        "https://example.org/onto/b",
    ]
    assert context.primary_writable_iri == "https://example.org/onto/a"
    assert len(context.snapshot.graph) >= 2
    assert context.assembly_mode == OntologyAssemblyMode.DOCUMENT_MERGED_REDUCED


def test_merged_document_context_is_independent_of_any_unit() -> None:
    """The merged facts context is a pure function of document-level state.

    This is what licenses building it once for the whole fan-out: it takes no
    unit argument, so no unit can influence it.
    """
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
    )
    merged = Ontology(
        iri="https://example.org/onto/merged",
        graph=RDFGraph._from_turtle_str(
            """
            @prefix ex: <https://example.org/onto/merged#> .
            ex:Class ex:label ex:Value .
            """
        ),
    )
    state.reduced_ontology_artifacts = [merged]

    first = cr.build_merged_document_ontology_context(
        UnitLoopContext.from_agent_state(state)
    )
    second = cr.build_merged_document_ontology_context(
        UnitLoopContext.from_agent_state(state)
    )

    assert first is not None and second is not None
    assert first.primary_writable_iri == merged.iri
    assert first.patch_sources == [merged.iri]
    assert first.assembly_mode == OntologyAssemblyMode.DOCUMENT_MERGED_REDUCED
    assert set(first.snapshot.graph) == set(second.snapshot.graph)
    assert first.patch_sources == second.patch_sources


def test_resolver_fixed_single_ontology_resolves_from_manager() -> None:
    finance_iri = "https://example.org/finance"
    finance_ontology = Ontology(
        ontology_id="finance",
        iri=finance_iri,
        graph=RDFGraph._from_turtle_str(
            "@prefix ex: <https://example.org/f#> . ex:F ex:has ex:X ."
        ),
    )

    class _StubOntologyManager:
        preferred_namespace_prefixes: dict[str, str] = {}

        def get_freshest_terminal_ontology(
            self, ontology_id: str | None = None
        ) -> Ontology | None:
            if ontology_id == "finance":
                return finance_ontology
            return None

    tools = _build_tools(
        patch_retriever=None,
        vector_store=None,
        ontology_manager=_StubOntologyManager(),
    )
    state = AgentState(
        ontology_context_mode=OntologyContextMode.FIXED_SINGLE_ONTOLOGY,
        ontology_context_fixed_ontology_id="finance",
    )
    result = asyncio.run(
        resolve_unit_ontology_context(
            UnitLoopContext.from_agent_state(state), tools, _build_unit()
        )
    )
    assert result.assembly_mode == OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY
    assert result.primary_writable_iri == finance_iri
