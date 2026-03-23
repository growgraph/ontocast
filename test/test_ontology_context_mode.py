"""Tests for ontology context mode and proposition-level retrieval selection."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

from rdflib import URIRef

from ontocast.agent.select_ontology import select_ontology
from ontocast.config import QdrantConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.util import split_proposition_windows
from ontocast.toolbox import ToolBox


class _StubOntologyManager:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int, int, int]] = []
        self.has_ontologies = False

    def get_patch_context_with_sources(
        self,
        query: str,
        top_k: int = 10,
        subgraph_depth: int = 1,
        max_triples: int = 2000,
    ) -> tuple[RDFGraph | None, list[str]]:
        self.queries.append((query, top_k, subgraph_depth, max_triples))
        graph = RDFGraph._from_turtle_str(
            """
            @prefix ex: <https://example.org/o#> .
            ex:Alpha ex:relatedTo ex:Beta .
            """
        )
        return graph, ["https://example.org/o"]

    def get_patch_contexts_with_sources(
        self,
        queries: list[str],
        top_k: int = 10,
        subgraph_depth: int = 1,
        max_triples: int = 2000,
    ) -> list[tuple[RDFGraph | None, list[str]]]:
        results: list[tuple[RDFGraph | None, list[str]]] = []
        for query in queries:
            results.append(
                self.get_patch_context_with_sources(
                    query=query,
                    top_k=top_k,
                    subgraph_depth=subgraph_depth,
                    max_triples=max_triples,
                )
            )
        return results

    async def aget_patch_contexts_with_sources(
        self,
        queries: list[str],
        top_k: int = 10,
        subgraph_depth: int = 1,
        max_triples: int = 2000,
    ) -> list[tuple[RDFGraph | None, list[str]]]:
        return self.get_patch_contexts_with_sources(
            queries=queries,
            top_k=top_k,
            subgraph_depth=subgraph_depth,
            max_triples=max_triples,
        )

    def get_freshest_terminal_ontology_by_iri(self, iri: str | None) -> None:
        del iri
        return None


def _build_tools(
    ontology_manager: _StubOntologyManager,
    proposition_enabled: bool = True,
) -> ToolBox:
    qdrant = QdrantConfig(
        top_k=6,
        induced_subgraph_depth=2,
        induced_subgraph_max_triples=123,
        proposition_window_sentences=2,
        proposition_max_windows=5,
        proposition_retrieval_enabled=proposition_enabled,
    )
    return cast(
        ToolBox,
        SimpleNamespace(
            llm=None,
            ontology_manager=ontology_manager,
            patch_retriever=object(),
            config=SimpleNamespace(tool_config=SimpleNamespace(qdrant=qdrant)),
        ),
    )


def test_select_ontology_uses_proposition_windows_in_retrieval_mode() -> None:
    manager = _StubOntologyManager()
    tools = _build_tools(manager, proposition_enabled=True)
    state = AgentState(
        content_units=[
            ContentUnit(
                text="Alpha is a concept. Beta is another concept. Alpha relates to Beta.",
                index=0,
                doc_iri=URIRef("https://example.org/doc/1"),
            )
        ],
        ontology_context_mode=OntologyContextMode.RETRIEVED_INDUCED_GRAPH,
    )

    result = asyncio.run(select_ontology(state, tools))

    assert result.current_ontology.iri == "https://example.org/o"
    assert result.ontology_patch_sources == ["https://example.org/o"]
    assert len(manager.queries) >= 2
    assert all(depth == 2 for _, _, depth, _ in manager.queries)
    assert all(max_triples == 123 for _, _, _, max_triples in manager.queries)


def test_select_ontology_full_ttl_mode_skips_vector_retrieval() -> None:
    manager = _StubOntologyManager()
    tools = _build_tools(manager, proposition_enabled=True)
    state = AgentState(
        content_units=[
            ContentUnit(
                text="Alpha is a concept. Beta is another concept.",
                index=0,
                doc_iri=URIRef("https://example.org/doc/2"),
            )
        ],
        ontology_context_mode=OntologyContextMode.FULL_TTL,
    )

    result = asyncio.run(select_ontology(state, tools))

    assert result.current_ontology.iri == NULL_ONTOLOGY.iri
    assert manager.queries == []


def test_split_proposition_windows_is_sentence_bounded() -> None:
    windows = split_proposition_windows(
        "One sentence. Two sentence. Three sentence. Four sentence.",
        max_sentences=2,
        max_windows=3,
    )
    assert windows == [
        "One sentence. Two sentence.",
        "Three sentence. Four sentence.",
    ]
