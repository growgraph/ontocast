from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

from rdflib import URIRef

from ontocast.config import QdrantConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import (
    OntologyAssemblyMode,
    OntologyContextMode,
    OntologySelectionPolicy,
    UnitContextStrategy,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.context_resolver import resolve_unit_ontology_context
from ontocast.tool.vector_store.core import GraphAtom, OntologySearchHit
from ontocast.toolbox import ToolBox


class _StubPatchRetriever:
    def __init__(self, graph: RDFGraph, sources: list[str]) -> None:
        self._graph = graph
        self._sources = sources

    async def aretrieve_ensemble(self, **kwargs) -> tuple[RDFGraph, list[str]]:
        _ = kwargs
        return self._graph, self._sources


class _StubVectorStore:
    def __init__(self, hits_by_query: list[list[OntologySearchHit]]) -> None:
        self._hits_by_query = hits_by_query

    async def asearch_patch_hits_many(
        self, queries: list[str], top_k: int | None = None
    ) -> list[list[OntologySearchHit]]:
        _ = queries, top_k
        return self._hits_by_query


class _StubOntologyManager:
    def __init__(self, ontologies: dict[str, Ontology]) -> None:
        self._ontologies = ontologies

    def get_freshest_terminal_ontology_by_iri(self, iri: str | None) -> Ontology | None:
        if iri is None:
            if not self._ontologies:
                return None
            return next(iter(self._ontologies.values()))
        return self._ontologies.get(iri)

    def get_ontology(self, ontology_iri: str | None = None, **kwargs) -> Ontology:
        _ = kwargs
        if ontology_iri is None:
            return Ontology()
        return self._ontologies.get(ontology_iri, Ontology())


def _build_unit() -> ContentUnit:
    return ContentUnit(
        text="Alpha is a concept. Beta is another concept.",
        index=0,
        doc_iri=URIRef("https://example.org/doc/1"),
    )


def _build_tools(
    *,
    patch_retriever: _StubPatchRetriever | None,
    vector_store: _StubVectorStore | None,
    ontology_manager: _StubOntologyManager,
) -> ToolBox:
    qdrant = QdrantConfig(top_k=3, proposition_retrieval_enabled=False)
    return cast(
        ToolBox,
        SimpleNamespace(
            patch_retriever=patch_retriever,
            vector_store=vector_store,
            ontology_manager=ontology_manager,
            config=SimpleNamespace(tool_config=SimpleNamespace(qdrant=qdrant)),
        ),
    )


def _hit(ontology_iri: str) -> OntologySearchHit:
    return OntologySearchHit(
        atom=GraphAtom(
            atom_id=f"atom-{ontology_iri}",
            ontology_iri=ontology_iri,
            iri=f"{ontology_iri}#Entity",
            core_representation="entity",
            neighborhood_representation="context",
        ),
        score=0.9,
    )


def test_resolver_ensemble_first_prefers_stitched_context() -> None:
    graph = RDFGraph._from_turtle_str(
        "@prefix ex: <https://example.org/o#> . ex:A ex:relatedTo ex:B ."
    )
    ontology_iri = "https://example.org/finance"
    tools = _build_tools(
        patch_retriever=_StubPatchRetriever(graph=graph, sources=[ontology_iri]),
        vector_store=None,
        ontology_manager=_StubOntologyManager({}),
    )
    state = AgentState(
        unit_context_strategy=UnitContextStrategy.ENSEMBLE_FIRST,
        ontology_context_mode=OntologyContextMode.RETRIEVED_INDUCED_GRAPH,
    )

    result = asyncio.run(resolve_unit_ontology_context(state, tools, _build_unit()))

    assert result.anchor_iri == ontology_iri
    assert len(result.ontology_snapshot.graph) > 0
    assert result.assembly_mode == OntologyAssemblyMode.ENSEMBLE_STITCHED


def test_resolver_vote_first_selects_majority_ontology() -> None:
    finance_iri = "https://example.org/finance"
    bio_iri = "https://example.org/biomed"
    finance_ontology = Ontology(
        iri=finance_iri,
        graph=RDFGraph._from_turtle_str(
            "@prefix ex: <https://example.org/f#> . ex:F ex:has ex:X ."
        ),
    )
    tools = _build_tools(
        patch_retriever=None,
        vector_store=_StubVectorStore(
            hits_by_query=[
                [_hit(finance_iri), _hit(finance_iri), _hit(bio_iri)],
            ]
        ),
        ontology_manager=_StubOntologyManager({finance_iri: finance_ontology}),
    )
    state = AgentState(
        unit_context_strategy=UnitContextStrategy.VOTE_FIRST,
        ontology_context_mode=OntologyContextMode.RETRIEVED_INDUCED_GRAPH,
    )

    result = asyncio.run(resolve_unit_ontology_context(state, tools, _build_unit()))

    assert result.anchor_iri == finance_iri
    assert result.assembly_mode == OntologyAssemblyMode.VOTE_MAJORITY_ONTOLOGY
    assert result.confidence > 0.5


def test_resolver_full_ttl_skips_retrieval_uses_primary() -> None:
    """FULL_TTL uses per-unit full ontology selection without retrieval."""
    finance_iri = "https://example.org/finance"
    finance_ontology = Ontology(
        iri=finance_iri,
        graph=RDFGraph._from_turtle_str(
            "@prefix ex: <https://example.org/f#> . ex:F ex:has ex:X ."
        ),
    )
    tools = _build_tools(
        patch_retriever=_StubPatchRetriever(graph=RDFGraph(), sources=[finance_iri]),
        vector_store=_StubVectorStore(
            hits_by_query=[[_hit(finance_iri), _hit(finance_iri)]],
        ),
        ontology_manager=_StubOntologyManager({finance_iri: finance_ontology}),
    )
    state = AgentState(
        unit_context_strategy=UnitContextStrategy.VOTE_FIRST,
        ontology_context_mode=OntologyContextMode.FULL_TTL,
    )

    result = asyncio.run(resolve_unit_ontology_context(state, tools, _build_unit()))

    assert result.assembly_mode == OntologyAssemblyMode.PRIMARY_WITHOUT_RETRIEVAL
    assert result.anchor_iri == finance_iri
    assert result.ontology_snapshot.iri == finance_iri


def test_resolver_strict_retrieval_returns_null_when_infra_unavailable() -> None:
    state = AgentState(
        ontology_context_mode=OntologyContextMode.RETRIEVED_INDUCED_GRAPH,
        ontology_selection_policy=OntologySelectionPolicy.STRICT_RETRIEVAL,
    )
    tools = _build_tools(
        patch_retriever=None,
        vector_store=None,
        ontology_manager=_StubOntologyManager({}),
    )

    result = asyncio.run(resolve_unit_ontology_context(state, tools, _build_unit()))

    assert result.assembly_mode == OntologyAssemblyMode.STRICT_RETRIEVAL_UNAVAILABLE
    assert result.ontology_snapshot.is_null()


def test_resolver_llm_selector_only_uses_full_ontology_catalog() -> None:
    finance_iri = "https://example.org/finance"
    finance_ontology = Ontology(
        iri=finance_iri,
        graph=RDFGraph._from_turtle_str(
            "@prefix ex: <https://example.org/f#> . ex:F ex:has ex:X ."
        ),
    )
    state = AgentState(
        ontology_context_mode=OntologyContextMode.RETRIEVED_INDUCED_GRAPH,
        ontology_selection_policy=OntologySelectionPolicy.LLM_SELECTOR_ONLY,
    )
    tools = _build_tools(
        patch_retriever=None,
        vector_store=None,
        ontology_manager=_StubOntologyManager({finance_iri: finance_ontology}),
    )

    result = asyncio.run(resolve_unit_ontology_context(state, tools, _build_unit()))

    assert result.assembly_mode == OntologyAssemblyMode.LLM_SELECTED_FULL_ONTOLOGY
    assert result.anchor_iri == finance_iri
