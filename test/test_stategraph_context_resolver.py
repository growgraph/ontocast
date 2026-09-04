from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from rdflib import URIRef

from ontocast.config import VectorStoreConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import (
    OntologyAssemblyMode,
    OntologyContextMode,
    RetrievalMetric,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import (
    EmptyOntologyContextError,
    OntologyContextConfigError,
)
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
    kwargs.setdefault("has_ontologies", True)
    return SimpleNamespace(preferred_namespace_prefixes={}, **kwargs)


def _build_tools(
    *,
    patch_retriever: _StubPatchRetriever | None,
    vector_store: object | None,
    ontology_manager: object,
    llm: object | None = None,
    ontology_context_required: bool = False,
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
            config=SimpleNamespace(
                tool_config=SimpleNamespace(vector_store=vcfg),
                server=SimpleNamespace(
                    ontology_context_required=ontology_context_required
                ),
            ),
        ),
    )


class _StubVectorStore:
    """Vector index that still lists ontologies the triple store has lost."""

    def __init__(self, indexed: list[str] | None = None) -> None:
        self._indexed = indexed if indexed is not None else ["https://example.org/o"]

    def list_indexed_ontology_iris(self) -> set[str]:
        return set(self._indexed)


def _empty_context_tools(metrics: dict, **kwargs) -> ToolBox:
    retriever = _StubPatchRetriever(graph=RDFGraph(), sources=[])
    retriever.last_retrieval_metrics = metrics
    return _build_tools(
        patch_retriever=retriever,
        vector_store=_StubVectorStore(),
        ontology_manager=_stub_ontology_manager(),
        **kwargs,
    )


def _resolve_empty(metrics: dict, **kwargs):
    tools = _empty_context_tools(metrics, **kwargs)
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
    )
    context = UnitLoopContext.from_agent_state(state)
    return context, asyncio.run(
        resolve_unit_ontology_context(context, tools, _build_unit())
    )


def test_empty_ontology_context_stops_the_run_when_required() -> None:
    """A finished run with no vocabulary is worse than no run.

    The renderer is told to extract "based on provided domain ontology"; handed
    nothing it falls back on generic vocabulary, and the SHACL gate then finds
    no node its shapes target, so the run reports a vacuous pass. This shipped
    once and was invisible at every checkpoint.

    This is the *facts* reading of an empty context, which is the only one it
    has: a facts unit cannot answer one.
    """
    with pytest.raises(EmptyOntologyContextError):
        _resolve_empty({"atoms_final": 99}, ontology_context_required=True)


def test_a_caller_that_can_create_vocabulary_is_handed_the_empty_context() -> None:
    """The ontology loop answers an empty context by inventing vocabulary.

    ``render_ontology`` branches on an empty seed into
    ``render_ontology_fresh``, which mints a catalog ontology from the text.
    Raising ahead of it made that branch unreachable, so a corpus with no
    ontology yet -- the starting point of every ontology-building run, and the
    documented first run -- read as a deployment fault. It also stopped a
    populated-catalog run whenever the selector honestly reported that no
    catalog ontology fits the unit.
    """
    tools = _empty_context_tools({"atoms_final": 99}, ontology_context_required=True)
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
    )
    context = UnitLoopContext.from_agent_state(state)

    result = asyncio.run(
        resolve_unit_ontology_context(
            context, tools, _build_unit(), can_create_vocabulary=True
        )
    )

    assert len(result.snapshot.graph) == 0


def test_empty_ontology_context_is_survivable_when_opted_out() -> None:
    context, result = _resolve_empty({"atoms_final": 99})

    assert len(result.snapshot.graph) == 0
    assert RetrievalMetric.EMPTY_SNAPSHOT_REASON in context.retrieval_metrics


def test_an_empty_catalog_is_not_reported_as_a_threshold_problem() -> None:
    """The diagnostic used to blame retrieval for a catalog fault.

    Atoms were selected and scored fine; the triple store simply listed no
    ontology to expand them against. Reporting that as "scored below the
    thresholds" sends an operator to tune numbers that were never involved.
    """
    context, _ = _resolve_empty(
        {
            "atoms_final": 99,
            "atoms_after_dedupe": 238,
            "catalog_context_triples": 0,
            "catalog_graph_cache_hits": 0,
            "catalog_graph_cache_misses": 0,
        }
    )
    reason = context.retrieval_metrics[RetrievalMetric.EMPTY_SNAPSHOT_REASON]

    assert "catalog" in reason
    assert "threshold" not in reason


def test_a_genuine_threshold_miss_still_reads_as_one() -> None:
    """Signalled by the pre-gate counts, not by ``atoms_after_dedupe``.

    That one is measured *after* the score gate, so a threshold rejection
    records zero of them -- the reading this test asserts was unreachable from
    a real run, which is why a threshold miss used to report as "no candidate
    atoms matched".
    """
    context, _ = _resolve_empty(
        {
            "candidate_hits": 238,
            "threshold_rejected": 238,
            "atoms_after_dedupe": 0,
            "catalog_context_triples": 1400,
            "catalog_graph_cache_hits": 15,
        }
    )
    reason = context.retrieval_metrics[RetrievalMetric.EMPTY_SNAPSHOT_REASON]

    assert "threshold" in reason


def test_an_empty_catalog_is_found_even_when_retrieval_never_asked_it() -> None:
    """The short-circuit on zero atoms leaves the catalog keys absent.

    ``metrics.get("catalog_context_triples") == 0`` is then ``None == 0``, so
    both catalog branches were skipped on exactly the run where the catalog was
    the cause and the empty index -- the symptom -- was reported instead.
    """
    retriever = _StubPatchRetriever(graph=RDFGraph(), sources=[])
    retriever.last_retrieval_metrics = {
        "query_count": 12,
        "atoms_final": 0,
        "atoms_after_dedupe": 0,
    }
    tools = _build_tools(
        patch_retriever=retriever,
        vector_store=_StubVectorStore(indexed=[]),
        ontology_manager=_stub_ontology_manager(has_ontologies=False),
    )
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
    )
    context = UnitLoopContext.from_agent_state(state)
    asyncio.run(resolve_unit_ontology_context(context, tools, _build_unit()))
    reason = str(context.retrieval_metrics[RetrievalMetric.EMPTY_SNAPSHOT_REASON])

    assert "catalog" in reason


def test_a_unit_with_no_text_may_have_an_empty_context() -> None:
    """It has nothing to extract either way, so this is not a catalog fault."""
    tools = _empty_context_tools({"atoms_final": 0}, ontology_context_required=True)
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
    )
    blank = ContentUnit(text="   ", index=0, doc_iri=URIRef("https://example.org/d/1"))

    result = asyncio.run(
        resolve_unit_ontology_context(
            UnitLoopContext.from_agent_state(state), tools, blank
        )
    )

    assert len(result.snapshot.graph) == 0


def test_the_empty_context_guard_covers_the_single_ontology_modes_too() -> None:
    """Every mode can return an empty context, so the check is mode-agnostic.

    It lives on the dispatcher rather than in the vector branch for the same
    reason the snapshot size is recorded there: the modes that bound nothing
    were also the modes that reported nothing.
    """
    manager = _stub_ontology_manager(
        aget_ontologies=AsyncMock(return_value=[]),
        aget_ontology_by_iri=AsyncMock(return_value=None),
    )
    tools = _build_tools(
        patch_retriever=None,
        vector_store=None,
        ontology_manager=manager,
        ontology_context_required=True,
    )
    state = AgentState(
        ontology_context_mode=OntologyContextMode.FIXED_SINGLE_ONTOLOGY,
        current_ontology_iri="https://example.org/missing",
    )

    with pytest.raises(EmptyOntologyContextError):
        asyncio.run(
            resolve_unit_ontology_context(
                UnitLoopContext.from_agent_state(state), tools, _build_unit()
            )
        )

    # ... and the phase exemption is equally mode-agnostic.
    assert (
        len(
            asyncio.run(
                resolve_unit_ontology_context(
                    UnitLoopContext.from_agent_state(state),
                    tools,
                    _build_unit(),
                    can_create_vocabulary=True,
                )
            ).snapshot.graph
        )
        == 0
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
