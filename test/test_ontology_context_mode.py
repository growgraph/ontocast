"""Tests for ontology context mode and proposition-level retrieval."""

from __future__ import annotations

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
from ontocast.onto.retrieval_capabilities import vector_retrieval_available
from ontocast.onto.state import AgentState
from ontocast.stategraph import context_resolver as cr
from ontocast.stategraph.context_resolver import resolve_unit_ontology_context
from ontocast.tool.chunk.proposition import split_proposition_windows
from ontocast.toolbox import ToolBox


@pytest.mark.anyio
async def test_full_ttl_does_not_invoke_ensemble_path(monkeypatch) -> None:
    """Full-TTL path should not run ensemble retrieval."""

    async def fail_ensemble(*args, **kwargs):
        raise AssertionError(
            "ensemble path should not run for selected_single_ontology"
        )

    finance_iri = "https://example.org/finance"
    finance_ontology = Ontology(
        iri=finance_iri,
        graph=RDFGraph._from_turtle_str(
            "@prefix ex: <https://example.org/f#> . ex:F ex:has ex:X ."
        ),
    )

    async def _select(*_a, **_k) -> Ontology:
        return finance_ontology

    monkeypatch.setattr(cr, "_resolve_ensemble_context", fail_ensemble)
    monkeypatch.setattr(cr, "select_catalog_ontology_for_excerpt", _select)
    state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        content_units=[
            ContentUnit(
                text="Hello world.",
                index=0,
                doc_iri=URIRef("https://example.org/doc/1"),
            )
        ],
    )
    vector_store_cfg = VectorStoreConfig()
    tools = cast(
        ToolBox,
        SimpleNamespace(
            ontology_manager=SimpleNamespace(),
            llm=AsyncMock(),
            config=SimpleNamespace(
                tool_config=SimpleNamespace(vector_store=vector_store_cfg)
            ),
        ),
    )
    unit = state.content_units[0]
    result = await resolve_unit_ontology_context(state, tools, unit)
    assert result.assembly_mode == OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM
    assert result.snapshot.source_iris == [finance_iri]


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


def test_split_proposition_windows_strides_instead_of_dropping_the_tail() -> None:
    """Over budget, windows are sampled across the whole text, not truncated."""
    text = " ".join(f"Sentence {index}." for index in range(20))
    windows = split_proposition_windows(text, max_sentences=2, max_windows=3)
    assert len(windows) == 3
    assert windows[0].startswith("Sentence 0.")
    # The closing sentences must still be represented; truncation lost them entirely.
    assert "Sentence 18." in windows[-1]


def test_vector_store_config_proposition_fields_exist() -> None:
    cfg = VectorStoreConfig()
    assert cfg.proposition_window_sentences >= 1


def test_vector_retrieval_available_requires_ready_state() -> None:
    tools = cast(
        ToolBox,
        SimpleNamespace(
            vector_store=object(),
            patch_retriever=object(),
            is_vector_store_ready=lambda: False,
        ),
    )
    assert vector_retrieval_available(tools) is False
