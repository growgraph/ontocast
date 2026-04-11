"""Tests for ontology context mode and proposition-level retrieval."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.config import QdrantConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import (
    OntologyAssemblyMode,
    OntologyContextMode,
    UnitContextStrategy,
)
from ontocast.onto.state import AgentState
from ontocast.stategraph import context_resolver as cr
from ontocast.tool.chunk.util import split_proposition_windows
from ontocast.toolbox import ToolBox


@pytest.mark.anyio
async def test_full_ttl_does_not_invoke_retrieval_paths(monkeypatch) -> None:
    """FULL_TTL must not call ensemble or vote-majority retrieval."""

    async def fail_ensemble(*args, **kwargs):
        raise AssertionError("ensemble path should not run for FULL_TTL")

    async def fail_vote(*args, **kwargs):
        raise AssertionError("vote-majority path should not run for FULL_TTL")

    monkeypatch.setattr(cr, "_resolve_ensemble_context", fail_ensemble)
    monkeypatch.setattr(cr, "_resolve_vote_majority_context", fail_vote)

    state = AgentState(
        ontology_context_mode=OntologyContextMode.FULL_TTL,
        unit_context_strategy=UnitContextStrategy.ENSEMBLE_FIRST,
        content_units=[
            ContentUnit(
                text="Hello world.",
                index=0,
                doc_iri=URIRef("https://example.org/doc/1"),
            )
        ],
    )
    tools = cast(ToolBox, SimpleNamespace())
    unit = state.content_units[0]
    result = await cr.resolve_unit_ontology_context(state, tools, unit)
    assert result.assembly_mode == OntologyAssemblyMode.PRIMARY_WITHOUT_RETRIEVAL


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


def test_qdrant_config_proposition_fields_exist() -> None:
    q = QdrantConfig()
    assert q.proposition_window_sentences >= 1
