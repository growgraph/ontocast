"""Tests for ToolBox ontology synchronization helpers."""

import asyncio
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ontocast.config import (
    Config,
    EmbeddingConfig,
    PathConfig,
    QdrantConfig,
    ToolConfig,
)
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.toolbox import ToolBox


def test_materialize_ontology_calls_vector_reindex(test_ontology):
    tb = MagicMock()
    tb.triple_store_manager = None
    tb.filesystem_manager = None
    reindexed: list = []

    def reindex(o):
        reindexed.append(o)

    tb.vector_store = MagicMock()
    tb.vector_store.reindex_ontology = reindex

    async def main():
        await ToolBox._materialize_ontology(tb, test_ontology)

    asyncio.run(main())
    assert reindexed == [test_ontology]


def test_materialize_ontology_skips_vector_reindex_when_store_not_ready(test_ontology):
    tb = MagicMock()
    tb.triple_store_manager = None
    tb.filesystem_manager = None
    tb.vector_store = MagicMock()
    tb.vector_store.reindex_ontology = MagicMock()
    tb.is_vector_store_ready = MagicMock(return_value=False)

    async def main():
        await ToolBox._materialize_ontology(tb, test_ontology)

    asyncio.run(main())
    tb.vector_store.reindex_ontology.assert_not_called()


def test_materialize_ontology_serializes_remote_triple_store(test_ontology):
    remote = MagicMock()
    remote.aserialize = AsyncMock(return_value=True)
    fs = MagicMock()
    tb = MagicMock()
    tb.triple_store_manager = remote
    tb.filesystem_manager = fs
    tb.vector_store = None

    async def main():
        await ToolBox._materialize_ontology(tb, test_ontology)

    asyncio.run(main())
    remote.aserialize.assert_awaited_once_with(test_ontology)


def test_initialize_materializes_then_adds_with_skip_vector(monkeypatch, test_ontology):
    monkeypatch.setattr(
        "ontocast.toolbox.update_ontology_manager",
        AsyncMock(),
    )

    materialized: list = []
    added: list = []

    async def fake_sync(self):
        return [test_ontology]

    async def fake_mat(self, o):
        materialized.append(o)

    def fake_add(ontology, *, skip_vector_index: bool = False):
        added.append((ontology, skip_vector_index))

    class Stub:
        vector_store = None
        triple_store_manager = None
        filesystem_manager = None
        llm = MagicMock()
        ontology_manager: MagicMock

        def __init__(self) -> None:
            self.ontology_manager = MagicMock()
            self.vector_store_ready = False
            self.vector_store_last_error = None

        def should_initialize_vector_store(self, ontology_context_mode):
            return ToolBox.should_initialize_vector_store(
                cast(ToolBox, self), ontology_context_mode
            )

        _synchronize_ontologies = fake_sync
        _materialize_ontology = fake_mat

    st = Stub()
    st.ontology_manager.add_ontology = MagicMock(side_effect=fake_add)

    async def main():
        await ToolBox.initialize(cast(ToolBox, st))

    asyncio.run(main())

    assert materialized == [test_ontology]
    assert added == [(test_ontology, True)]


def test_toolbox_rejects_mismatched_qdrant_vector_size_and_embedding_dim() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        od = wd / "ontologies"
        od.mkdir()
        tool_config = ToolConfig(
            path_config=PathConfig(working_directory=wd, ontology_directory=od),
            embedding=EmbeddingConfig(dimension=384),
            qdrant=QdrantConfig(uri="http://localhost:6333", vector_size=8),
        )
        with pytest.raises(ValueError, match="vector_size must match"):
            ToolBox(Config(tool_config=tool_config))


def test_toolbox_always_wires_bm25_when_vector_search_enabled() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        od = wd / "ontologies"
        od.mkdir()
        tool_config = ToolConfig(
            path_config=PathConfig(working_directory=wd, ontology_directory=od),
            embedding=EmbeddingConfig(dimension=384),
            qdrant=QdrantConfig(uri="http://localhost:6333"),
        )
        toolbox = ToolBox(Config(tool_config=tool_config))
        assert toolbox.vector_store is not None
        assert toolbox.vector_store.sparse_embedding is not None


def test_initialize_skips_vector_store_in_full_ttl_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "ontocast.toolbox.update_ontology_manager",
        AsyncMock(),
    )
    synchronized: list = []

    class Stub:
        triple_store_manager = None
        filesystem_manager = None
        llm = MagicMock()
        ontology_manager: MagicMock

        def __init__(self) -> None:
            self.vector_store = MagicMock()
            self.vector_store.initialize = AsyncMock()
            self.vector_store_ready = False
            self.vector_store_last_error = None
            self.ontology_manager = MagicMock()

        async def _synchronize_ontologies(self):
            return synchronized

        async def _materialize_ontology(self, _):
            return None

        def should_initialize_vector_store(self, ontology_context_mode):
            return ToolBox.should_initialize_vector_store(
                cast(ToolBox, self), ontology_context_mode
            )

    st = Stub()
    asyncio.run(
        ToolBox.initialize(
            cast(ToolBox, st),
            ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
            fail_on_vector_store_error=False,
        )
    )
    st.vector_store.initialize.assert_not_awaited()


def test_initialize_vector_store_failure_is_non_fatal_when_configured(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "ontocast.toolbox.update_ontology_manager",
        AsyncMock(),
    )

    class Stub:
        triple_store_manager = None
        filesystem_manager = None
        llm = MagicMock()
        ontology_manager: MagicMock

        def __init__(self) -> None:
            self.vector_store = MagicMock()
            self.vector_store.initialize = AsyncMock(
                side_effect=RuntimeError("qdrant unavailable")
            )
            self.vector_store_ready = False
            self.vector_store_last_error = None
            self.ontology_manager = MagicMock()

        async def _synchronize_ontologies(self):
            return []

        async def _materialize_ontology(self, _):
            return None

        def should_initialize_vector_store(self, ontology_context_mode):
            return ToolBox.should_initialize_vector_store(
                cast(ToolBox, self), ontology_context_mode
            )

    st = Stub()
    asyncio.run(
        ToolBox.initialize(
            cast(ToolBox, st),
            ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY,
            fail_on_vector_store_error=False,
        )
    )
    assert st.vector_store_ready is False
    assert st.vector_store_last_error is not None


def test_ingest_ontology_ttl_rejects_identity_conflict_before_persisting() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        od = wd / "ontologies"
        od.mkdir()
        tool_config = ToolConfig(
            path_config=PathConfig(working_directory=wd, ontology_directory=od)
        )
        config = Config(tool_config=tool_config)
        ontology_manager = OntologyManager()

        existing = Ontology(
            graph=RDFGraph._from_turtle_str(
                """
                @prefix owl: <http://www.w3.org/2002/07/owl#> .
                <https://example.org/finance> a owl:Ontology .
                """
            ),
            iri="https://example.org/finance",
            ontology_id="finance",
        )
        ontology_manager.add_ontology(existing)

        class Stub:
            def __init__(self) -> None:
                self.config = config
                self.ontology_manager = ontology_manager
                self._materialize_ontology = AsyncMock()

        incoming_ttl = b"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://example.com/finance> a owl:Ontology .
        """
        stub = Stub()

        with pytest.raises(ValueError, match="already bound to IRI"):
            asyncio.run(ToolBox.ingest_ontology_ttl(cast(ToolBox, stub), incoming_ttl))

        stub._materialize_ontology.assert_not_awaited()
        assert list(od.glob("*.ttl")) == []
