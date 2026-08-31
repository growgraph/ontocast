"""Tests for ToolBox ontology_directory seed loading."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest

from ontocast.config import Config, PathConfig, ToolConfig
from ontocast.onto.enum import OntologyContextMode, RenderMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager
from ontocast.tool.vector_store.core import VectorStoreManager
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


def _write_seed_ttl(directory: Path) -> Ontology:
    ttl = """
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    <https://example.org/seed> a owl:Ontology ;
        rdfs:label "Seed Ontology" .
    """
    path = directory / "seed.ttl"
    path.write_text(ttl, encoding="utf-8")
    return Ontology.from_file(path)


def _write_seed_ttl_with_terms(directory: Path) -> Ontology:
    """A seed that defines a class, not only its own ``owl:Ontology`` header."""
    ttl = """
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    <https://example.org/seed> a owl:Ontology ;
        rdfs:label "Seed Ontology" .

    <https://example.org/seed#Thing> a owl:Class ;
        rdfs:label "Thing" .
    """
    path = directory / "seed.ttl"
    path.write_text(ttl, encoding="utf-8")
    return Ontology.from_file(path)


def test_load_seed_ontologies_from_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        od = Path(tmp) / "ontologies"
        od.mkdir()
        expected = _write_seed_ttl(od)
        tool_config = ToolConfig(path_config=PathConfig(ontology_directory=od))
        toolbox = ToolBox(Config(tool_config=tool_config))
        seeds = toolbox._load_seed_ontologies_from_directory()
        assert len(seeds) == 1
        assert seeds[0].iri == expected.iri


def test_synchronize_ontologies_materializes_missing_seed(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        od = Path(tmp) / "ontologies"
        od.mkdir()
        _write_seed_ttl(od)
        manager = InMemoryTripleStoreManager()
        tool_config = ToolConfig(path_config=PathConfig(ontology_directory=od))
        toolbox = ToolBox(Config(tool_config=tool_config))
        toolbox.triple_store_manager = manager
        materialize = AsyncMock()
        monkeypatch.setattr(toolbox, "_materialize_ontology", materialize)

        synced = asyncio.run(toolbox._synchronize_ontologies())
        assert len(synced) == 1
        assert synced[0].iri == "https://example.org/seed"
        assert manager.fetch_ontologies() == []


def test_delete_ontology_leaves_the_seed_directory_intact() -> None:
    """Deleting an ontology must not unlink the seed TTL that declares it.

    ``ontology_directory`` is a read-only fixture the next startup reloads from.
    Deletion used to glob it and unlink any file matching the IRI, so a
    store-level delete silently destroyed curated input with no way back.
    """
    with tempfile.TemporaryDirectory() as tmp:
        od = Path(tmp) / "ontologies"
        od.mkdir()
        seed = _write_seed_ttl(od)
        seed_path = od / "seed.ttl"

        tool_config = ToolConfig(path_config=PathConfig(ontology_directory=od))
        toolbox = ToolBox(Config(tool_config=tool_config))
        toolbox.triple_store_manager = InMemoryTripleStoreManager()
        toolbox.vector_store = None

        asyncio.run(toolbox.delete_ontology_by_iri(seed.iri))

        assert seed_path.exists()
        assert toolbox._load_seed_ontologies_from_directory()[0].iri == seed.iri


def test_ingest_ontology_ttl_without_a_seed_directory() -> None:
    """Ingestion no longer requires ``ontology_directory``.

    It never wrote there, so demanding it only blocked registering an ontology
    in the triple store on deployments that seed nothing from disk.
    """
    toolbox = ToolBox(Config(tool_config=ToolConfig(path_config=PathConfig())))
    toolbox.triple_store_manager = InMemoryTripleStoreManager()
    toolbox.vector_store = None

    ttl = b"""
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    <https://example.org/ingested> a owl:Ontology ;
        rdfs:label "Ingested Ontology" .
    """
    ontology = asyncio.run(toolbox.ingest_ontology_ttl(ttl))
    assert ontology.iri == "https://example.org/ingested"


def test_a_seed_repairs_an_iri_the_store_serves_no_graph_for() -> None:
    """Listing an ontology is not serving it.

    A partition can name an ontology and return nothing for it -- a dataset
    restored empty, a graph dropped from under a still-registered header. The
    seed rule keyed on the IRI alone, so the one copy that could have repaired
    it was skipped and the catalog stayed empty. With a wiped vector index there
    is then nothing left to extract against at all.
    """
    with tempfile.TemporaryDirectory() as tmp:
        od = Path(tmp) / "ontologies"
        od.mkdir()
        seed = _write_seed_ttl(od)
        empty_shell = Ontology(
            graph=RDFGraph(), iri=seed.iri, ontology_id="seed", version="0.0.1"
        )

        toolbox = ToolBox(
            Config(
                tool_config=ToolConfig(path_config=PathConfig(ontology_directory=od))
            )
        )
        repairs = toolbox._seed_ontologies_missing_from([empty_shell], [seed])

        assert [o.iri for o in repairs] == [seed.iri]


def test_a_served_ontology_is_never_overwritten_by_an_older_seed() -> None:
    """Ontology-mode runs write evolved terminals back to the store.

    Re-materializing the on-disk seed over one of those would revert the
    catalog to its bootstrap state without saying so.
    """
    with tempfile.TemporaryDirectory() as tmp:
        od = Path(tmp) / "ontologies"
        od.mkdir()
        seed = _write_seed_ttl_with_terms(od)

        toolbox = ToolBox(
            Config(
                tool_config=ToolConfig(path_config=PathConfig(ontology_directory=od))
            )
        )
        repairs = toolbox._seed_ontologies_missing_from([seed], [seed])

        assert repairs == []


def test_synchronize_prefers_the_seed_over_an_empty_store_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        od = Path(tmp) / "ontologies"
        od.mkdir()
        seed = _write_seed_ttl_with_terms(od)
        empty_shell = Ontology(
            graph=RDFGraph(), iri=seed.iri, ontology_id="seed", version="0.0.1"
        )

        class _Store:
            async def afetch_ontologies(self) -> list[Ontology]:
                return [empty_shell]

        toolbox = ToolBox(
            Config(
                tool_config=ToolConfig(path_config=PathConfig(ontology_directory=od))
            )
        )
        toolbox.triple_store_manager = cast(InMemoryTripleStoreManager, _Store())

        synced = asyncio.run(toolbox._synchronize_ontologies())

        assert [o.iri for o in synced] == [seed.iri]
        assert ToolBox._defines_terms(synced[0])
        assert [o.iri for o in toolbox._last_seed_repairs] == [seed.iri]


def _empty_catalog_toolbox(tmp: str, render_mode: RenderMode) -> ToolBox:
    empty_dir = Path(tmp) / "ontologies"
    empty_dir.mkdir()
    config = Config(
        tool_config=ToolConfig(path_config=PathConfig(ontology_directory=empty_dir))
    )
    config.server.render_mode = render_mode
    config.server.ontology_context_required = True
    toolbox = ToolBox(config)
    toolbox.triple_store_manager = InMemoryTripleStoreManager()
    toolbox.vector_store = None
    return toolbox


def test_a_facts_run_refuses_to_start_with_no_catalog_at_all() -> None:
    """End to end through ``initialize``, which is where the flag is wired.

    The unit checks below it can pass while the call site still does not set
    ``require_populated_catalog``, which is exactly the wiring that let a run
    start and fail one unit at a time.
    """
    from ontocast.onto.retrieval_capabilities import EmptyOntologyContextError

    with tempfile.TemporaryDirectory() as tmp:
        toolbox = _empty_catalog_toolbox(tmp, RenderMode.FACTS)

        with pytest.raises(EmptyOntologyContextError):
            asyncio.run(toolbox.initialize(require_populated_catalog=True))


def test_an_ontology_run_starts_with_no_catalog_at_all() -> None:
    """A corpus with no ontology is what this render mode is for.

    Refusing here is what made the documented first run -- ``process`` against
    a document with no seed directory -- fail at startup.
    """
    with tempfile.TemporaryDirectory() as tmp:
        toolbox = _empty_catalog_toolbox(tmp, RenderMode.ONTOLOGY_AND_FACTS)

        asyncio.run(toolbox.initialize(require_populated_catalog=True))


def test_a_server_may_start_with_no_catalog_and_be_filled_over_http() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        empty_dir = Path(tmp) / "ontologies"
        empty_dir.mkdir()
        toolbox = ToolBox(
            Config(
                tool_config=ToolConfig(
                    path_config=PathConfig(ontology_directory=empty_dir)
                )
            )
        )
        toolbox.triple_store_manager = InMemoryTripleStoreManager()
        toolbox.vector_store = None

        asyncio.run(toolbox.initialize())

        ontology = asyncio.run(
            toolbox.ingest_ontology_ttl(
                b"""
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            <https://example.org/late> a owl:Ontology .
            """
            )
        )
        assert ontology.iri == "https://example.org/late"


def test_a_wipe_is_followed_by_a_reindex_from_the_seed_directory(monkeypatch) -> None:
    """The property no test held: the refill actually happens.

    The wipe is unconditional and the refill is not, so the two have to be
    asserted together -- a passing "wipe_store was awaited" says nothing about
    whether anything was put back, which is the state the failure leaves behind.
    """
    monkeypatch.setattr("ontocast.toolbox.update_ontology_manager", AsyncMock())

    class _RecordingVectorStore:
        def __init__(self) -> None:
            self.indexed: dict[str, int] = {}
            self.wiped = False

        async def wipe_store(self) -> None:
            self.wiped = True
            self.indexed.clear()

        async def initialize(self) -> None:
            return None

        def reindex_ontology(self, ontology: Ontology) -> int:
            self.indexed[ontology.iri] = len(ontology.graph)
            return len(ontology.graph)

        def list_indexed_ontology_iris(self) -> set[str]:
            return set(self.indexed)

        def prune_orphan_ontology_iris(self, keep_iris: set[str]) -> list[str]:
            return []

    with tempfile.TemporaryDirectory() as tmp:
        od = Path(tmp) / "ontologies"
        od.mkdir()
        seed = _write_seed_ttl_with_terms(od)

        config = Config(
            tool_config=ToolConfig(path_config=PathConfig(ontology_directory=od))
        )
        config.server.ontology_context_mode = (
            OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
        )
        toolbox = ToolBox(config)
        toolbox.triple_store_manager = InMemoryTripleStoreManager()
        store = _RecordingVectorStore()
        toolbox.vector_store = cast(VectorStoreManager, store)

        asyncio.run(
            toolbox.initialize(
                ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY,
                wipe_vector_store=True,
                require_populated_catalog=True,
            )
        )

        assert store.wiped
        assert set(store.indexed) == {seed.iri}
