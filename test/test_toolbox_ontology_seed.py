"""Tests for ToolBox ontology_directory seed loading."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from ontocast.config import Config, PathConfig, ToolConfig
from ontocast.onto.ontology import Ontology
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager
from ontocast.toolbox import ToolBox


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
