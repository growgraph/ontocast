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
