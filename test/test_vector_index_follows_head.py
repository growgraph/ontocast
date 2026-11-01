"""The vector index follows the catalog's head after a version is written to the store.

A version can reach the triple store without passing through this ToolBox
(another process writes it). ``initialize`` is what brings the index back in
line: every atom of an ontology must then belong to its one head.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from ontocast.config import Config, EmbeddingConfig, LanceDBConfig, VectorStoreConfig
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager
from ontocast.tool.vector_store.core import VectorStoreManager
from ontocast.tool.vector_store.embedding import FastembedBm25SparseTool
from ontocast.tool.vector_store.lancedb import LanceDBVectorStoreManager
from ontocast.toolbox import ToolBox
from test.ontology_helpers import next_version, stamped_ontology
from test.test_vector_store_pipeline import CountingEmbeddingTool

# A real on-disk LanceDB table.
pytestmark = pytest.mark.integration

VECTOR = OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
ONTO = "https://example.org/onto"
OTHER = "https://example.org/other"


def _ontology(iri: str, *names: str) -> Ontology:
    terms = "\n".join(
        f'<{iri}#{name}> a owl:Class ; rdfs:label "{name} concept" .' for name in names
    )
    return stamped_ontology(
        f"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <{iri}> a owl:Ontology ; rdfs:label "Ontology" .
        {terms}
        """
    )


def _revised(base: Ontology, *names: str) -> Ontology:
    """The child of ``base`` whose terms are exactly ``names``."""
    return next_version(base, _ontology(base.iri, *names).graph)


def _toolbox(tmp_path: Path) -> tuple[ToolBox, InMemoryTripleStoreManager]:
    config = Config()
    config.server.ontology_context_mode = VECTOR
    toolbox = ToolBox(config)
    triples = InMemoryTripleStoreManager()
    toolbox.triple_store_manager = triples
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    toolbox.vector_store = cast(
        VectorStoreManager,
        LanceDBVectorStoreManager(
            store_config=VectorStoreConfig(embedding_batch_size=4, top_k=5),
            lancedb_config=LanceDBConfig(enabled=True, data_dir=tmp_path),
            embedding=embedding,
            sparse_embedding=FastembedBm25SparseTool(config=embedding.config),
        ),
    )
    return toolbox, triples


def _atoms(toolbox: ToolBox) -> list[dict[str, Any]]:
    """Every row of the ontology atom table."""
    store = cast(LanceDBVectorStoreManager, toolbox.vector_store)
    db = store._connect()
    name = store._ontology_table_name()
    if name not in store._list_tables(db):
        return []
    table = db.open_table(name)
    columns = ["ontology_iri", "ontology_hash", "iri"]
    return table.search().select(columns).limit(table.count_rows()).to_list()


def _entities(atoms: list[dict[str, Any]], iri: str) -> set[str]:
    return {a["iri"] for a in atoms if a["ontology_iri"] == iri and "#" in a["iri"]}


def _versions(atoms: list[dict[str, Any]], iri: str) -> set[str]:
    return {a["ontology_hash"] for a in atoms if a["ontology_iri"] == iri}


def _start_job(toolbox: ToolBox) -> None:
    """What a job does before extracting: resync the catalog and the index."""
    asyncio.run(toolbox.initialize(ontology_context_mode=VECTOR))


@pytest.fixture(autouse=True)
def _no_enrichment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ontocast.toolbox.update_ontology_manager", AsyncMock())


def test_a_new_version_replaces_the_old_atoms(tmp_path: Path) -> None:
    toolbox, triples = _toolbox(tmp_path)
    v1 = _ontology(ONTO, "Alpha", "Beta")
    triples.serialize(v1)
    _start_job(toolbox)
    atoms = _atoms(toolbox)
    assert _entities(atoms, ONTO) == {f"{ONTO}#Alpha", f"{ONTO}#Beta"}
    assert _versions(atoms, ONTO) == {v1.version_key}

    # Written by someone else, straight into the store.
    v2 = _revised(v1, "Beta", "Gamma")
    triples.serialize(v2)
    _start_job(toolbox)

    atoms = _atoms(toolbox)
    assert _entities(atoms, ONTO) == {f"{ONTO}#Beta", f"{ONTO}#Gamma"}
    assert _versions(atoms, ONTO) == {v2.version_key}
    # Re-materializing the head wrote no further version.
    heads = [o for o in triples.fetch_ontologies() if o.iri == ONTO]
    assert [h.version_key for h in heads] == [v2.version_key]


def test_a_deleted_term_cannot_be_retrieved(tmp_path: Path) -> None:
    toolbox, triples = _toolbox(tmp_path)
    v1 = _ontology(ONTO, "Alpha", "Beta")
    triples.serialize(v1)
    _start_job(toolbox)
    triples.serialize(_revised(v1, "Beta"))
    _start_job(toolbox)

    store = toolbox.vector_store
    assert store is not None
    hits = store.search_patches("Alpha concept", top_k=20)
    assert hits and f"{ONTO}#Alpha" not in {h.iri for h in hits}


def test_a_diverged_ontology_leaves_the_index(tmp_path: Path) -> None:
    toolbox, triples = _toolbox(tmp_path)
    v1 = _ontology(ONTO, "Alpha")
    triples.serialize(v1)
    triples.serialize(_ontology(OTHER, "Delta"))
    _start_job(toolbox)
    assert {a["ontology_iri"] for a in _atoms(toolbox)} == {ONTO, OTHER}

    # Two children of one version: two heads, which nothing extracts against.
    triples.serialize(_revised(v1, "Alpha", "Beta"))
    triples.serialize(_revised(v1, "Alpha", "Gamma"))
    _start_job(toolbox)

    assert {a["ontology_iri"] for a in _atoms(toolbox)} == {OTHER}


def test_a_catalog_whose_only_ontology_diverged(tmp_path: Path) -> None:
    toolbox, triples = _toolbox(tmp_path)
    v1 = _ontology(ONTO, "Alpha")
    triples.serialize(v1)
    _start_job(toolbox)

    triples.serialize(_revised(v1, "Alpha", "Beta"))
    triples.serialize(_revised(v1, "Alpha", "Gamma"))
    _start_job(toolbox)

    assert _atoms(toolbox) == []
