"""Tests for embedded LanceDB vector store backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from ontocast.config import (
    EmbeddingConfig,
    LanceDBConfig,
    QdrantConfig,
    ToolConfig,
    VectorStoreConfig,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.tenancy import (
    DEFAULT_PROJECT,
    DEFAULT_TENANT,
    tenant_project_ontologies_name,
)
from ontocast.tool.vector_store.embedding import FastembedBm25SparseTool
from ontocast.tool.vector_store.factory import create_vector_store_manager
from ontocast.tool.vector_store.lancedb import LanceDBVectorStoreManager
from test.test_vector_store_pipeline import CountingEmbeddingTool

# Every test here creates a real on-disk LanceDB table (and imports pyarrow),
# so the file is an integration surface rather than a unit one.
pytestmark = pytest.mark.integration


def _sample_ontology(iri: str = "https://example.org/smoke") -> Ontology:
    ttl = f"""
    @prefix ex: <{iri}#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ex:Alpha a ex:Concept ;
        rdfs:label "Alpha concept" ;
        rdfs:comment "Used for LanceDB smoke testing." .
    ex:Beta a ex:Concept ;
        rdfs:label "Beta concept" ;
        rdfs:comment "Related to alpha." .
    """
    return Ontology(iri=iri, graph=RDFGraph._from_turtle_str(ttl))


def _build_store(tmp_path: Path) -> LanceDBVectorStoreManager:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    sparse = FastembedBm25SparseTool(config=embedding.config)
    return LanceDBVectorStoreManager(
        store_config=VectorStoreConfig(embedding_batch_size=2, top_k=5),
        lancedb_config=LanceDBConfig(enabled=True, data_dir=tmp_path),
        embedding=embedding,
        sparse_embedding=sparse,
    )


@pytest.mark.anyio
async def test_lancedb_index_search_delete(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    await store.initialize()
    ontology = _sample_ontology()
    indexed = store.index_ontology(ontology)
    assert indexed > 0

    hits = store.search_patches(query="alpha concept", top_k=3)
    assert hits
    assert all(hit.ontology_iri == ontology.iri for hit in hits)

    store.delete_ontology(ontology.iri)
    filtered = store.search_patches(
        query="alpha concept", top_k=3, filter_iri=ontology.iri
    )
    assert filtered == []


@pytest.mark.anyio
async def test_lancedb_tenancy_partition_isolation(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    store.apply_tenancy("tenant_a", "proj_x")
    await store.initialize()
    ontology = _sample_ontology("https://example.org/tenant-a")
    store.index_ontology(ontology)

    store.apply_tenancy("tenant_b", "proj_y")
    await store.initialize()
    hits = store.search_patches(query="alpha concept", top_k=3)
    assert hits == []

    await store.clean_tenancy("tenant_a", "proj_x")
    store.apply_tenancy("tenant_a", "proj_x")
    await store.initialize()
    hits_after_clean = store.search_patches(query="alpha concept", top_k=3)
    assert hits_after_clean == []


def test_tool_config_rejects_dual_vector_backends() -> None:
    with pytest.raises(ValueError, match="only one vector store backend"):
        ToolConfig(
            qdrant=QdrantConfig(uri="http://localhost:6333"),
            lancedb=LanceDBConfig(enabled=True),
        )


def test_factory_selects_lancedb_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QDRANT_URI", raising=False)
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    sparse = FastembedBm25SparseTool(config=embedding.config)
    tool_config = ToolConfig(
        qdrant=QdrantConfig(uri=None),
        lancedb=LanceDBConfig(enabled=True, data_dir=tmp_path),
    )
    manager = create_vector_store_manager(tool_config, embedding, sparse)
    assert isinstance(manager, LanceDBVectorStoreManager)


@pytest.mark.anyio
async def test_lancedb_wipe_and_prune_orphan_iris(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    await store.initialize()
    keep = _sample_ontology("https://example.org/keep")
    orphan = _sample_ontology("https://example.org/orphan")
    store.index_ontology(keep)
    store.index_ontology(orphan)
    assert store.list_indexed_ontology_iris() == {keep.iri, orphan.iri}

    deleted = store.prune_orphan_ontology_iris({keep.iri})
    assert deleted == [orphan.iri]
    assert store.list_indexed_ontology_iris() == {keep.iri}

    await store.wipe_store()
    await store.initialize()
    assert store.list_indexed_ontology_iris() == set()


def test_lancedb_default_tables_use_default_tenant_project(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    expected = tenant_project_ontologies_name(DEFAULT_TENANT, DEFAULT_PROJECT)
    assert store._ontology_table_name() == expected
    assert store._data_dir() == tmp_path.resolve()


def _ontology_without_optional_text(iri: str) -> Ontology:
    """An ontology whose atoms populate none of the optional payload columns."""
    ttl = f"""
    @prefix ex: <{iri}#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ex:Plain a ex:Concept ;
        rdfs:label "Plain concept" .
    """
    return Ontology(iri=iri, graph=RDFGraph._from_turtle_str(ttl))


def _ontology_with_symbols(iri: str) -> Ontology:
    """An ontology whose atoms do populate them (symbols, notation, versions)."""
    ttl = f"""
    @prefix ex: <{iri}#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
    @prefix qudt: <http://qudt.org/schema/qudt/> .
    <{iri}> a owl:Ontology ; owl:versionInfo "1.2.3" .
    ex:Millivolt a ex:Unit ;
        rdfs:label "millivolt" ;
        skos:altLabel "mV" ;
        skos:notation "mV" ;
        qudt:symbol "mV" ;
        rdfs:comment "A unit carrying symbol surfaces." .
    """
    return Ontology(iri=iri, graph=RDFGraph._from_turtle_str(ttl))


@pytest.mark.anyio
async def test_lancedb_indexes_a_second_ontology_that_fills_new_columns(
    tmp_path: Path,
) -> None:
    """Indexing order must not decide which ontologies a catalog can hold.

    The table schema used to be inferred from the first batch written, so a column
    empty throughout the first ontology was typed ``null`` and every later ontology
    that populated it failed to merge. A catalog silently capped at its first
    ontology is indistinguishable from one where retrieval simply found nothing.
    """
    store = _build_store(tmp_path)
    await store.initialize()

    plain = _ontology_without_optional_text("https://example.org/plain")
    symbols = _ontology_with_symbols("https://example.org/symbols")

    assert store.index_ontology(plain) > 0
    # Same table, and this one carries versions, notations and symbol surfaces.
    assert store.index_ontology(symbols) > 0

    # Retrievable, not merely written: a row whose optional columns were dropped
    # on the way in would still count as indexed.
    hits = store.search_patches(query="millivolt", top_k=5)
    assert any(hit.ontology_iri == symbols.iri for hit in hits)
