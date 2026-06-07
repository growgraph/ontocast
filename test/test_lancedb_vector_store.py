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


def test_lancedb_default_tables_use_default_tenant_project(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    expected = tenant_project_ontologies_name(DEFAULT_TENANT, DEFAULT_PROJECT)
    assert store._ontology_table_name() == expected
    assert store._data_dir() == tmp_path.resolve()
