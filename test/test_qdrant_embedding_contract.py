"""Qdrant collection embedding contract: reject wrong dimension or model at init."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from qdrant_client import QdrantClient

from ontocast.config import EmbeddingConfig, QdrantConfig
from ontocast.tool.vector_store.embedding import FastembedBm25SparseTool
from ontocast.tool.vector_store.qdrant import (
    EmbeddingContractMismatchError,
    QdrantVectorStore,
)
from test.qdrant_util import DeterministicEmbeddingTool, qdrant_reachable


def test_embedding_model_fingerprint_includes_dense_and_bm25() -> None:
    emb = DeterministicEmbeddingTool(
        config=EmbeddingConfig(
            dimension=8,
            model_name="contract-a",
            bm25_model_name="Qdrant/bm25",
        )
    )
    sparse = FastembedBm25SparseTool(config=emb.config)
    store = QdrantVectorStore(
        config=QdrantConfig(),
        embedding=emb,
        sparse_embedding=sparse,
    )
    fp = store._embedding_model_fingerprint()
    assert store._embedding_fingerprint_matches(fp)
    assert fp.startswith("dense:")
    assert "|bm25=Qdrant/bm25" in fp


def _require_live_qdrant() -> QdrantConfig:
    base = QdrantConfig()
    if base.uri is None:
        pytest.skip("QDRANT_URI not configured")
    if not qdrant_reachable(uri=base.uri, api_key=base.api_key):
        pytest.skip(f"Qdrant not reachable at {base.uri}")
    return base


def _delete_if_exist(
    client: QdrantClient,
    names: tuple[str | None, str | None],
) -> None:
    for name in names:
        if name and client.collection_exists(collection_name=name):
            client.delete_collection(collection_name=name)


def test_initialize_rejects_mismatched_embedding_dimension() -> None:
    base = _require_live_qdrant()
    run = uuid.uuid4().hex[:8]
    onto = f"ontocast_contract_dim_{run}_ontologies"
    facts = f"ontocast_contract_dim_{run}_facts"
    qcfg = base.model_copy(
        update={"ontology_collection": onto, "facts_collection": facts}
    )

    emb8 = DeterministicEmbeddingTool(
        config=EmbeddingConfig(dimension=8, model_name="contract-a")
    )
    emb16 = DeterministicEmbeddingTool(
        config=EmbeddingConfig(dimension=16, model_name="contract-a")
    )
    store_a = QdrantVectorStore(config=qcfg, embedding=emb8)
    store_b = QdrantVectorStore(config=qcfg, embedding=emb16)
    client = store_a.client

    try:
        asyncio.run(store_a.initialize())
        with pytest.raises(
            EmbeddingContractMismatchError,
            match=r"vector 'core' size 8 does not match configured dense size 16",
        ):
            asyncio.run(store_b.initialize())
    finally:
        _delete_if_exist(client, (onto, facts))


def test_initialize_rejects_mismatched_embedding_model() -> None:
    base = _require_live_qdrant()
    run = uuid.uuid4().hex[:8]
    onto = f"ontocast_contract_model_{run}_ontologies"
    facts = f"ontocast_contract_model_{run}_facts"
    qcfg = base.model_copy(
        update={"ontology_collection": onto, "facts_collection": facts}
    )

    emb_a = DeterministicEmbeddingTool(
        config=EmbeddingConfig(dimension=8, model_name="contract-a")
    )
    emb_b = DeterministicEmbeddingTool(
        config=EmbeddingConfig(dimension=8, model_name="contract-b")
    )
    store_a = QdrantVectorStore(config=qcfg, embedding=emb_a)
    store_b = QdrantVectorStore(config=qcfg, embedding=emb_b)
    client = store_a.client

    try:
        asyncio.run(store_a.initialize())
        with pytest.raises(ValueError, match="embedding contract mismatch"):
            asyncio.run(store_b.initialize())
    finally:
        _delete_if_exist(client, (onto, facts))
