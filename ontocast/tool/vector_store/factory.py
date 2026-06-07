"""Factory for vector store backend selection."""

from __future__ import annotations

from ontocast.config import ToolConfig
from ontocast.tool.vector_store.core import VectorStoreManager
from ontocast.tool.vector_store.embedding import (
    EmbeddingTool,
    FastembedBm25SparseTool,
)
from ontocast.tool.vector_store.lancedb import LanceDBVectorStoreManager
from ontocast.tool.vector_store.qdrant import QdrantVectorStoreManager


def create_vector_store_manager(
    tool_config: ToolConfig,
    embedding: EmbeddingTool,
    sparse_embedding: FastembedBm25SparseTool,
) -> VectorStoreManager | None:
    """Return a vector store manager when a backend URI is configured."""
    if tool_config.qdrant.uri:
        q_vs = tool_config.qdrant.vector_size
        emb_dim = tool_config.embedding.dimension
        if q_vs is not None and q_vs != emb_dim:
            raise ValueError(
                "QdrantConfig.vector_size must match "
                "EmbeddingConfig.dimension when set "
                f"(got vector_size={q_vs}, embedding.dimension={emb_dim})"
            )
        return QdrantVectorStoreManager(
            store_config=tool_config.vector_store,
            qdrant_config=tool_config.qdrant,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
        )
    if tool_config.lancedb.enabled:
        return LanceDBVectorStoreManager(
            store_config=tool_config.vector_store,
            lancedb_config=tool_config.lancedb,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
        )
    return None
