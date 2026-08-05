"""Factory for vector store backend selection.

Backend modules are imported inside the selected branch rather than at module
scope: importing :mod:`ontocast.tool.vector_store.qdrant` pulls the Qdrant SDK
(and, through it, gRPC and an ONNX runtime), and OntoCast's base install ships
neither. Only the backend actually configured is loaded.
"""

from __future__ import annotations

from ontocast.config import ToolConfig
from ontocast.onto.enum import VectorStoreBackend
from ontocast.tool.vector_store.core import VectorStoreManager
from ontocast.tool.vector_store.embedding import (
    EmbeddingTool,
    FastembedBm25SparseTool,
)


def create_vector_store_manager(
    tool_config: ToolConfig,
    embedding: EmbeddingTool,
    sparse_embedding: FastembedBm25SparseTool | None = None,
) -> VectorStoreManager | None:
    """Return a vector store manager for the configured backend.

    Selection is driven by ``VectorStoreConfig.backend``. The default,
    :attr:`~ontocast.onto.enum.VectorStoreBackend.AUTO`, preserves the historical
    behaviour of inferring the backend from whichever connection setting is
    populated, and falls back to the dependency-free in-memory store.

    Args:
        tool_config: The resolved tool configuration.
        embedding: Dense embedding provider.
        sparse_embedding: BM25 sparse provider. Required by the Qdrant and
            LanceDB backends; optional for the in-memory backend, which runs
            dense-only when it is absent.

    Returns:
        A manager for the selected backend, or ``None`` when the backend is
        explicitly disabled.

    Raises:
        ValueError: If an explicitly requested backend is not configured, or if
            Qdrant's ``vector_size`` contradicts the embedding dimension.
    """
    backend = _resolve_backend(tool_config)

    if backend is VectorStoreBackend.NONE:
        return None

    if backend is VectorStoreBackend.QDRANT:
        q_vs = tool_config.qdrant.vector_size
        emb_dim = tool_config.embedding.dimension
        if q_vs is not None and q_vs != emb_dim:
            raise ValueError(
                "QdrantConfig.vector_size must match "
                "EmbeddingConfig.dimension when set "
                f"(got vector_size={q_vs}, embedding.dimension={emb_dim})"
            )
        from ontocast.tool.vector_store.qdrant import QdrantVectorStoreManager

        return QdrantVectorStoreManager(
            store_config=tool_config.vector_store,
            qdrant_config=tool_config.qdrant,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
        )

    if backend is VectorStoreBackend.LANCEDB:
        from ontocast.tool.vector_store.lancedb import LanceDBVectorStoreManager

        return LanceDBVectorStoreManager(
            store_config=tool_config.vector_store,
            lancedb_config=tool_config.lancedb,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
        )

    from ontocast.tool.vector_store.in_memory import InMemoryVectorStoreManager

    return InMemoryVectorStoreManager(
        store_config=tool_config.vector_store,
        embedding=embedding,
        sparse_embedding=sparse_embedding,
    )


def _resolve_backend(tool_config: ToolConfig) -> VectorStoreBackend:
    """Resolve ``AUTO`` against the populated connection settings.

    ``AUTO`` falls back to ``NONE``, not ``MEMORY``. Vector retrieval is one of
    three ontology-context modes and the other two are the defaults, so an
    unconfigured deployment has never had a vector store; silently giving every
    such deployment one would change indexing behaviour and embedding cost
    without anyone asking. The in-memory backend is opt-in via
    ``VECTOR_STORE_BACKEND=memory`` or :meth:`Config.in_memory`.
    """
    backend = tool_config.vector_store.backend
    if backend is not VectorStoreBackend.AUTO:
        if backend is VectorStoreBackend.QDRANT and not tool_config.qdrant.uri:
            raise ValueError(
                "VECTOR_STORE_BACKEND=qdrant requires QDRANT_URI to be set."
            )
        return backend
    if tool_config.qdrant.uri:
        return VectorStoreBackend.QDRANT
    if tool_config.lancedb.enabled:
        return VectorStoreBackend.LANCEDB
    return VectorStoreBackend.NONE
