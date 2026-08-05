"""Embedding provider abstraction for vector store workflows."""

from __future__ import annotations

import abc
import threading
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import Field, PrivateAttr, SecretStr

from ontocast.config import EmbeddingConfig, EmbeddingProvider
from ontocast.onto.sparse import SparseVector
from ontocast.tool.onto import Tool
from ontocast.util.optional import require

# Shared across EmbeddingTool instances so parallel ontology reindex does not
# race HuggingFace SentenceTransformer.encode / remote client state.
_EMBED_LOCK = threading.Lock()
_SPARSE_EMBED_LOCK = threading.Lock()


class EmbeddingTool(Tool):
    """Base embedding tool with provider-specific implementations."""

    config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    @abc.abstractmethod
    def _embed_unlocked(self, texts: list[str]) -> list[list[float]]:
        """Return vectors for all given texts (caller holds the embed lock)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return vectors for all given texts as *documents* (thread-safe)."""
        if not texts:
            return []
        with _EMBED_LOCK:
            return self._embed_unlocked(self._apply(self.config.document_prefix, texts))

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Return vectors for all given texts as *queries* (thread-safe).

        Asymmetric retrieval models are trained with distinct query and document
        instructions and lose accuracy when both sides are encoded identically. With
        empty prefixes — the default, suiting a symmetric paraphrase model — this is
        exactly :meth:`embed`.
        """
        if not texts:
            return []
        with _EMBED_LOCK:
            return self._embed_unlocked(self._apply(self.config.query_prefix, texts))

    @staticmethod
    def _apply(prefix: str, texts: list[str]) -> list[str]:
        return texts if not prefix else [f"{prefix}{text}" for text in texts]

    def embed_one(self, text: str) -> list[float]:
        """Return a vector for one query text."""
        vectors = self.embed_query([text])
        if not vectors:
            raise ValueError("Embedding provider returned no vectors for query text")
        return vectors[0]

    @classmethod
    def create(cls, config: EmbeddingConfig) -> "EmbeddingTool":
        """Factory for provider-specific embedding tools."""
        if config.provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbeddingTool(config=config)
        if config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddingTool(config=config)
        if config.provider == EmbeddingProvider.OLLAMA:
            return OllamaEmbeddingTool(config=config)
        raise ValueError(f"Unsupported embedding provider: {config.provider}")


class HuggingFaceEmbeddingTool(EmbeddingTool):
    """Local HuggingFace/SentenceTransformer embeddings."""

    _embedder: Any = PrivateAttr(default=None)

    def _get_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        sentence_transformers = require(
            "sentence_transformers",
            feature=(
                "Local HuggingFace embeddings. For a light install, set "
                "EMBEDDING_PROVIDER=openai or =ollama to embed via an API instead"
            ),
        )
        self._embedder = sentence_transformers.SentenceTransformer(
            self.config.model_name
        )
        return self._embedder

    def _embed_unlocked(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_embedder().encode(
            texts, convert_to_numpy=True, show_progress_bar=len(texts) > 100
        )
        return [vector.tolist() for vector in vectors]


class _LangChainEmbeddingTool(EmbeddingTool):
    """Base adapter for LangChain embedding clients."""

    _embedder: Embeddings | None = PrivateAttr(default=None)

    @abc.abstractmethod
    def _build_embedder(self) -> Embeddings:
        """Construct provider-specific LangChain embedding instance."""

    def _get_embedder(self) -> Embeddings:
        if self._embedder is None:
            self._embedder = self._build_embedder()
        return self._embedder

    def _embed_unlocked(self, texts: list[str]) -> list[list[float]]:
        return self._get_embedder().embed_documents(texts)


class OpenAIEmbeddingTool(_LangChainEmbeddingTool):
    """OpenAI embeddings via langchain-openai."""

    def _build_embedder(self) -> Embeddings:
        api_key = (
            SecretStr(self.config.api_key) if self.config.api_key is not None else None
        )
        OpenAIEmbeddings = require(
            "langchain_openai", feature="OpenAI embeddings"
        ).OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=self.config.model_name,
            api_key=api_key,
            base_url=self.config.base_url,
        )


class OllamaEmbeddingTool(_LangChainEmbeddingTool):
    """Ollama embeddings using either LangChain or direct API fallback."""

    def _build_embedder(self) -> Embeddings:
        OllamaEmbeddings = require(
            "langchain_ollama.embeddings", feature="Ollama embeddings"
        ).OllamaEmbeddings
        return OllamaEmbeddings(
            model=self.config.model_name,
            base_url=self.config.base_url,
        )

    def _embed_unlocked(self, texts: list[str]) -> list[list[float]]:
        try:
            return super()._embed_unlocked(texts)
        except Exception:
            return self._embed_via_http(texts)

    def _embed_via_http(self, texts: list[str]) -> list[list[float]]:
        base_url = self.config.base_url or "http://localhost:11434"
        endpoint = f"{base_url.rstrip('/')}/api/embeddings"
        vectors: list[list[float]] = []
        with httpx.Client(timeout=30.0) as client:
            for text in texts:
                response = client.post(
                    endpoint,
                    json={"model": self.config.model_name, "prompt": text},
                )
                response.raise_for_status()
                payload = response.json()
                vector = payload.get("embedding")
                if not isinstance(vector, list):
                    raise ValueError(
                        "Ollama embedding response missing 'embedding' vector"
                    )
                vectors.append(vector)
        return vectors


class FastembedBm25SparseTool(Tool):
    """BM25-style sparse text embeddings via fastembed (Qdrant-compatible)."""

    config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    _embedder: Any = PrivateAttr(default=None)

    def _get_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        fastembed_mod = require("fastembed", feature="BM25 sparse embeddings")
        sparse_cls = getattr(fastembed_mod, "SparseTextEmbedding", None)
        if sparse_cls is None:
            raise ImportError("fastembed.SparseTextEmbedding is not available")
        self._embedder = sparse_cls(model_name=self.config.bm25_model_name)
        return self._embedder

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Return Qdrant sparse vectors for indexing all given texts (thread-safe)."""
        if not texts:
            return []
        with _SPARSE_EMBED_LOCK:
            return self._embed_sparse_unlocked(texts)

    def embed_sparse_query(self, texts: list[str]) -> list[SparseVector]:
        """Return Qdrant sparse vectors for *querying* with all given texts.

        BM25 is asymmetric: documents carry term-frequency saturation weights, queries
        carry flat per-term weights, and the IDF factor is applied by the store. Encoding
        queries with the document encoder instead squares the term-frequency weighting and
        drops the query/document distinction entirely.
        """
        if not texts:
            return []
        with _SPARSE_EMBED_LOCK:
            return self._embed_sparse_unlocked(texts, query=True)

    def _embed_sparse_unlocked(
        self, texts: list[str], *, query: bool = False
    ) -> list[SparseVector]:
        model = self._get_embedder()
        encode = model.query_embed if query else model.embed
        out: list[SparseVector] = []
        for sparse_emb in encode(texts):
            payload = sparse_emb.as_object()
            indices_raw = payload["indices"]
            values_raw = payload["values"]
            indices_list = indices_raw.tolist()
            values_list = values_raw.tolist()
            out.append(
                SparseVector(
                    indices=[int(i) for i in indices_list],
                    values=[float(v) for v in values_list],
                )
            )
        if len(out) != len(texts):
            raise ValueError("BM25 embedder returned mismatched sparse vector count")
        return out

    def embed_one_sparse(self, text: str) -> SparseVector:
        vectors = self.embed_sparse_query([text])
        if not vectors:
            raise ValueError("BM25 embedder returned no sparse vector for query text")
        return vectors[0]
