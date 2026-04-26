"""Embedding provider abstraction for vector store workflows."""

from __future__ import annotations

import abc
import importlib
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import Field, PrivateAttr, SecretStr
from qdrant_client.http import models as qdrant_models

from ontocast.config import EmbeddingConfig, EmbeddingProvider
from ontocast.tool.onto import Tool


class EmbeddingTool(Tool):
    """Base embedding tool with provider-specific implementations."""

    config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return vectors for all given texts."""

    def embed_one(self, text: str) -> list[float]:
        """Return a vector for one text."""
        vectors = self.embed([text])
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
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError as error:
            raise ImportError(
                "HuggingFace embeddings require sentence-transformers. "
                "Install it with: uv add sentence-transformers"
            ) from error
        self._embedder = sentence_transformers.SentenceTransformer(
            self.config.model_name
        )
        return self._embedder

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._get_embedder().embed_documents(texts)


class OpenAIEmbeddingTool(_LangChainEmbeddingTool):
    """OpenAI embeddings via langchain-openai."""

    def _build_embedder(self) -> Embeddings:
        api_key = (
            SecretStr(self.config.api_key) if self.config.api_key is not None else None
        )
        return OpenAIEmbeddings(
            model=self.config.model_name,
            openai_api_key=api_key,
            openai_api_base=self.config.base_url,
        )


class OllamaEmbeddingTool(_LangChainEmbeddingTool):
    """Ollama embeddings using either LangChain or direct API fallback."""

    def _build_embedder(self) -> Embeddings:
        return OllamaEmbeddings(
            model=self.config.model_name,
            base_url=self.config.base_url,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return super().embed(texts)
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
        try:
            fastembed_mod = importlib.import_module("fastembed")
        except ImportError as error:
            raise ImportError(
                "BM25 sparse embeddings require fastembed. "
                "Install it with: uv add 'fastembed[all]'"
            ) from error
        sparse_cls = getattr(fastembed_mod, "SparseTextEmbedding", None)
        if sparse_cls is None:
            raise ImportError("fastembed.SparseTextEmbedding is not available")
        self._embedder = sparse_cls(model_name=self.config.bm25_model_name)
        return self._embedder

    def embed_sparse(self, texts: list[str]) -> list[qdrant_models.SparseVector]:
        """Return Qdrant sparse vectors for all given texts."""
        if not texts:
            return []
        model = self._get_embedder()
        out: list[qdrant_models.SparseVector] = []
        for sparse_emb in model.embed(texts):
            payload = sparse_emb.as_object()
            indices_raw = payload["indices"]
            values_raw = payload["values"]
            indices_list = indices_raw.tolist()
            values_list = values_raw.tolist()
            out.append(
                qdrant_models.SparseVector(
                    indices=[int(i) for i in indices_list],
                    values=[float(v) for v in values_list],
                )
            )
        if len(out) != len(texts):
            raise ValueError("BM25 embedder returned mismatched sparse vector count")
        return out

    def embed_one_sparse(self, text: str) -> qdrant_models.SparseVector:
        vectors = self.embed_sparse([text])
        if not vectors:
            raise ValueError("BM25 embedder returned no sparse vector for query text")
        return vectors[0]
