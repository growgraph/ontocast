"""Shared helpers for Qdrant-backed tests (reachability, deterministic embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from ontocast.config import QdrantConfig
from ontocast.tool.vector_store.embedding import EmbeddingTool
from ontocast.util.hash import render_text_hash


@dataclass(frozen=True)
class QdrantSessionTestContext:
    """Session Qdrant collection names and temp workspace for integration tests."""

    qdrant_config: QdrantConfig
    working_directory: Path
    ontology_directory: Path


def qdrant_reachable(*, uri: str, api_key: str | None) -> bool:
    """Return True if Qdrant responds on the collections endpoint."""
    candidates = [api_key] if api_key else [None, "abc123-qwe"]
    for candidate in candidates:
        headers = {"api-key": candidate} if candidate else None
        try:
            response = httpx.get(
                f"{uri.rstrip('/')}/collections",
                headers=headers,
                timeout=2.0,
            )
            if response.status_code == 200:
                return True
        except Exception:
            continue
    return False


class DeterministicEmbeddingTool(EmbeddingTool):
    """Deterministic embedding for tests (no sentence-transformers)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = render_text_hash(text, digits=None)
            seed = int(digest[:16], 16)
            vector = [
                (((seed + i * 97) % 2000) / 1000.0) - 1.0
                for i in range(self.config.dimension)
            ]
            vectors.append(vector)
        return vectors
