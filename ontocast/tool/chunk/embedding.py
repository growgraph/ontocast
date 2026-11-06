"""Embedding helpers shared by the heading tiers: the embedder type and cosine."""

from __future__ import annotations

import math
from typing import Protocol


class TextEmbedder(Protocol):
    """Embeds short texts, or returns ``None`` when no model is available."""

    def __call__(self, texts: list[str]) -> list[list[float]] | None: ...


def unit_vector(vector: list[float]) -> list[float]:
    """``vector`` scaled to length one (a zero vector is returned unchanged)."""
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two unit vectors."""
    return sum(a * b for a, b in zip(left, right))


__all__ = ["TextEmbedder", "cosine", "unit_vector"]
