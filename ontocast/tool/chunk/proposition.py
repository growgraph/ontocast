"""Lightweight text splitting helpers (no ML dependencies)."""

from __future__ import annotations

import re

# Regex pattern for splitting text into sentences
# Matches: paragraph breaks (double newlines) OR sentence endings followed by capital letters
SENTENCE_SPLIT_REGEX = r"(?:\n\s*\n+)|(?<=[.!?])\s+(?=[A-Z][a-z])"


def split_proposition_windows(
    text: str,
    max_sentences: int = 2,
    max_windows: int = 16,
) -> list[str]:
    """Split text into short proposition-like windows for retrieval."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_sentences <= 0:
        raise ValueError("max_sentences must be >= 1")
    if max_windows <= 0:
        raise ValueError("max_windows must be >= 1")

    # Keep this splitter lightweight and deterministic.
    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n\s*\n+", cleaned)
        if part.strip()
    ]
    if not sentence_parts:
        return [cleaned[:1000]] if cleaned else []

    windows: list[str] = []
    step = max_sentences
    for index in range(0, len(sentence_parts), step):
        window = " ".join(sentence_parts[index : index + max_sentences]).strip()
        if window:
            windows.append(window)
        if len(windows) >= max_windows:
            break
    return windows or [cleaned[:1000]]
