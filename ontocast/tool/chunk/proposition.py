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
    stride: int | None = None,
    max_chars: int | None = None,
) -> list[str]:
    """Split text into short proposition-like windows for retrieval.

    Args:
        text: Passage to split.
        max_sentences: Sentences per window. Not applied when ``max_chars`` is
            set: the two are alternative ways of saying how much text one query
            carries, and honouring both means the tighter one silently wins.
        max_windows: Ceiling on the windows returned; over it, windows are sampled
            evenly across the passage rather than truncated, so the tail still
            contributes a query.
        stride: Sentences advanced between windows. ``None`` (default) strides by
            the window's own length, giving contiguous, disjoint windows. A smaller
            stride overlaps them, which is what lets a statement spanning a window
            boundary appear whole in some window -- with disjoint windows it appears
            in none, and neither half retrieves what the pair together names.
        max_chars: Characters per window. ``None`` (default) bounds windows by
            sentence count instead, which is the historical behaviour and is
            reproduced exactly.

            A sentence count is a poor bound on how much text a query carries. Two
            sentences of technical prose range over an order of magnitude in
            length, and the encoder truncates on *tokens*, so a sentence-bounded
            window can silently lose its tail. The splitter also has no
            abbreviation handling and breaks on every ``.``, so "J. Phys. Chem.
            Lett." is four "sentences" and a two-sentence window over a citation
            carries nothing retrievable at all.

            A character budget addresses both ends: it caps the long windows that
            truncate, and it *coalesces* short fragments, because it keeps taking
            sentences until the budget is reached. A single sentence longer than
            the budget is emitted whole rather than cut -- the encoder truncates it
            either way, and cutting first only loses more.

    Returns:
        list[str]: Windows in document order.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_sentences <= 0:
        raise ValueError("max_sentences must be >= 1")
    if max_windows <= 0:
        raise ValueError("max_windows must be >= 1")
    if max_chars is not None and max_chars <= 0:
        raise ValueError("max_chars must be >= 1")
    step = max_sentences if stride is None else stride
    if step <= 0:
        raise ValueError("stride must be >= 1")

    # Keep this splitter lightweight and deterministic.
    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n\s*\n+", cleaned)
        if part.strip()
    ]
    if not sentence_parts:
        return [cleaned[:1000]] if cleaned else []

    pieces = (
        _windows_by_sentence_count(sentence_parts, max_sentences, step)
        if max_chars is None
        else _windows_by_char_budget(sentence_parts, max_chars, stride)
    )

    windows: list[str] = []
    seen: set[str] = set()
    for window in pieces:
        # A stride below the window size makes the final windows suffixes of one
        # another once the tail is shorter than a full window; emitting those twice
        # would pay for an identical query and skew the fusion ranks it feeds.
        if window and window not in seen:
            seen.add(window)
            windows.append(window)

    if len(windows) > max_windows:
        # Sample evenly across the text rather than keeping the first ``max_windows``.
        # Truncating dropped the tail of a long chunk entirely, so its closing sections
        # never contributed a retrieval query at all. Positions span both endpoints, so
        # the final window is always represented.
        if max_windows == 1:
            windows = windows[:1]
        else:
            last = len(windows) - 1
            picked = {
                round(position * last / (max_windows - 1))
                for position in range(max_windows)
            }
            windows = [windows[index] for index in sorted(picked)]

    return windows or [cleaned[:1000]]


def _windows_by_sentence_count(
    sentence_parts: list[str], max_sentences: int, step: int
) -> list[str]:
    """Windows of a fixed sentence count, strided by ``step`` sentences."""
    return [
        " ".join(sentence_parts[index : index + max_sentences]).strip()
        for index in range(0, len(sentence_parts), step)
    ]


def _windows_by_char_budget(
    sentence_parts: list[str], max_chars: int, stride: int | None
) -> list[str]:
    """Windows packed to a character budget, never splitting a sentence.

    Each window starts at a sentence and takes as many following sentences as fit,
    so a window's sentence count varies with how long its sentences are -- which is
    the point. A sentence that alone exceeds the budget becomes its own window.

    ``stride`` keeps meaning sentences: ``None`` starts the next window where the
    last one ended (disjoint), and a number starts one every that many sentences,
    which overlaps them. Overlap is a sentence-level idea and stays one.
    """
    windows: list[str] = []
    start = 0
    while start < len(sentence_parts):
        length = 0
        stop = start
        while stop < len(sentence_parts):
            addition = len(sentence_parts[stop]) + (1 if stop > start else 0)
            if stop > start and length + addition > max_chars:
                break
            length += addition
            stop += 1
        windows.append(" ".join(sentence_parts[start:stop]).strip())
        start = stop if stride is None else start + stride
    return windows
