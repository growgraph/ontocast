"""Shared text sizing helpers for OntoCast chunking."""

from __future__ import annotations

import re
from collections.abc import Callable

from ontocast.config import ChunkConfig
from ontocast.tool.chunk.proposition import SENTENCE_SPLIT_REGEX
from ontocast.util.measurement_lexicon import unit_adjacent_numbers

DEFAULT_PART_SEPARATOR = "\n\n"

_SENTENCE_BOUNDARY_RE = re.compile(SENTENCE_SPLIT_REGEX)


def hard_cap_parts(parts: list[str], max_size: int) -> list[str]:
    """Split parts that still exceed ``max_size`` at word or character boundaries."""
    if max_size <= 0:
        raise ValueError("max_size must be >= 1")

    capped: list[str] = []
    for part in parts:
        if len(part) <= max_size:
            capped.append(part)
            continue

        start = 0
        while start < len(part):
            end = min(start + max_size, len(part))
            if end < len(part):
                space = part.rfind(" ", start, end)
                if space > start:
                    end = space
            piece = part[start:end].strip()
            if piece:
                capped.append(piece)
            if end <= start:
                end = min(start + max_size, len(part))
            start = end

    return capped


def merge_small_parts(
    parts: list[str],
    min_size: int,
    max_size: int,
    *,
    separator: str = DEFAULT_PART_SEPARATOR,
) -> list[str]:
    """Greedy merge of undersized parts without exceeding ``max_size``."""
    if not parts:
        return []
    if min_size > max_size:
        raise ValueError("min_size must be <= max_size")

    merged: list[str] = []
    accumulator = ""

    def flush() -> None:
        nonlocal accumulator
        if accumulator:
            merged.append(accumulator)
        accumulator = ""

    for part in parts:
        if not accumulator:
            accumulator = part
            continue

        combined = (
            accumulator + part if not separator else f"{accumulator}{separator}{part}"
        )
        if len(accumulator) < min_size and len(combined) <= max_size:
            accumulator = combined
        else:
            flush()
            accumulator = part

    flush()

    if len(merged) <= 1:
        return merged

    coalesced: list[str] = []
    for part in merged:
        if coalesced and (len(part) < min_size or len(coalesced[-1]) < min_size):
            combined = (
                coalesced[-1] + part
                if not separator
                else f"{coalesced[-1]}{separator}{part}"
            )
            if len(combined) <= max_size:
                coalesced[-1] = combined
                continue
        coalesced.append(part)
    return coalesced


def size_text_parts(
    parts: list[str],
    min_size: int,
    max_size: int,
    *,
    separator: str = DEFAULT_PART_SEPARATOR,
) -> list[str]:
    """Hard-cap oversized parts, then merge to respect ``min_size`` / ``max_size``."""
    if not parts:
        return []
    return merge_small_parts(
        hard_cap_parts(parts, max_size),
        min_size,
        max_size,
        separator=separator,
    )


def size_bounded_text(
    text: str,
    config: ChunkConfig,
    split_fn: Callable[[str], list[str]],
    *,
    separator: str = DEFAULT_PART_SEPARATOR,
) -> list[str]:
    """Split ``text`` when needed, then enforce OntoCast chunk size bounds."""
    text = text.strip()
    if not text:
        return []

    if len(text) > config.max_size:
        parts = [part.strip() for part in split_fn(text) if part.strip()]
        if not parts:
            parts = [text]
    else:
        parts = [text]

    return size_text_parts(
        parts,
        config.min_size,
        config.max_size,
        separator=separator,
    )


def _midpoint_sentence_cut(text: str, min_size: int) -> tuple[int, int] | None:
    """The sentence boundary nearest the midpoint leaving both halves ``>= min_size``.

    Returns:
        ``(start, end)`` of the boundary whitespace, so the left half is
        ``text[:start]`` and the right half ``text[end:]``; ``None`` when no
        boundary satisfies the floor.
    """
    midpoint = len(text) / 2
    best: tuple[int, int] | None = None
    best_distance = float("inf")
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        start, end = match.span()
        if start < min_size or len(text) - end < min_size:
            continue
        distance = abs(start - midpoint)
        if distance < best_distance:
            best, best_distance = (start, end), distance
    return best


def split_by_measurement_density(
    text: str,
    *,
    max_measurements: int,
    min_size: int,
) -> list[str]:
    """Split ``text`` while it states more measurements than ``max_measurements``.

    Extraction loss tracks how many measurements a unit packs, not how long
    it is, so the cut is by density rather than by size: the text is cut at
    the sentence or paragraph boundary nearest its midpoint and each half is
    re-checked, recursively. No piece is ever shorter than ``min_size``; a
    dense unit that cannot be cut without producing one is returned whole.

    Args:
        text: Unit text.
        max_measurements: Cap on unit-adjacent numbers per piece; ``<= 0``
            disables splitting.
        min_size: Floor on piece length in characters.

    Returns:
        The stripped text as one piece, or its pieces in text order.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if max_measurements <= 0:
        return [stripped]
    if len(unit_adjacent_numbers(stripped)) <= max_measurements:
        return [stripped]
    cut = _midpoint_sentence_cut(stripped, min_size)
    if cut is None:
        return [stripped]
    start, end = cut
    return split_by_measurement_density(
        stripped[:start], max_measurements=max_measurements, min_size=min_size
    ) + split_by_measurement_density(
        stripped[end:], max_measurements=max_measurements, min_size=min_size
    )
