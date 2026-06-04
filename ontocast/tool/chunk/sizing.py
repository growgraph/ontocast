"""Shared text sizing helpers for OntoCast chunking."""

from __future__ import annotations

from collections.abc import Callable

from ontocast.config import ChunkConfig

DEFAULT_PART_SEPARATOR = "\n\n"


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
