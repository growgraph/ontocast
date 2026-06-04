"""Tests for shared chunk sizing helpers."""

import pytest

from ontocast.config import ChunkConfig
from ontocast.tool.chunk.sizing import (
    hard_cap_parts,
    merge_small_parts,
    size_bounded_text,
    size_text_parts,
)


def test_hard_cap_parts_splits_at_word_boundary() -> None:
    parts = ["word " * 100]
    capped = hard_cap_parts(parts, max_size=50)
    assert len(capped) > 1
    assert all(len(part) <= 50 for part in capped)


def test_merge_small_parts_respects_bounds() -> None:
    parts = ["short", "also short", "tiny"]
    merged = merge_small_parts(parts, min_size=20, max_size=100)
    assert len(merged) == 1
    assert len(merged[0]) >= 20


def test_merge_small_parts_does_not_exceed_max_size() -> None:
    parts = ["a" * 40, "b" * 40, "c" * 40]
    merged = merge_small_parts(parts, min_size=50, max_size=90)
    assert all(len(part) <= 90 for part in merged)


def test_size_text_parts_hard_caps_before_merge() -> None:
    parts = ["word " * 200]
    sized = size_text_parts(parts, min_size=50, max_size=80)
    assert len(sized) >= 2
    assert all(len(part) <= 80 for part in sized)


def test_size_bounded_text_uses_split_fn() -> None:
    config = ChunkConfig(min_size=10, max_size=30)

    def split_fn(text: str) -> list[str]:
        return [text[i : i + 15] for i in range(0, len(text), 15)]

    sized = size_bounded_text("abcdefghijklmnop", config, split_fn)
    assert sized
    assert all(len(part) <= 30 for part in sized)


def test_merge_small_parts_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="min_size must be <= max_size"):
        merge_small_parts(["a"], min_size=100, max_size=10)
