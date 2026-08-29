"""Tests for pre-tag segment coalescing."""

from ontocast.config.section_labels import load_section_label_schema
from ontocast.tool.chunk.segment import (
    PrepareSegment,
    coalesce_small_segments_right,
    is_abstract_exempt,
    merge_into_right,
)


def _academic_schema():
    return load_section_label_schema("academic")


def test_merge_into_right_prefers_right_label() -> None:
    left = PrepareSegment(text="tiny", section_label=None)
    right = PrepareSegment(text="bigger body text here", section_label="results")
    merged = merge_into_right(left, right)
    assert merged.section_label == "results"
    assert "tiny" in merged.text
    assert "bigger" in merged.text


def test_coalesce_chain_of_smalls_into_right() -> None:
    schema = _academic_schema()
    segments = [
        PrepareSegment(text="a"),
        PrepareSegment(text="b"),
        PrepareSegment(text="enough characters in this segment to stand alone"),
    ]
    result = coalesce_small_segments_right(segments, min_chars=10, schema=schema)
    assert len(result) == 1
    assert "a" in result[0].text
    assert "b" in result[0].text


def test_coalesce_trailing_small_merges_left() -> None:
    schema = _academic_schema()
    segments = [
        PrepareSegment(text="enough characters in the leading segment"),
        PrepareSegment(text="x"),
    ]
    result = coalesce_small_segments_right(segments, min_chars=10, schema=schema)
    assert len(result) == 1
    assert result[0].text.endswith("x") or "x" in result[0].text


def test_abstract_exempt_not_coalesced() -> None:
    schema = _academic_schema()
    segments = [
        PrepareSegment(text="# Abstract\nShort body."),
        PrepareSegment(
            text=(
                "# Introduction\n"
                "enough characters in the introduction section here for sizing"
            )
        ),
    ]
    assert is_abstract_exempt(segments[0], schema)
    result = coalesce_small_segments_right(segments, min_chars=80, schema=schema)
    assert len(result) == 2
    assert "Short body" in result[0].text


def test_does_not_merge_across_section_heading() -> None:
    schema = _academic_schema()
    segments = [
        PrepareSegment(text="tiny orphan"),
        PrepareSegment(text="# Methods\nWe used a benchmark dataset for evaluation."),
    ]
    result = coalesce_small_segments_right(segments, min_chars=20, schema=schema)
    assert len(result) == 2


def test_small_tail_before_section_merges_into_left() -> None:
    schema = _academic_schema()
    segments = [
        PrepareSegment(
            text="enough characters in the introduction section for sizing here"
        ),
        PrepareSegment(text="tiny tail"),
        PrepareSegment(text="# Methods\nWe used a benchmark dataset for evaluation."),
    ]
    result = coalesce_small_segments_right(segments, min_chars=20, schema=schema)
    assert len(result) == 2
    assert "tiny tail" in result[0].text
    assert result[1].text.startswith("# Methods")
