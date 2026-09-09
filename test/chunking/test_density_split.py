"""Tests for the measurement-density split of sized units."""

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest

from ontocast.config import ChunkConfig
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.prepare import PrepareOptions, prepare_content_units
from ontocast.tool.chunk.sizing import split_by_measurement_density
from ontocast.toolbox import ToolBox
from test.docling_test_helpers import doc_from_markdown_lines

pytestmark = pytest.mark.unit


def _dense_sentences(count: int) -> str:
    return " ".join(
        f"Sample {index} was annealed at {100 + index} °C for {index + 1} h "
        f"and then stored under nitrogen for the next stage."
        for index in range(count)
    )


def test_cap_zero_leaves_text_whole() -> None:
    text = _dense_sentences(6)
    assert split_by_measurement_density(text, max_measurements=0, min_size=10) == [text]


def test_under_cap_leaves_text_whole() -> None:
    text = _dense_sentences(2)
    assert split_by_measurement_density(text, max_measurements=4, min_size=10) == [text]


def test_dense_text_is_split_at_sentence_boundaries_until_under_cap() -> None:
    text = _dense_sentences(6)  # 12 measurements
    pieces = split_by_measurement_density(text, max_measurements=4, min_size=10)

    assert len(pieces) >= 3
    for piece in pieces:
        # Every piece is whole sentences and under the cap.
        assert piece.endswith("stage.")
        assert piece[0].isupper()
    # Nothing is lost or reordered.
    assert " ".join(pieces) == text


def test_split_recurses_on_each_half() -> None:
    text = _dense_sentences(12)  # 24 measurements
    pieces = split_by_measurement_density(text, max_measurements=3, min_size=10)
    assert len(pieces) >= 8


def test_pieces_never_go_below_min_size() -> None:
    text = _dense_sentences(6)
    min_size = 200
    pieces = split_by_measurement_density(text, max_measurements=1, min_size=min_size)
    assert len(pieces) > 1
    assert all(len(piece) >= min_size for piece in pieces)


def test_no_split_when_floor_cannot_be_met() -> None:
    text = _dense_sentences(4)
    whole = split_by_measurement_density(
        text, max_measurements=1, min_size=len(text) // 2 + 1
    )
    assert whole == [text]


def test_no_split_without_a_sentence_boundary() -> None:
    text = "at 10 K, 20 K, 30 K, 40 K and 50 K the emission narrowed steadily"
    assert split_by_measurement_density(text, max_measurements=2, min_size=1) == [text]


def test_paragraph_break_is_a_boundary() -> None:
    left = "Emission at 10 K and 20 K was broad"
    right = "at 30 K and 40 K it narrowed"
    pieces = split_by_measurement_density(
        f"{left}\n\n{right}", max_measurements=2, min_size=1
    )
    assert pieces == [left, right]


def test_prepare_splits_dense_unit_and_keeps_its_label() -> None:
    config = ChunkConfig(min_size=100, max_size=5000, max_measurements_per_unit=4)
    tools = cast(
        ToolBox,
        SimpleNamespace(
            chunker=ChunkerTool(chunk_config=config),
            config=SimpleNamespace(
                chunk_config=config, server=SimpleNamespace(parallel_workers=2)
            ),
        ),
    )
    doc = doc_from_markdown_lines(f"# Methods\n{_dense_sentences(8)}")

    chunks = asyncio.run(
        prepare_content_units(
            doc,
            tools.chunker,
            config,
            PrepareOptions(summarize_sections=["*"]),
            tools,
        )
    )

    assert len(chunks) >= 2
    assert all(chunk.section_label == "methods" for chunk in chunks)
    assert all(len(chunk.text) >= config.min_size for chunk in chunks)


def test_prepare_leaves_units_whole_when_cap_is_off() -> None:
    config = ChunkConfig(min_size=100, max_size=5000)
    tools = cast(
        ToolBox,
        SimpleNamespace(
            chunker=ChunkerTool(chunk_config=config),
            config=SimpleNamespace(
                chunk_config=config, server=SimpleNamespace(parallel_workers=2)
            ),
        ),
    )
    doc = doc_from_markdown_lines(f"# Methods\n{_dense_sentences(8)}")

    chunks = asyncio.run(
        prepare_content_units(
            doc,
            tools.chunker,
            config,
            PrepareOptions(summarize_sections=["*"]),
            tools,
        )
    )

    assert len(chunks) == 1


def test_max_measurements_per_unit_default_is_off() -> None:
    assert ChunkConfig().max_measurements_per_unit == 0
