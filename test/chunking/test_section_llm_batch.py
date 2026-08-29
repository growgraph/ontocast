"""Batched LLM section classification.

The cost claim being pinned: one LLM call classifies a whole document's
residual, instead of one call per unlabeled chunk. The fallback claim: a
response that cannot be used degrades to the per-chunk path rather than losing
the labels.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import cast

from ontocast.config import ChunkConfig
from ontocast.onto.enum import SectionLabelSource
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.section_llm import llm_backfill_section_labels
from ontocast.tool.chunk.segment import PrepareSegment
from ontocast.toolbox import ToolBox


class _RecordingLLM:
    """Fake LLM that counts calls and replays canned responses."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    async def __call__(self, _prompt):
        self.calls += 1
        payload = (
            self._responses.pop(0) if self._responses else self._responses_fallback()
        )
        return SimpleNamespace(content=payload)

    @staticmethod
    def _responses_fallback() -> str:
        return json.dumps({"label": "results"})


def _tools(llm) -> ToolBox:
    config = ChunkConfig()
    return cast(
        ToolBox,
        SimpleNamespace(
            chunker=ChunkerTool(chunk_config=config),
            config=SimpleNamespace(
                chunk_config=config,
                server=SimpleNamespace(parallel_workers=4),
            ),
            llm=llm,
        ),
    )


def _segments(count: int) -> list[PrepareSegment]:
    return [
        PrepareSegment(
            text=f"Passage number {index} with enough text to classify. " * 4
        )
        for index in range(count)
    ]


def _batch_payload(count: int, label: str = "results") -> str:
    return json.dumps(
        {"assignments": [{"index": index, "label": label} for index in range(count)]}
    )


def test_one_call_classifies_every_segment():
    segments = _segments(6)
    llm = _RecordingLLM([_batch_payload(6)])

    asyncio.run(
        llm_backfill_section_labels(segments, _tools(llm), section_schema_id="academic")
    )

    assert llm.calls == 1
    assert [segment.section_label for segment in segments] == ["results"] * 6
    assert all(
        segment.section_label_source is SectionLabelSource.LLM for segment in segments
    )


def test_batch_size_splits_large_documents():
    segments = _segments(10)
    llm = _RecordingLLM([_batch_payload(10), _batch_payload(10)])

    asyncio.run(
        llm_backfill_section_labels(
            segments, _tools(llm), section_schema_id="academic", batch_size=4
        )
    )

    assert llm.calls == 3  # 4 + 4 + 2


def test_malformed_batch_response_falls_back_to_per_segment():
    segments = _segments(3)
    # First call is the batch attempt and is unusable; the rest are per-segment.
    llm = _RecordingLLM(["not json at all"] + [json.dumps({"label": "methods"})] * 3)

    asyncio.run(
        llm_backfill_section_labels(segments, _tools(llm), section_schema_id="academic")
    )

    assert llm.calls == 4
    assert [segment.section_label for segment in segments] == ["methods"] * 3


def test_batching_can_be_disabled():
    segments = _segments(3)
    llm = _RecordingLLM([json.dumps({"label": "methods"})] * 3)

    asyncio.run(
        llm_backfill_section_labels(
            segments, _tools(llm), section_schema_id="academic", batch_size=0
        )
    )

    assert llm.calls == 3


def test_already_labeled_segments_are_not_sent():
    segments = _segments(3)
    segments[0].section_label = "introduction"
    llm = _RecordingLLM([_batch_payload(3)])

    asyncio.run(
        llm_backfill_section_labels(segments, _tools(llm), section_schema_id="academic")
    )

    assert llm.calls == 1
    assert segments[0].section_label == "introduction"


def test_unknown_labels_are_discarded():
    segments = _segments(2)
    llm = _RecordingLLM(
        [
            json.dumps(
                {
                    "assignments": [
                        {"index": 0, "label": "not_a_real_label"},
                        {"index": 1, "label": "results"},
                    ]
                }
            )
        ]
    )

    asyncio.run(
        llm_backfill_section_labels(segments, _tools(llm), section_schema_id="academic")
    )

    assert segments[0].section_label is None
    assert segments[1].section_label == "results"
