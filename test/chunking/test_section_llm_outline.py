"""LLM section tier: one call over the outline, weak votes, failure costs only votes.

The model is shown each headed section once, by its own heading and opening
text, and each part of unheaded text on its own. Its label reaches every
segment of a region as weak evidence: it decides where no other tier spoke and
is recorded where one did.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import cast

import pytest

from ontocast.config import ChunkConfig
from ontocast.config.section_labels import load_section_label_schema
from ontocast.onto.enum import LabelStrength, SectionLabelSource
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.prepare import (
    PreparedDocument,
    PrepareOptions,
    prepare_content_units,
)
from ontocast.tool.chunk.section_llm import (
    llm_label_outline,
    outline_regions,
    text_excerpt,
)
from ontocast.tool.chunk.segment import PrepareSegment, SectionRegion
from ontocast.toolbox import ToolBox
from test.docling_test_helpers import doc_from_markdown_lines

pytestmark = pytest.mark.unit

ACADEMIC = load_section_label_schema("academic")
S = SectionLabelSource

_FINDING = (
    "Films kept 92 % of their initial efficiency after 500 h in humid air, "
    "which the encapsulation explains."
)
_SPIN = (
    "Films were spin coated at 4000 rpm for 30 s from a 1.2 M precursor "
    "solution and annealed at 100 °C for 10 min on a hot plate."
)


class _RecordingLLM:
    """Fake LLM that records prompts and replays canned responses in order."""

    def __init__(self, responses: list[str | Exception]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def __call__(self, prompt):
        self.prompts.append(prompt.to_string())
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response)


def _labels(*labels: str | None) -> str:
    return json.dumps(
        {"labels": [{"index": i, "label": label} for i, label in enumerate(labels)]}
    )


def _tools(llm, config: ChunkConfig | None = None) -> ToolBox:
    config = config or ChunkConfig()
    return cast(
        ToolBox,
        SimpleNamespace(
            chunker=ChunkerTool(chunk_config=config),
            config=SimpleNamespace(
                chunk_config=config,
                server=SimpleNamespace(parallel_workers=2),
            ),
            llm=llm,
        ),
    )


def _headless(count: int) -> list[PrepareSegment]:
    return [
        PrepareSegment(text=f"Passage {index} with enough text to classify. " * 3)
        for index in range(count)
    ]


def _prepare(markdown: str, llm) -> PreparedDocument:
    config = ChunkConfig(
        min_size=10,
        max_size=4000,
        section_tag_min_chars=0,
        chunking_mode="naive",
        section_classifier="llm",
    )
    tools = _tools(llm, config)
    return asyncio.run(
        prepare_content_units(
            doc_from_markdown_lines(markdown),
            tools.chunker,
            config,
            PrepareOptions(section_schema_id="academic"),
            tools,
        )
    )


# --- what the model is shown ------------------------------------------------------


def test_a_headed_section_is_one_region_shown_by_its_own_heading() -> None:
    region = SectionRegion(index=3, excerpt="## Results\nFilms kept 92 %")
    segments = [
        PrepareSegment(text="## Results\n\nFilms kept 92 %", region=region),
        PrepareSegment(text="More results.", region=region),
        PrepareSegment(text="Unheaded closing remarks."),
    ]
    regions = outline_regions(segments)
    assert [(r.excerpt, r.segments) for r in regions] == [
        ("## Results\nFilms kept 92 %", (0, 1)),
        ("Unheaded closing remarks.", (2,)),
    ]


def test_an_excerpt_keeps_the_heading_line_and_flattens_the_body() -> None:
    excerpt = text_excerpt("## Methods\n\nFilms were\nannealed.\n\nThen cooled.")
    assert excerpt == "## Methods\nFilms were annealed. Then cooled."
    assert len(text_excerpt("x" * 1000)) == 300


def test_a_later_heading_never_names_an_earlier_region() -> None:
    """F10: each section is shown by its own heading, not by one further down."""
    llm = _RecordingLLM([_labels("results", "references")])
    _prepare(f"## Results\n{_FINDING}\n## References\n1. Smith J. Nature 2020.\n", llm)
    prompt = llm.prompts[0]
    assert "[0] ## Results\nFilms kept" in prompt
    assert "[1] ## References\n1. Smith" in prompt


def test_the_prompt_lists_each_label_with_its_description() -> None:
    llm = _RecordingLLM([_labels("results")])
    _prepare(f"## Results\n{_FINDING}\n", llm)
    methods = next(label for label in ACADEMIC.labels if label.id == "methods")
    assert f"- methods: {methods.description}" in llm.prompts[0]


# --- one call, weak votes ---------------------------------------------------------------


def test_one_call_labels_every_region_including_labelled_ones() -> None:
    llm = _RecordingLLM([_labels("introduction", "results", "methods")])
    document = _prepare(
        f"## Introduction\n{_FINDING}\n## Miscellany\n{_FINDING}\n## Methods\n{_SPIN}\n",
        llm,
    )
    assert len(llm.prompts) == 1
    by_text = {
        unit.headings[-1] if unit.headings else "": unit for unit in document.routed
    }
    miscellany = by_text["Miscellany"]
    assert (miscellany.section_label, miscellany.section_label_source) == (
        "results",
        S.LLM,
    )
    assert miscellany.section_label_strength is LabelStrength.WEAK


def test_the_model_never_overrides_another_tier_but_its_vote_is_kept() -> None:
    llm = _RecordingLLM([_labels("acknowledgements")])
    document = _prepare(f"## Methods\n{_SPIN}\n", llm)
    unit = document.routed[0]
    assert (unit.section_label, unit.section_label_source) == (
        "methods",
        S.HEADING_PATTERN,
    )
    assert ("acknowledgements", S.LLM) in {
        (e.label, e.source) for e in unit.section_label_evidence
    }


def test_unheaded_segments_are_shown_one_by_one() -> None:
    segments = _headless(3)
    llm = _RecordingLLM([_labels("introduction", None, "results")])
    asyncio.run(llm_label_outline(segments, _tools(llm), ACADEMIC))
    assert [s.section_label for s in segments] == ["introduction", None, "results"]
    assert segments[0].section_label_source is S.LLM


def test_a_long_outline_is_sent_in_windows() -> None:
    segments = _headless(10)
    llm = _RecordingLLM([_labels(*["results"] * 4)] * 2 + [_labels("results")] * 1)
    asyncio.run(llm_label_outline(segments, _tools(llm), ACADEMIC, window=4))
    assert len(llm.prompts) == 3
    assert "[0]" in llm.prompts[2] and "[2]" not in llm.prompts[2]
    labelled = [s.section_label for s in segments]
    assert labelled == ["results"] * 9 + [None]


def test_a_failed_window_costs_only_its_votes(caplog) -> None:
    segments = _headless(4)
    llm = _RecordingLLM(["not json", _labels("methods", "methods")])
    with caplog.at_level(logging.WARNING):
        asyncio.run(llm_label_outline(segments, _tools(llm), ACADEMIC, window=2))
    assert len(llm.prompts) == 2
    assert [s.section_label for s in segments] == [None, None, "methods", "methods"]
    assert "Outline labeling failed" in caplog.text


def test_unknown_labels_and_numbers_are_discarded() -> None:
    segments = _headless(2)
    response = json.dumps(
        {
            "labels": [
                {"index": 0, "label": "not_a_real_label"},
                {"index": 1, "label": "results"},
                {"index": 7, "label": "methods"},
            ]
        }
    )
    asyncio.run(
        llm_label_outline(segments, _tools(_RecordingLLM([response])), ACADEMIC)
    )
    assert [s.section_label for s in segments] == [None, "results"]


def test_a_failed_window_is_reported_by_error_class() -> None:
    segments = _headless(4)
    llm = _RecordingLLM([_labels("methods", "methods"), TimeoutError("slow")])
    report = asyncio.run(llm_label_outline(segments, _tools(llm), ACADEMIC, window=2))
    assert (report.windows, report.windows_failed) == (2, 1)
    assert report.errors == {"TimeoutError": 1}
    assert [s.section_label for s in segments] == ["methods", "methods", None, None]


def test_the_prepared_document_carries_the_model_tier_report() -> None:
    document = _prepare(f"## Results\n{_FINDING}\n", _RecordingLLM([ValueError()]))
    assert document.routed, "a failed tier still prepares the document"
    assert document.outline_labels is not None
    assert document.outline_labels.windows_failed == 1


def test_no_report_when_the_model_tier_is_off() -> None:
    config = ChunkConfig(min_size=10, max_size=4000, section_classifier="heuristic")
    tools = _tools(_RecordingLLM([]), config)
    document = asyncio.run(
        prepare_content_units(
            doc_from_markdown_lines(f"## Results\n{_FINDING}\n"),
            tools.chunker,
            config,
            PrepareOptions(section_schema_id="academic"),
            tools,
        )
    )
    assert document.outline_labels is None
