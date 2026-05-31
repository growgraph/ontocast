"""Tests for section tagging, filtering, and optional graph routing."""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.agent.chunk_text import chunk_text
from ontocast.cli.http_parse import parse_sections_list_param
from ontocast.cli.server import expand_input_to_states
from ontocast.config import Config
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyContextMode, RenderMode, Status, WorkflowNode
from ontocast.onto.section import (
    SectionSpan,
    _match_section_label,
    assign_section_labels,
    detect_section_spans,
    filter_units_by_target_sections,
    normalise_user_section_label,
    should_summarize_unit,
)
from ontocast.onto.state import AgentState
from ontocast.stategraph.routing import (
    route_after_chunk_pre,
    route_after_convert,
)
from ontocast.tool.section_classifier import SectionClassifierTool
from ontocast.toolbox import ToolBox

_SAMPLE_DOC = """# Introduction
We survey prior work.

## Methods
We used a benchmark.

## Results
Accuracy improved by 10%.

## Future Work
We may extend the model.
"""


def test_detect_section_spans_finds_headings() -> None:
    spans = detect_section_spans(_SAMPLE_DOC)
    labels = [span.label for span in spans]
    assert "introduction" in labels
    assert "methods" in labels
    assert "results" in labels
    assert "future_work" in labels


def test_assign_section_labels_on_units() -> None:
    spans = detect_section_spans(_SAMPLE_DOC)
    units = [
        ContentUnit(
            text="Accuracy improved by 10%.",
            index=0,
            doc_iri=URIRef("http://example.org/doc"),
        )
    ]
    assign_section_labels(units, _SAMPLE_DOC, spans)
    assert units[0].section_label == "results"


def test_filter_units_by_target_sections() -> None:
    units = [
        ContentUnit(
            text="a",
            index=0,
            doc_iri=URIRef("http://example.org/doc"),
            section_label="results",
        ),
        ContentUnit(
            text="b",
            index=1,
            doc_iri=URIRef("http://example.org/doc"),
            section_label="introduction",
        ),
    ]
    filtered = filter_units_by_target_sections(units, ["results"])
    assert len(filtered) == 1
    assert filtered[0].section_label == "results"


def test_should_summarize_unit_wildcard_and_named() -> None:
    unit = ContentUnit(
        text="x",
        index=0,
        doc_iri=URIRef("http://example.org/doc"),
        section_label="results",
    )
    assert should_summarize_unit(unit, []) is True
    assert should_summarize_unit(unit, ["*"]) is True
    assert should_summarize_unit(unit, ["results"]) is True
    assert should_summarize_unit(unit, ["methods"]) is False
    assert should_summarize_unit(unit, None) is False


def test_agent_state_optional_routing_flags() -> None:
    default = AgentState()
    assert default.use_section_tagging is False
    assert default.use_summarization is False
    assert route_after_convert(default) == WorkflowNode.CHUNK
    assert route_after_chunk_pre(default) == WorkflowNode.RENDER_ONTOLOGY_UPDATE

    tagged = AgentState(target_sections=["results"])
    assert tagged.use_section_tagging is True
    assert route_after_convert(tagged) == WorkflowNode.TAG_SECTIONS

    summarized = AgentState(
        summarize_sections=["results"], render_mode=RenderMode.FACTS
    )
    assert summarized.use_summarization is True
    assert route_after_chunk_pre(summarized) == WorkflowNode.SUMMARIZE_CHUNKS


def test_expand_input_to_states_passes_section_params(tmp_path: Path) -> None:
    input_file = tmp_path / "doc.json"
    input_file.write_text(json.dumps({"text": "hello"}), encoding="utf-8")
    config = Config()
    states = expand_input_to_states(
        input_file,
        config=config,
        head_chunks=2,
        ontology_context_mode_value=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        tenant="t",
        project="p",
        target_sections=["results"],
        summarize_sections=["*"],
        summary_max_sentences=3,
        document_type_hint="annual report",
    )
    assert len(states) == 1
    state = states[0]
    assert state.target_sections == ["results"]
    assert state.summarize_sections == ["*"]
    assert state.summary_max_sentences == 3
    assert state.document_type_hint == "annual report"
    assert state.use_section_tagging is True
    assert state.use_summarization is True


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("Experimental Results", "results"),
        ("Materials and Methods", "methods"),
        ("Concluding Remarks", "conclusion"),
        ("Literature Review", "related_work"),
        ("II. Results", "results"),
        ("Chapter 3: Methods", "methods"),
        ("Section II: Results", "results"),
        ("Executive Summary", "abstract"),
        ("Bibliography", "references"),
        ("Appendices", "appendix"),
    ],
)
def test_regex_matches_section_synonyms(heading: str, expected: str) -> None:
    assert _match_section_label(heading) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Related Literature", "related_work"),
        ("Findings", "results"),
        ("Executive Summary", "abstract"),
        ("*", "*"),
        ("garbage", None),
        ("methods", "methods"),
    ],
)
def test_normalise_user_section_label_synonyms(raw: str, expected: str | None) -> None:
    assert normalise_user_section_label(raw) == expected


def test_parse_sections_list_param_normalizes() -> None:
    parsed = parse_sections_list_param("Related Literature,Methods,Findings")
    assert parsed == ["related_work", "methods", "results"]


def test_detect_section_spans_new_labels() -> None:
    doc = """# Data
We describe the corpus.

## Appendix
Extra tables.

## References
[1] Smith et al.
"""
    spans = detect_section_spans(doc)
    labels = [span.label for span in spans]
    assert "data" in labels
    assert "appendix" in labels
    assert "references" in labels


def test_chunk_text_drops_non_summarized_sections() -> None:
    class FakeChunker:
        def __call__(self, _text: str) -> list[str]:
            return ["intro text", "methods text", "results text"]

    tools = SimpleNamespace(
        chunker=FakeChunker(),
        section_classifier=SectionClassifierTool(),
        embedding_tool=SimpleNamespace(embed=lambda texts: [[0.0] * 3 for _ in texts]),
    )
    state = AgentState(
        input_text="intro text\nmethods text\nresults text",
        summarize_sections=["methods", "results"],
        section_spans=[
            SectionSpan(label="introduction", start=0, end=11),
            SectionSpan(label="methods", start=11, end=24),
            SectionSpan(label="results", start=24, end=37),
        ],
    )
    result = asyncio.run(chunk_text(state, cast(ToolBox, tools)))
    assert result.status == Status.SUCCESS
    assert len(result.content_units) == 2
    labels = {unit.section_label for unit in result.content_units}
    assert labels == {"methods", "results"}


def test_detect_section_spans_roman_numeral_results() -> None:
    doc = "II. Results\nAccuracy was 95%.\n"
    spans = detect_section_spans(doc)
    assert any(span.label == "results" for span in spans)


def test_chunk_text_max_chunks_applied_after_section_filter() -> None:
    """max_chunks must not drop target sections that appear only in later chunks."""

    chunks = [f"intro chunk {i}" for i in range(5)] + [
        "results-only chunk with accuracy metrics"
    ]

    class FakeChunker:
        def __call__(self, _text: str) -> list[str]:
            return chunks

    tools = SimpleNamespace(
        chunker=FakeChunker(),
        section_classifier=SectionClassifierTool(content_threshold=0.01),
        embedding_tool=SimpleNamespace(
            embed=lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
        ),
    )
    state = AgentState(
        input_text="\n".join(chunks),
        target_sections=["results"],
        max_chunks=2,
        section_spans=[
            SectionSpan(label="introduction", start=0, end=80),
            SectionSpan(label="results", start=80, end=200),
        ],
    )
    result = asyncio.run(chunk_text(state, cast(ToolBox, tools)))
    assert result.status == Status.SUCCESS
    assert len(result.content_units) >= 1
    assert all(unit.section_label == "results" for unit in result.content_units)


def test_chunk_text_warns_when_section_filter_drops_all(caplog) -> None:
    class FakeChunker:
        def __call__(self, _text: str) -> list[str]:
            return ["only introduction content"]

    tools = SimpleNamespace(
        chunker=FakeChunker(),
        section_classifier=SectionClassifierTool(),
        embedding_tool=SimpleNamespace(embed=lambda texts: [[0.0] * 3 for _ in texts]),
    )
    state = AgentState(
        input_text="only introduction content",
        target_sections=["results"],
        section_spans=[
            SectionSpan(label="introduction", start=0, end=30),
        ],
    )
    with caplog.at_level(logging.WARNING):
        asyncio.run(chunk_text(state, cast(ToolBox, tools)))
    assert any("removed all" in record.message for record in caplog.records)


def test_content_unit_extraction_text_prefers_summary() -> None:
    unit = ContentUnit(
        text="original long text",
        index=0,
        doc_iri=URIRef("http://example.org/doc"),
        summary="short summary",
    )
    assert unit.extraction_text == "short summary"
