"""Tests for section tagging, filtering, and optional graph routing."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.agent.chunk_text import chunk_text
from ontocast.agent.summarize_chunks import should_summarize_unit
from ontocast.api.parse import parse_sections_list_param
from ontocast.api.process_helpers import expand_input_to_states
from ontocast.config import Config
from ontocast.config.section_labels import (
    load_section_label_schema,
    match_heading_line,
    normalise_user_section_label,
)
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyContextMode, RenderMode, Status, WorkflowNode
from ontocast.onto.state import AgentState
from ontocast.stategraph.routing import route_after_chunk, route_after_convert
from ontocast.tool.chunk.prepare import (
    PrepareSegment,
    _forward_fill_section_labels,
    _tag_segments,
)
from ontocast.tool.chunk.sections import (
    detect_section_spans,
    document_text_for_section_tagging,
    label_from_headings,
    label_text_from_spans,
    resolve_section_label,
)
from ontocast.toolbox import ToolBox
from test.docling_test_helpers import doc_from_markdown_lines

_SAMPLE_DOC = """# Introduction
We survey prior work.

## Methods
We used a benchmark.

## Results
Accuracy improved by 10%.

## Future Work
We may extend the model.
"""


def _academic_schema():
    return load_section_label_schema("academic")


_UNHEADED_ABSTRACT_DOC = """We study nanocrystal superlattices and report cooperative emission
under disorder. Our model explains the low yield of superradiance.

# Introduction
We survey prior work on perovskite assemblies.

## Methods
We used optical measurements.
"""


def test_detect_section_spans_injects_unheaded_abstract() -> None:
    doc = doc_from_markdown_lines(_UNHEADED_ABSTRACT_DOC)
    document_text = document_text_for_section_tagging(doc)
    spans = detect_section_spans(document_text, _academic_schema())
    labels = [span.label for span in spans]
    assert labels[0] == "abstract"
    assert "introduction" in labels
    label, _ = label_text_from_spans(
        "We study nanocrystal superlattices",
        document_text,
        spans,
        0,
    )
    assert label == "abstract"


def test_detect_section_spans_on_exported_markdown() -> None:
    doc = doc_from_markdown_lines(_SAMPLE_DOC)
    document_text = document_text_for_section_tagging(doc)
    spans = detect_section_spans(document_text, _academic_schema())
    labels = [span.label for span in spans]
    assert "introduction" in labels
    assert "methods" in labels
    assert "results" in labels
    assert "future_work" in labels


def test_label_text_from_spans_inherits_section_in_order() -> None:
    doc = doc_from_markdown_lines(_SAMPLE_DOC)
    document_text = document_text_for_section_tagging(doc)
    spans = detect_section_spans(document_text, _academic_schema())
    search_from = 0
    label, search_from = label_text_from_spans(
        "We survey prior work.", document_text, spans, search_from
    )
    assert label == "introduction"
    label, search_from = label_text_from_spans(
        "We used a benchmark.", document_text, spans, search_from
    )
    assert label == "methods"


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
    assert default.needs_section_prepare is False
    assert default.use_summarization is False
    assert route_after_convert(default) == WorkflowNode.CHUNK
    assert route_after_chunk(default) == WorkflowNode.RENDER_ONTOLOGY_UPDATE

    tagged = AgentState(target_sections=["results"])
    assert tagged.needs_section_prepare is True
    assert route_after_chunk(tagged) == WorkflowNode.RENDER_ONTOLOGY_UPDATE

    summarized = AgentState(
        summarize_sections=["results"], render_mode=RenderMode.FACTS
    )
    assert summarized.use_summarization is True
    assert route_after_chunk(summarized) == WorkflowNode.SUMMARIZE_CHUNKS


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
        section_schema_id="financial",
    )
    assert len(states) == 1
    state = states[0]
    assert state.target_sections == ["results"]
    assert state.summarize_sections == ["*"]
    assert state.summary_max_sentences == 3
    assert state.document_type_hint == "annual report"
    assert state.section_schema_id == "financial"
    assert state.needs_section_prepare is True
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
        ("Abstract.", "abstract"),
        ("ABSTRACT", "abstract"),
        ("Abstract —", "abstract"),
        ("Bibliography", "references"),
        ("Appendices", "appendix"),
    ],
)
def test_regex_matches_section_synonyms(heading: str, expected: str) -> None:
    schema = _academic_schema()
    assert match_heading_line(heading, schema) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Related Literature", "related_work"),
        ("Findings", "results"),
        ("Executive Summary", "abstract"),
        ("*", "*"),
        ("garbage", None),
        ("methods", "methods"),
        ("risk_factors", "risk_factors"),
    ],
)
def test_normalise_user_section_label_synonyms(raw: str, expected: str | None) -> None:
    assert normalise_user_section_label(raw) == expected


def test_parse_sections_list_param_normalizes() -> None:
    parsed = parse_sections_list_param("Related Literature,Methods,Findings")
    assert parsed == ["related_work", "methods", "results"]


def _chunk_tools() -> ToolBox:
    from ontocast.config import ChunkConfig
    from ontocast.tool.chunk.chunker import ChunkerTool

    config = ChunkConfig(min_size=50, max_size=2000)

    async def noop_llm(_prompt):
        return SimpleNamespace(content='{"label": null}')

    return cast(
        ToolBox,
        SimpleNamespace(
            chunker=ChunkerTool(chunk_config=config),
            config=SimpleNamespace(
                chunk_config=config,
                server=SimpleNamespace(parallel_workers=2),
            ),
            llm=noop_llm,
        ),
    )


def _sample_units() -> list[ContentUnit]:
    return [
        ContentUnit(
            text="We survey prior work.",
            index=0,
            doc_iri=URIRef("http://example.org/doc"),
        ),
        ContentUnit(
            text="We used a benchmark.",
            index=1,
            doc_iri=URIRef("http://example.org/doc"),
        ),
        ContentUnit(
            text="Accuracy improved by 10%.",
            index=2,
            doc_iri=URIRef("http://example.org/doc"),
        ),
        ContentUnit(
            text="We may extend the model.",
            index=3,
            doc_iri=URIRef("http://example.org/doc"),
        ),
    ]


@pytest.mark.anyio
async def test_chunk_prepare_filters_summarize_sections_allowlist() -> None:
    doc = doc_from_markdown_lines(_SAMPLE_DOC)
    state = AgentState(
        docling_doc=doc,
        summarize_sections=["methods", "results"],
    )
    result = await chunk_text(state, _chunk_tools())
    assert result.status == Status.SUCCESS
    labels = {unit.section_label for unit in result.content_units}
    assert labels <= {"methods", "results"}
    assert "methods" in labels
    assert "results" in labels


@pytest.mark.anyio
async def test_chunk_text_max_chunks_after_prepare() -> None:
    doc = doc_from_markdown_lines(_SAMPLE_DOC)
    state = AgentState(
        docling_doc=doc,
        target_sections=["results"],
        max_chunks=2,
    )
    result = await chunk_text(state, _chunk_tools())
    assert result.status == Status.SUCCESS
    assert len(result.content_units) <= 2
    assert all(unit.section_label == "results" for unit in result.content_units)


@pytest.mark.anyio
async def test_chunk_prepare_warns_when_section_filter_drops_all(caplog) -> None:
    doc = doc_from_markdown_lines("# Introduction\nOnly intro.\n")
    state = AgentState(
        docling_doc=doc,
        target_sections=["results"],
    )
    with caplog.at_level(logging.WARNING):
        await chunk_text(state, _chunk_tools())
    assert any("removed all" in record.message for record in caplog.records)


def test_content_unit_extraction_text_prefers_summary() -> None:
    unit = ContentUnit(
        text="original long text",
        index=0,
        doc_iri=URIRef("http://example.org/doc"),
        summary="short summary",
    )
    assert unit.extraction_text == "short summary"


# ---------------------------------------------------------------------------
# label_from_headings — heading-breadcrumb labeling
# ---------------------------------------------------------------------------


def test_label_from_headings_most_specific_wins() -> None:
    schema = _academic_schema()
    # "Dataset" matches the "data" label — most specific (last) heading wins.
    assert label_from_headings(["Paper Title", "Methods", "Dataset"], schema) == "data"


def test_label_from_headings_falls_back_to_broader() -> None:
    schema = _academic_schema()
    # "Unknown Subsection" does not match; falls back to "Results" which does.
    assert label_from_headings(["Results", "Unknown Subsection"], schema) == "results"


def test_label_from_headings_none_when_no_match() -> None:
    schema = _academic_schema()
    assert label_from_headings(["Paper Title", "Some Unknown Section"], schema) is None


def test_label_from_headings_empty_returns_none() -> None:
    schema = _academic_schema()
    assert label_from_headings([], schema) is None
    assert label_from_headings(None, schema) is None


# ---------------------------------------------------------------------------
# resolve_section_label — search_from cursor preservation
# ---------------------------------------------------------------------------


def test_resolve_section_label_preserves_cursor_on_miss() -> None:
    """When chunk text is not found, search_from must not reset to 0."""

    document_text = "# Introduction\nBody text.\n# Results\nFindings here."
    spans = detect_section_spans(document_text, _academic_schema())
    # Advance cursor past the introduction span.
    _, advanced = resolve_section_label("Body text.", document_text, spans, 0)
    assert advanced > 0
    # Now a chunk whose text doesn't appear in the document at all.
    label, cursor_after_miss = resolve_section_label(
        "TEXT_NOT_IN_DOCUMENT", document_text, spans, advanced
    )
    assert label is None
    assert cursor_after_miss == advanced, (
        "Cursor must not reset to 0 on a failed find; "
        f"expected {advanced}, got {cursor_after_miss}"
    )


# ---------------------------------------------------------------------------
# _tag_segments — headings take priority over span search
# ---------------------------------------------------------------------------


def test_tag_segments_prefers_headings_over_span_search() -> None:
    """Segments with docling heading metadata are labeled via headings, not span search."""
    schema = _academic_schema()
    doc = doc_from_markdown_lines(_SAMPLE_DOC)
    document_text = document_text_for_section_tagging(doc)
    spans = detect_section_spans(document_text, schema)

    # Segment whose text does NOT appear verbatim in document_text but whose
    # headings clearly identify the section.
    segment = PrepareSegment(
        text="This text is not in the document at all.",
        headings=["Paper Title", "Results"],
    )
    _tag_segments([segment], document_text, spans, schema)
    assert segment.section_label == "results"


def test_tag_segments_falls_back_to_span_when_no_headings() -> None:
    schema = _academic_schema()
    doc = doc_from_markdown_lines(_SAMPLE_DOC)
    document_text = document_text_for_section_tagging(doc)
    spans = detect_section_spans(document_text, schema)

    segment = PrepareSegment(
        text="We used a benchmark.",
        headings=None,
    )
    _tag_segments([segment], document_text, spans, schema)
    assert segment.section_label == "methods"


# ---------------------------------------------------------------------------
# _forward_fill_section_labels
# ---------------------------------------------------------------------------


def test_forward_fill_propagates_label_to_unlabeled_neighbor() -> None:
    schema = _academic_schema()
    segments = [
        PrepareSegment(text="We discuss findings.", section_label="results"),
        PrepareSegment(text="Further analysis shows improvement."),
        PrepareSegment(text="Table 3 summarizes the data."),
    ]
    _forward_fill_section_labels(segments, schema)
    assert segments[1].section_label == "results"
    assert segments[2].section_label == "results"


def test_forward_fill_does_not_propagate_before_first_label() -> None:
    schema = _academic_schema()
    segments = [
        PrepareSegment(text="Some preamble without a label."),
        PrepareSegment(text="Introduction body.", section_label="introduction"),
    ]
    _forward_fill_section_labels(segments, schema)
    assert segments[0].section_label is None


def test_forward_fill_blocked_by_section_heading_line() -> None:
    """A segment that starts with a heading must not inherit the preceding label."""
    schema = _academic_schema()
    segments = [
        PrepareSegment(text="Results body text here.", section_label="results"),
        PrepareSegment(text="# Methods\nWe used a dataset."),
    ]
    _forward_fill_section_labels(segments, schema)
    assert segments[1].section_label is None
