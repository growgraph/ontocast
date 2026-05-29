"""Tests for section tagging, filtering, and optional graph routing."""

import json
from pathlib import Path

from rdflib import URIRef

from ontocast.cli.server import expand_input_to_states
from ontocast.config import Config
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyContextMode, RenderMode, WorkflowNode
from ontocast.onto.section import (
    assign_section_labels,
    detect_section_spans,
    filter_units_by_target_sections,
    should_summarize_unit,
)
from ontocast.onto.state import AgentState
from ontocast.stategraph.routing import (
    route_after_chunk_pre,
    route_after_convert,
)

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
    )
    assert len(states) == 1
    state = states[0]
    assert state.target_sections == ["results"]
    assert state.summarize_sections == ["*"]
    assert state.summary_max_sentences == 3
    assert state.use_section_tagging is True
    assert state.use_summarization is True


def test_content_unit_extraction_text_prefers_summary() -> None:
    unit = ContentUnit(
        text="original long text",
        index=0,
        doc_iri=URIRef("http://example.org/doc"),
        summary="short summary",
    )
    assert unit.extraction_text == "short summary"
