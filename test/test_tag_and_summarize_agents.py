"""Unit tests for tag_sections and summarize_chunks agents."""

import logging
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.output_parsers import PydanticOutputParser
from rdflib import URIRef

from ontocast.agent.summarize_chunks import summarize_chunk
from ontocast.agent.tag_sections import (
    HeadingClassification,
    HeadingClassifications,
    tag_sections,
)
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status, WorkflowNode
from ontocast.onto.state import AgentState
from ontocast.stategraph.node_factories import make_summarize_chunks_node
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


def _mock_embed(texts: list[str]) -> list[list[float]]:
    """Axis-aligned vectors so prototype centroids align by section family."""

    def vector_for(text: str) -> list[float]:
        lowered = text.lower()
        if any(
            token in lowered
            for token in (
                "result",
                "finding",
                "evaluation",
                "experiment",
                "ablation",
            )
        ):
            return [1.0, 0.0, 0.0]
        if any(
            token in lowered
            for token in ("method", "approach", "setup", "implementation")
        ):
            return [0.0, 1.0, 0.0]
        return [0.1, 0.1, 0.1]

    return [vector_for(text) for text in texts]


def _build_tools(
    *,
    llm=None,
    embed=None,
    section_classifier: SectionClassifierTool | None = None,
    parallel_workers: int = 2,
) -> ToolBox:
    async def default_llm(_prompt):
        raise AssertionError("LLM should not be called")

    return cast(
        ToolBox,
        SimpleNamespace(
            llm=llm or default_llm,
            section_classifier=section_classifier
            or SectionClassifierTool(content_threshold=0.01, heading_threshold=0.01),
            embedding_tool=SimpleNamespace(embed=embed or _mock_embed),
            config=SimpleNamespace(
                server=SimpleNamespace(parallel_workers=parallel_workers)
            ),
        ),
    )


def _llm_classifications_json(
    *items: tuple[str, str | None],
) -> str:
    parser = PydanticOutputParser(pydantic_object=HeadingClassifications)
    payload = HeadingClassifications(
        classifications=[
            HeadingClassification(heading=heading, label=label)
            for heading, label in items
        ]
    )
    parser.parse(payload.model_dump_json())
    return payload.model_dump_json()


def _force_llm_classifier() -> SimpleNamespace:
    """Classifier that always defers to LLM for non-regex headings."""

    return SimpleNamespace(
        classify_heading_with_confidence=lambda _heading, _embed_fn: (
            None,
            0.0,
            True,
        ),
        normalise_llm_label=SectionClassifierTool.normalise_llm_label,
    )


def _content_unit(
    *,
    text: str = "Long text with facts.",
    index: int = 0,
    section_label: str | None = "results",
) -> ContentUnit:
    return ContentUnit(
        text=text,
        index=index,
        doc_iri=URIRef("http://example.org/doc"),
        section_label=section_label,
    )


@pytest.mark.anyio
async def test_tag_sections_noop_when_disabled() -> None:
    state = AgentState(input_text=_SAMPLE_DOC)
    assert state.use_section_tagging is False

    result = await tag_sections(state, _build_tools())

    assert result is state
    assert result.section_spans == []
    assert result.status == Status.SUCCESS


@pytest.mark.anyio
async def test_tag_sections_fails_on_empty_input() -> None:
    state = AgentState(target_sections=["results"], input_text="")

    result = await tag_sections(state, _build_tools())

    assert result.status == Status.FAILED
    assert result.section_spans == []


@pytest.mark.anyio
async def test_tag_sections_regex_only_without_llm() -> None:
    llm_called = False

    async def llm(_prompt):
        nonlocal llm_called
        llm_called = True
        raise AssertionError("regex headings should not need LLM")

    state = AgentState(input_text=_SAMPLE_DOC, target_sections=["results"])
    result = await tag_sections(state, _build_tools(llm=llm))

    assert result.status == Status.SUCCESS
    labels = {span.label for span in result.section_spans}
    assert labels >= {"introduction", "methods", "results", "future_work"}
    assert llm_called is False


@pytest.mark.anyio
async def test_tag_sections_embedding_high_confidence_without_llm() -> None:
    llm_called = False

    async def llm(_prompt):
        nonlocal llm_called
        llm_called = True
        raise AssertionError("high-confidence embedding should not need LLM")

    doc = "# Key Findings\nAccuracy improved by 10%.\n"
    state = AgentState(input_text=doc, target_sections=["results"])
    result = await tag_sections(state, _build_tools(llm=llm))

    assert result.status == Status.SUCCESS
    assert any(span.label == "results" for span in result.section_spans)
    assert llm_called is False


@pytest.mark.anyio
async def test_tag_sections_llm_fallback_classifies_ambiguous_heading() -> None:
    doc = "# Obscure Heading\nSome content.\n"

    async def llm(_prompt):
        return SimpleNamespace(
            content=_llm_classifications_json(("Obscure Heading", "discussion"))
        )

    state = AgentState(input_text=doc, target_sections=["discussion"])
    tools = _build_tools(
        llm=llm,
        section_classifier=cast(SectionClassifierTool, _force_llm_classifier()),
    )
    result = await tag_sections(state, tools)

    assert result.status == Status.SUCCESS
    assert any(span.label == "discussion" for span in result.section_spans)


@pytest.mark.anyio
async def test_tag_sections_llm_failure_is_non_fatal(caplog) -> None:
    doc = "# Obscure Heading\nSome content.\n"

    async def llm(_prompt):
        raise RuntimeError("LLM unavailable")

    state = AgentState(input_text=doc, target_sections=["results"])
    tools = _build_tools(
        llm=llm,
        section_classifier=cast(SectionClassifierTool, _force_llm_classifier()),
    )
    with caplog.at_level(logging.WARNING):
        result = await tag_sections(state, tools)

    assert result.status == Status.SUCCESS
    assert result.section_spans == []
    assert any(
        "LLM section heading classification failed" in record.message
        for record in caplog.records
    )


@pytest.mark.anyio
async def test_summarize_chunk_returns_stripped_summary() -> None:
    captured: dict[str, object] = {}

    async def llm(prompt):
        captured["prompt"] = prompt
        return SimpleNamespace(content="  Compressed summary.  ")

    unit = _content_unit(text="Original long text with many details.")
    summary = await summarize_chunk(unit, _build_tools(llm=llm), max_sentences=2)

    assert summary == "Compressed summary."
    prompt_text = str(captured["prompt"])
    assert "results" in prompt_text
    assert "Original long text with many details." in prompt_text


@pytest.mark.anyio
async def test_summarize_chunk_uses_unclassified_when_section_label_missing() -> None:
    captured: dict[str, object] = {}

    async def llm(prompt):
        captured["prompt"] = prompt
        return SimpleNamespace(content="Summary.")

    unit = _content_unit(section_label=None)
    summary = await summarize_chunk(unit, _build_tools(llm=llm), max_sentences=3)

    assert summary == "Summary."
    assert "unclassified" in str(captured["prompt"])


@pytest.mark.anyio
async def test_summarize_chunk_raises_on_empty_response() -> None:
    async def llm(_prompt):
        return SimpleNamespace(content="   ")

    unit = _content_unit()
    with pytest.raises(ValueError, match="empty text"):
        await summarize_chunk(unit, _build_tools(llm=llm), max_sentences=2)


@pytest.mark.anyio
async def test_summarize_chunks_node_skips_when_disabled() -> None:
    unit = _content_unit()
    state = AgentState(content_units=[unit], summarize_sections=None)
    node = make_summarize_chunks_node(_build_tools())

    result = await node(state)

    assert result.status == Status.SUCCESS
    assert unit.summary is None


@pytest.mark.anyio
async def test_summarize_chunks_node_filters_by_section() -> None:
    calls: list[int] = []

    async def llm(_prompt):
        unit_index = calls[-1]
        return SimpleNamespace(content=f"summary for unit {unit_index}")

    async def tracking_llm(prompt):
        calls.append(len(calls))
        return await llm(prompt)

    units = [
        _content_unit(text="results text", index=0, section_label="results"),
        _content_unit(text="intro text", index=1, section_label="introduction"),
    ]
    state = AgentState(
        content_units=units,
        summarize_sections=["results"],
        summary_max_sentences=2,
    )
    node = make_summarize_chunks_node(_build_tools(llm=tracking_llm))

    result = await node(state)

    assert result.status == Status.SUCCESS
    assert result.get_node_status(WorkflowNode.SUMMARIZE_CHUNKS) == Status.SUCCESS
    assert units[0].summary == "summary for unit 0"
    assert units[1].summary is None
    assert len(calls) == 1


@pytest.mark.anyio
async def test_summarize_chunks_node_tolerates_per_unit_failure() -> None:
    call_count = 0

    async def llm(_prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("summarization failed")
        return SimpleNamespace(content="ok summary")

    units = [
        _content_unit(text="first", index=0, section_label="results"),
        _content_unit(text="second", index=1, section_label="results"),
    ]
    state = AgentState(
        content_units=units,
        summarize_sections=["*"],
        summary_max_sentences=2,
    )
    node = make_summarize_chunks_node(_build_tools(llm=llm))

    result = await node(state)

    assert result.status == Status.SUCCESS
    assert units[0].summary is None
    assert units[1].summary == "ok summary"
