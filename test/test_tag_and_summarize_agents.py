"""Unit tests for chunk prepare (section tagging) and summarize_chunks agents."""

import logging
from types import SimpleNamespace
from typing import Literal, cast

import pytest
from rdflib import URIRef

from ontocast.agent.chunk_text import chunk_text
from ontocast.agent.summarize_chunks import ensure_unit_summary, summarize_chunk
from ontocast.config import ChunkConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.section_llm import ChunkSectionClassification
from ontocast.toolbox import ToolBox
from test.docling_test_helpers import doc_from_markdown_lines

pytestmark = pytest.mark.unit

_SAMPLE_DOC = """# Introduction
We survey prior work.

## Methods
We used a benchmark.

## Results
Accuracy improved by 10%.

## Future Work
We may extend the model.
"""


def _build_tools(
    *,
    llm=None,
    parallel_workers: int = 2,
    min_size: int = 50,
    max_size: int = 2000,
    section_classifier: Literal["llm", "heuristic", "heading", "off"] = "llm",
) -> ToolBox:
    config = ChunkConfig(
        min_size=min_size,
        max_size=max_size,
        section_classifier=section_classifier,
    )

    async def default_llm(_prompt):
        raise AssertionError("LLM should not be called")

    return cast(
        ToolBox,
        SimpleNamespace(
            chunker=ChunkerTool(chunk_config=config),
            config=SimpleNamespace(
                chunk_config=config,
                server=SimpleNamespace(parallel_workers=parallel_workers),
            ),
            llm=llm or default_llm,
        ),
    )


def _chunk_label_json(label: str | None) -> str:
    payload = ChunkSectionClassification(label=label)
    return payload.model_dump_json()


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
async def test_chunk_prepare_no_section_options_still_classifies() -> None:
    """Classification is default-on: headed chunks get labels with zero LLM calls."""
    state = AgentState(
        docling_doc=doc_from_markdown_lines(_SAMPLE_DOC),
    )
    assert state.needs_section_prepare is False
    result = await chunk_text(state, _build_tools())
    assert result.status == Status.SUCCESS
    assert result.content_units
    labels = {unit.section_label for unit in result.content_units}
    assert labels & {"introduction", "methods", "results", "future_work"}


@pytest.mark.anyio
async def test_chunk_prepare_classifier_off_uses_simple_path() -> None:
    state = AgentState(
        docling_doc=doc_from_markdown_lines(_SAMPLE_DOC),
    )
    result = await chunk_text(state, _build_tools(section_classifier="off"))
    assert result.status == Status.SUCCESS
    assert result.content_units
    assert all(unit.section_label is None for unit in result.content_units)


@pytest.mark.anyio
async def test_chunk_prepare_fails_without_docling_doc() -> None:
    state = AgentState(target_sections=["results"])
    result = await chunk_text(state, _build_tools())
    assert result.status == Status.FAILED


@pytest.mark.anyio
async def test_chunk_prepare_regex_labels_without_llm() -> None:
    llm_calls = 0

    async def llm(_prompt):
        nonlocal llm_calls
        llm_calls += 1
        raise AssertionError("LLM should not be called")

    state = AgentState(
        docling_doc=doc_from_markdown_lines(_SAMPLE_DOC),
        target_sections=["results"],
    )
    result = await chunk_text(state, _build_tools(llm=llm))
    assert result.status == Status.SUCCESS
    assert result.content_units
    assert all(unit.section_label == "results" for unit in result.content_units)
    assert llm_calls == 0


@pytest.mark.slow
@pytest.mark.anyio
async def test_chunk_prepare_keyword_tier_labels_non_regex_chunk() -> None:
    """A heading the anchored patterns miss is still resolved without an LLM."""
    llm_calls = 0

    async def llm(_prompt):
        nonlocal llm_calls
        llm_calls += 1
        raise AssertionError("LLM should not be called")

    state = AgentState(
        docling_doc=doc_from_markdown_lines(
            "# Key Findings\nAccuracy improved by 10%.\n"
        ),
        target_sections=["results"],
    )
    result = await chunk_text(state, _build_tools(llm=llm))
    assert result.status == Status.SUCCESS
    assert result.content_units
    assert result.content_units[0].section_label == "results"
    assert llm_calls == 0


@pytest.mark.anyio
async def test_chunk_prepare_llm_classifies_unnamed_chunk() -> None:
    """Only headings no deterministic tier can name reach the LLM."""
    llm_calls = 0

    async def llm(_prompt):
        nonlocal llm_calls
        llm_calls += 1
        return SimpleNamespace(
            content='{"assignments": [{"index": 0, "label": "results"}]}'
        )

    state = AgentState(
        docling_doc=doc_from_markdown_lines(
            "# Widget Telemetry Rollup\nAccuracy improved by 10%.\n"
        ),
        target_sections=["results"],
    )
    result = await chunk_text(state, _build_tools(llm=llm))
    assert result.status == Status.SUCCESS
    assert result.content_units
    assert result.content_units[0].section_label == "results"
    assert llm_calls == 1


@pytest.mark.anyio
async def test_chunk_prepare_llm_failure_is_non_fatal(caplog) -> None:
    async def llm(_prompt):
        raise RuntimeError("LLM unavailable")

    state = AgentState(
        docling_doc=doc_from_markdown_lines("# Obscure Heading\nSome content.\n"),
        summarize_sections=["*"],
    )
    with caplog.at_level(logging.WARNING):
        result = await chunk_text(state, _build_tools(llm=llm))
    assert result.status == Status.SUCCESS


@pytest.mark.anyio
async def test_summarize_chunk_returns_stripped_summary() -> None:
    async def llm(_prompt):
        return SimpleNamespace(content="  Short summary.  ")

    unit = _content_unit()
    summary = await summarize_chunk(unit, _build_tools(llm=llm), max_sentences=3)
    assert summary == "Short summary."


@pytest.mark.anyio
async def test_summarize_chunk_uses_unclassified_when_section_label_missing() -> None:
    captured: dict[str, str] = {}

    async def llm(prompt):
        captured["prompt"] = str(prompt)
        return SimpleNamespace(content="Summary.")

    unit = _content_unit(section_label=None)
    await summarize_chunk(unit, _build_tools(llm=llm), max_sentences=2)
    assert "unclassified" in captured["prompt"]


@pytest.mark.anyio
async def test_summarize_chunk_raises_on_empty_response() -> None:
    async def llm(_prompt):
        return SimpleNamespace(content="   ")

    unit = _content_unit()
    with pytest.raises(ValueError, match="empty"):
        await summarize_chunk(unit, _build_tools(llm=llm), max_sentences=3)


@pytest.mark.anyio
async def test_ensure_unit_summary_skips_when_disabled() -> None:
    state = AgentState(
        content_units=[_content_unit()],
        summarize_sections=None,
    )
    await ensure_unit_summary(state, 0, _build_tools())
    assert state.content_units[0].summary is None


@pytest.mark.anyio
async def test_ensure_unit_summary_filters_by_section() -> None:
    summarized: list[str] = []

    async def llm(prompt):
        summarized.append(str(prompt))
        return SimpleNamespace(content="Summary text.")

    state = AgentState(
        content_units=[
            _content_unit(text="results text", index=0, section_label="results"),
            _content_unit(text="intro text", index=1, section_label="introduction"),
        ],
        summarize_sections=["results"],
    )
    tools = _build_tools(llm=llm)
    await ensure_unit_summary(state, 0, tools)
    await ensure_unit_summary(state, 1, tools)
    assert state.content_units[0].summary == "Summary text."
    assert state.content_units[1].summary is None
    assert len(summarized) == 1


@pytest.mark.anyio
async def test_ensure_unit_summary_tolerates_per_unit_failure() -> None:
    calls = 0

    async def llm(_prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return SimpleNamespace(content="ok")

    state = AgentState(
        content_units=[
            _content_unit(text="first", index=0, section_label="results"),
            _content_unit(text="second", index=1, section_label="results"),
        ],
        summarize_sections=["results"],
    )
    tools = _build_tools(llm=llm)
    await ensure_unit_summary(state, 0, tools)
    await ensure_unit_summary(state, 1, tools)
    assert state.content_units[0].summary is None
    assert state.content_units[1].summary == "ok"


@pytest.mark.anyio
async def test_ensure_unit_summary_is_idempotent() -> None:
    """Both fan-outs call it, so the second must not re-spend a provider call."""
    calls = 0

    async def llm(_prompt):
        nonlocal calls
        calls += 1
        return SimpleNamespace(content="Summary text.")

    state = AgentState(
        content_units=[_content_unit(text="body", index=0, section_label="results")],
        summarize_sections=["results"],
    )
    tools = _build_tools(llm=llm)
    await ensure_unit_summary(state, 0, tools)
    await ensure_unit_summary(state, 0, tools)
    assert state.content_units[0].summary == "Summary text."
    assert calls == 1
