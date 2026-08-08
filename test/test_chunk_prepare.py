"""Tests for chunk prepare pipeline (segment, tag, filter, size)."""

import asyncio
from types import SimpleNamespace
from typing import cast

from ontocast.agent.chunk_text import chunk_text
from ontocast.config import ChunkConfig
from ontocast.onto.enum import RenderMode
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.prepare import PrepareOptions, prepare_content_units
from ontocast.toolbox import ToolBox
from test.docling_test_helpers import doc_from_markdown_lines

_SECTION_OPTS = PrepareOptions(summarize_sections=["*"])

_MULTI_SECTION_DOC = """# Abstract
Short abstract body.

# Introduction
We survey prior work on knowledge graphs.

# Related Work
Many systems extract triples from text.

# Methods
We used a benchmark dataset.
"""


def _tools(
    *,
    min_size: int = 80,
    max_size: int = 500,
    llm=None,
) -> ToolBox:
    config = ChunkConfig(min_size=min_size, max_size=max_size)

    async def default_llm(_prompt):
        raise AssertionError("LLM should not be called")

    return cast(
        ToolBox,
        SimpleNamespace(
            chunker=ChunkerTool(chunk_config=config),
            config=SimpleNamespace(
                chunk_config=config,
                server=SimpleNamespace(parallel_workers=2),
            ),
            llm=llm or default_llm,
        ),
    )


async def _prepare(
    doc,
    *,
    min_size: int = 80,
    max_size: int = 500,
    options: PrepareOptions | None = None,
    llm=None,
):
    tools = _tools(min_size=min_size, max_size=max_size, llm=llm)
    opts = options if options is not None else _SECTION_OPTS
    return await prepare_content_units(
        doc,
        tools.chunker,
        tools.chunker.config,
        opts,
        tools,
    )


def test_prepare_merges_segments_within_section() -> None:
    doc = doc_from_markdown_lines(_MULTI_SECTION_DOC)
    chunks = asyncio.run(_prepare(doc, min_size=80, max_size=500))

    assert chunks
    assert all(len(chunk.text) > 15 for chunk in chunks)
    labels = {chunk.section_label for chunk in chunks}
    assert "abstract" in labels
    assert "introduction" in labels
    assert "related_work" in labels


def test_prepare_does_not_merge_across_sections() -> None:
    doc = doc_from_markdown_lines(_MULTI_SECTION_DOC)
    chunks = asyncio.run(
        _prepare(
            doc,
            min_size=80,
            max_size=2000,
            options=PrepareOptions(),
        )
    )

    for chunk in chunks:
        if chunk.section_label == "introduction":
            assert "Related Work" not in chunk.text
            assert "prior work" in chunk.text.lower()
        if chunk.section_label == "related_work":
            assert "Introduction" not in chunk.text
            assert "triples" in chunk.text.lower()


def test_prepare_sets_section_label_on_chunks() -> None:
    doc = doc_from_markdown_lines(_MULTI_SECTION_DOC)
    chunks = asyncio.run(
        _prepare(
            doc,
            min_size=50,
            max_size=500,
        )
    )

    labeled = [chunk for chunk in chunks if chunk.section_label is not None]
    assert labeled
    assert all(chunk.section_label for chunk in labeled)


def test_prepare_splits_oversized_section() -> None:
    long_body = "Word " * 400
    doc = doc_from_markdown_lines(f"# Results\n{long_body.strip()}")
    chunks = asyncio.run(
        _prepare(doc, min_size=100, max_size=300, options=_SECTION_OPTS)
    )

    assert len(chunks) >= 2
    assert all(chunk.section_label == "results" for chunk in chunks)
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_prepare_fallback_when_no_structural_segments() -> None:
    doc = doc_from_markdown_lines("")
    chunks = asyncio.run(_prepare(doc, options=PrepareOptions()))
    assert chunks == []


def test_prepare_merges_small_same_label_segments() -> None:
    doc = doc_from_markdown_lines(_MULTI_SECTION_DOC)
    chunks = asyncio.run(_prepare(doc, min_size=200, max_size=2000))

    intro_chunks = [c for c in chunks if c.section_label == "introduction"]
    assert len(intro_chunks) == 1
    assert "prior work" in intro_chunks[0].text.lower()


def test_prepare_target_sections_filters_before_merge() -> None:
    doc = doc_from_markdown_lines(_MULTI_SECTION_DOC)
    chunks = asyncio.run(
        _prepare(
            doc,
            min_size=50,
            max_size=2000,
            options=PrepareOptions(target_sections=["methods"]),
        )
    )

    assert chunks
    assert all(chunk.section_label == "methods" for chunk in chunks)
    assert "benchmark" in chunks[0].text.lower()


def test_prepare_different_labels_not_merged() -> None:
    doc = doc_from_markdown_lines(_MULTI_SECTION_DOC)
    chunks = asyncio.run(_prepare(doc, min_size=500, max_size=2000))

    assert len(chunks) >= 3


def test_chunk_text_assigns_section_labels_and_avoids_tiny_chunks() -> None:
    tools = _tools(min_size=80, max_size=500)
    state = AgentState(
        render_mode=RenderMode.ONTOLOGY,
        summarize_sections=["*"],
    )
    state.set_docling_doc(doc_from_markdown_lines(_MULTI_SECTION_DOC))

    asyncio.run(chunk_text(state, tools))

    assert state.content_units
    assert all(len(unit.text) > 15 for unit in state.content_units)
    assert any(unit.section_label == "introduction" for unit in state.content_units)
    assert any(unit.section_label == "related_work" for unit in state.content_units)


def test_chunk_text_resets_content_units_on_each_call() -> None:
    from ontocast.onto.docling_helpers import plain_text_to_docling_doc

    tools = _tools()
    state = AgentState(render_mode=RenderMode.ONTOLOGY)
    state.set_docling_doc(plain_text_to_docling_doc("first invocation text", "doc"))
    asyncio.run(chunk_text(state, tools))
    assert len(state.content_units) == 1

    state.set_docling_doc(plain_text_to_docling_doc("second invocation text", "doc"))
    asyncio.run(chunk_text(state, tools))
    assert len(state.content_units) == 1


_EXCLUDABLE_DOC = """# Introduction
We survey prior work on knowledge graphs and their applications in materials science.

# Results
Accuracy improved by ten percent on the benchmark, and precision remained stable.

# Acknowledgements
We thank the funding agency and colleagues for helpful discussions of this work.

# Appendix
Supplementary derivations and additional tables are provided for completeness.
"""


def test_academic_default_exclude_drops_acknowledgements_and_appendix() -> None:
    doc = doc_from_markdown_lines(_EXCLUDABLE_DOC)
    chunks = asyncio.run(
        _prepare(
            doc,
            min_size=40,
            max_size=500,
            options=PrepareOptions(document_type_hint="research article"),
        )
    )
    labels = {chunk.section_label for chunk in chunks}
    assert "introduction" in labels
    assert "results" in labels
    assert "acknowledgements" not in labels
    assert "appendix" not in labels


def test_explicit_empty_exclude_overrides_schema_default() -> None:
    doc = doc_from_markdown_lines(_EXCLUDABLE_DOC)
    chunks = asyncio.run(
        _prepare(
            doc,
            min_size=40,
            max_size=500,
            options=PrepareOptions(
                document_type_hint="research article",
                exclude_sections=[],
            ),
        )
    )
    labels = {chunk.section_label for chunk in chunks}
    assert "acknowledgements" in labels
    assert "appendix" in labels


def test_explicit_exclude_denylist_is_used_verbatim() -> None:
    doc = doc_from_markdown_lines(_EXCLUDABLE_DOC)
    chunks = asyncio.run(
        _prepare(
            doc,
            min_size=40,
            max_size=500,
            options=PrepareOptions(
                document_type_hint="research article",
                exclude_sections=["results"],
            ),
        )
    )
    labels = {chunk.section_label for chunk in chunks}
    assert "results" not in labels
    # Schema defaults do not apply when an explicit denylist is given.
    assert "acknowledgements" in labels


def test_allowlist_and_denylist_compose() -> None:
    doc = doc_from_markdown_lines(_EXCLUDABLE_DOC)
    chunks = asyncio.run(
        _prepare(
            doc,
            min_size=40,
            max_size=500,
            options=PrepareOptions(
                document_type_hint="research article",
                target_sections=["results", "acknowledgements"],
            ),
        )
    )
    labels = {chunk.section_label for chunk in chunks}
    # Allowlist admits results+acknowledgements; schema default then drops
    # acknowledgements.
    assert labels == {"results"}


def test_sections_first_semantic_labels_inherited_without_llm() -> None:
    """Semantic default: labels come from span blocks, zero LLM calls."""
    doc = doc_from_markdown_lines(_EXCLUDABLE_DOC)

    async def exploding_llm(_prompt):
        raise AssertionError("LLM should not be called for fully headed docs")

    chunks = asyncio.run(
        _prepare(
            doc,
            min_size=40,
            max_size=500,
            options=PrepareOptions(document_type_hint="research article"),
            llm=exploding_llm,
        )
    )
    assert chunks
    assert all(chunk.section_label is not None for chunk in chunks)


def test_docling_segmenter_option_still_supported() -> None:
    doc = doc_from_markdown_lines(_EXCLUDABLE_DOC)
    config = ChunkConfig(min_size=40, max_size=500, segmenter="docling")

    async def default_llm(_prompt):
        raise AssertionError("LLM should not be called")

    tools = cast(
        ToolBox,
        SimpleNamespace(
            chunker=ChunkerTool(chunk_config=config),
            config=SimpleNamespace(
                chunk_config=config,
                server=SimpleNamespace(parallel_workers=2),
            ),
            llm=default_llm,
        ),
    )
    chunks = asyncio.run(
        prepare_content_units(
            doc,
            tools.chunker,
            tools.chunker.config,
            PrepareOptions(document_type_hint="research article"),
            tools,
        )
    )
    labels = {chunk.section_label for chunk in chunks}
    assert "results" in labels
    assert "acknowledgements" not in labels


_UNRECOGNISED_HEADINGS_DOC = """# Introduction
Perovskite solar cells have attracted attention for a decade now.

# Experimental Section
Films were deposited by spin coating from a precursor solution at speed.

# Results and Discussion
The efficiency reached 24.1% after 500 hours of continuous operation.

# Conclusions and Outlook
We demonstrated an approach to substantially improved operational stability.
"""


def test_target_sections_selects_only_the_requested_section():
    """End-to-end guard for the span-smear regression.

    Every heading here except ``Introduction`` is invisible to the anchored
    patterns. Before the outline fix, the ``introduction`` span ran to the end
    of the document and ``target_sections=["results"]`` returned introduction
    text. No LLM is involved: the stub raises if it is called.
    """
    chunks = asyncio.run(
        _prepare(
            doc_from_markdown_lines(_UNRECOGNISED_HEADINGS_DOC),
            min_size=1,
            max_size=5000,
            options=PrepareOptions(target_sections=["results"]),
        )
    )

    assert chunks
    assert {chunk.section_label for chunk in chunks} == {"results"}
    body = "\n".join(chunk.text for chunk in chunks)
    assert "24.1%" in body
    assert "Perovskite solar cells" not in body
    assert "spin coating" not in body


def test_unlabeled_chunks_from_distinct_sections_are_not_merged():
    """A run of unresolved chunks must not be merged back together."""
    doc = doc_from_markdown_lines(
        "# Notes\nFirst unnamed section body text here.\n\n"
        "# Vocabulary\nSecond unnamed section body text here.\n"
    )
    chunks = asyncio.run(
        _prepare(doc, min_size=5000, max_size=8000, options=PrepareOptions())
    )

    unlabeled = [chunk for chunk in chunks if chunk.section_label is None]
    assert len(unlabeled) == 2
