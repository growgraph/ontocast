"""Embedding heading tier: an unnamed section-like heading votes for its nearest label.

The vote is weak and needs a near, clear winner; topical headings abstain.
Only headings the lexical tiers left unnamed are embedded, and only section-
like ones: a descriptive subsection title belongs to its parent section.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from ontocast.config import ChunkConfig
from ontocast.config.section_labels import load_section_label_schema
from ontocast.onto.enum import LabelStrength, RouteReason, SectionLabelSource
from ontocast.tool.chunk import prepare
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.heading_embedding import (
    HEADING_MIN_MARGIN,
    HEADING_MIN_SIMILARITY,
    label_prototypes,
    nearest_label,
)
from ontocast.tool.chunk.outline import build_document_outline
from ontocast.tool.chunk.prepare import (
    PreparedChunk,
    PreparedDocument,
    PrepareOptions,
    heading_embedder,
    prepare_content_units,
)
from test.docling_test_helpers import doc_from_markdown_lines

pytestmark = pytest.mark.unit

ACADEMIC = load_section_label_schema("academic")
S = SectionLabelSource

_SPIN = (
    "Films were spin coated at 4000 rpm for 30 s from a 1.2 M precursor "
    "solution and annealed at 100 °C for 10 min on a hot plate."
)
_FINDING = (
    "Films kept 92 % of their initial efficiency after 500 h in humid air, "
    "which the encapsulation explains."
)


class _ConceptEmbedder:
    """Embeds a text as the axis of the concept it names, exactly.

    The named texts map to their concept's axis; every other text to a last
    axis of its own, far from every label.
    """

    def __init__(self, axes: dict[str, int], dims: int = 8):
        self._axes = {text.lower(): axis for text, axis in axes.items()}
        self._dims = dims
        self.seen: list[str] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        vectors = []
        for text in texts:
            vector = [0.0] * self._dims
            axis = self._axes.get(text.lower())
            vector[axis if axis is not None else self._dims - 1] = 1.0
            vectors.append(vector)
        return vectors


_CONCLUSION, _METHODS, _BACK_MATTER = 0, 1, 2


def _embedder() -> _ConceptEmbedder:
    return _ConceptEmbedder(
        {
            "conclusion": _CONCLUSION,
            "Closing Thoughts": _CONCLUSION,
            "methods": _METHODS,
            "Lab Work": _METHODS,
            "back matter": _BACK_MATTER,
            "Access": _BACK_MATTER,
        }
    )


# --- the vote --------------------------------------------------------------------------


def test_prototypes_are_each_label_name_and_its_keywords() -> None:
    prototypes = label_prototypes(ACADEMIC)
    assert ("related_work", "related work") in prototypes
    assert ("results", "findings") in prototypes


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ([1.0, 0.0, 0.0], "conclusion"),  # near and clear
        ([0.6, 0.0, 0.8], None),  # nearest is too far
        ([0.72, 0.66, 0.2145], None),  # near, but no clear winner
    ],
)
def test_a_vote_needs_a_near_and_clear_nearest_label(
    heading: list[float], expected: str | None
) -> None:
    assert HEADING_MIN_SIMILARITY == 0.70 and HEADING_MIN_MARGIN == 0.10
    prototypes = [("conclusion", [1.0, 0.0, 0.0]), ("methods", [0.0, 1.0, 0.0])]
    assert nearest_label(heading, prototypes) == expected


def test_an_unnamed_section_heading_takes_its_nearest_label_weakly() -> None:
    embed = _embedder()
    nodes = build_document_outline(
        "## Methods\n\nBody.\n\n## Closing Thoughts\n\nBody.\n\n"
        "## Perovskite Stability\n\nBody.\n",
        ACADEMIC,
        embed=embed,
    ).nodes
    assert (nodes[1].label, nodes[1].source, nodes[1].strength) == (
        "conclusion",
        S.HEADING_EMBEDDING,
        LabelStrength.WEAK,
    )
    assert nodes[2].label is None
    assert "Methods" not in embed.seen  # named by its pattern, never embedded


def test_a_heading_nearest_a_non_content_label_abstains() -> None:
    """A banner heading near back matter must not relabel the content under it."""
    nodes = build_document_outline(
        "## Access\n\nBody text of the whole paper.\n", ACADEMIC, embed=_embedder()
    ).nodes
    assert nodes[0].label is None


def test_a_descriptive_title_is_not_embedded() -> None:
    embed = _embedder()
    build_document_outline(
        "## Results\n\nBody.\n\n"
        "## Stability of films under humid air for weeks\n\nBody.\n",
        ACADEMIC,
        embed=embed,
    )
    assert not any(text.startswith("Stability") for text in embed.seen)


def test_numbered_children_inherit_an_embedded_label_as_its_vote() -> None:
    """The child's label is still the embedding's, so it ranks as that tier."""
    nodes = build_document_outline(
        "# 2 Lab Work\n\nBody.\n\n"
        "# 2.1 Spin coating of the precursor solution on glass\n\nBody.\n",
        ACADEMIC,
        embed=_embedder(),
    ).nodes
    assert (nodes[1].label, nodes[1].source, nodes[1].strength) == (
        "methods",
        S.HEADING_EMBEDDING,
        LabelStrength.WEAK,
    )


# --- in preparation ----------------------------------------------------------------


def _prepare(markdown: str, **options) -> PreparedDocument:
    config = ChunkConfig(
        min_size=10, max_size=4000, section_tag_min_chars=0, chunking_mode="naive"
    )
    return asyncio.run(
        prepare_content_units(
            doc_from_markdown_lines(markdown),
            ChunkerTool(chunk_config=config),
            config,
            PrepareOptions(section_schema_id="academic", **options),
        )
    )


def _unit_with(document: PreparedDocument, needle: str) -> PreparedChunk:
    return next(unit for unit in document.routed if needle in unit.text)


def test_an_embedded_label_selects_but_never_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare, "heading_embedder", lambda _s, _c: _embedder())
    markdown = f"## Results\n{_FINDING}\n## Lab Work\n{_SPIN}\n"

    selected = _unit_with(_prepare(markdown, target_sections=["methods"]), "spin")
    assert (selected.section_label, selected.section_label_source) == (
        "methods",
        S.HEADING_EMBEDDING,
    )

    excluded = _unit_with(_prepare(markdown, exclude_sections=["methods"]), "spin")
    assert excluded.route_reason is not RouteReason.EXCLUDED_BY_REQUEST


@pytest.mark.parametrize(
    ("classifier", "runs"),
    [("off", False), ("heading", False), ("heuristic", True), ("llm", True)],
)
def test_the_tier_runs_from_the_heuristic_level(
    classifier: Literal["llm", "heuristic", "heading", "off"], runs: bool
) -> None:
    config = ChunkConfig(section_classifier=classifier, chunking_mode="naive")
    splitter = ChunkerTool(chunk_config=config)
    assert (heading_embedder(splitter, config) is not None) is runs


# --- with the chunker's model ----------------------------------------------------------


def test_the_chunker_model_names_paraphrases_and_spares_topics() -> None:
    splitter = ChunkerTool(chunk_config=ChunkConfig(chunking_mode="naive"))
    if splitter.embed_texts(["probe"]) is None:
        pytest.skip("semantic extras / embedding model unavailable")
    nodes = build_document_outline(
        "## Closing Thoughts\n\nBody.\n\n## Simulation Setup\n\nBody.\n\n"
        "## Perovskite Stability\n\nBody.\n\n## Miscellany\n\nBody.\n",
        ACADEMIC,
        embed=splitter.embed_texts,
    ).nodes
    assert [node.label for node in nodes] == ["conclusion", "methods", None, None]
