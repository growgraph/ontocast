"""Label assignment: every tier adds evidence, one rule decides, strength is explicit.

A label's strength decides whether it may take part in a drop, so it is
recorded, not inferred from the tier: inheritance through a numbered edge from
a strong parent is strong, inheritance guessed from a heading's wording is
weak. Strong evidence beats weak; between equals the heading wins.
"""

from __future__ import annotations

import asyncio

import pytest

from ontocast.config import ChunkConfig
from ontocast.config.section_labels import load_section_label_schema
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import LabelStrength, RouteReason, SectionLabelSource
from ontocast.onto.section_models import LabelEvidence
from ontocast.tool.chunk import assignment
from ontocast.tool.chunk.assignment import content_evidence, decide
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.outline import build_document_outline
from ontocast.tool.chunk.prepare import (
    PreparedChunk,
    PreparedDocument,
    PrepareOptions,
    prepare_content_units,
)
from test.docling_test_helpers import doc_from_markdown_lines, plain_doc

pytestmark = pytest.mark.unit

ACADEMIC = load_section_label_schema("academic")
STRONG, WEAK = LabelStrength.STRONG, LabelStrength.WEAK
S = SectionLabelSource

_THANKS = (
    "We thank the beamline staff for their generous help during the experiments. "
    "This work was supported by the National Science Foundation under grant "
    "DMR-1234567. We are grateful to A. Roe for helpful discussions on the "
    "manuscript. The project was funded by the European Research Council."
)
_FINDING = (
    "Films were annealed at 100 °C for 10 min and kept 92 % of their initial "
    "efficiency after 500 h in humid air, which the encapsulation explains."
)
_SPIN = (
    "Films were spin coated at 4000 rpm for 30 s from a 1.2 M precursor "
    "solution and annealed at 100 °C for 10 min on a hot plate in a glovebox."
)


def _evidence(label: str, source: SectionLabelSource, strength: LabelStrength):
    return LabelEvidence(label=label, source=source, strength=strength)


def _prepare(markdown: str, *, plain: bool = False, **options) -> PreparedDocument:
    config = ChunkConfig(
        min_size=10,
        max_size=options.pop("max_size", 4000),
        section_tag_min_chars=0,
        chunking_mode="naive",
    )
    doc = plain_doc(markdown) if plain else doc_from_markdown_lines(markdown)
    return asyncio.run(
        prepare_content_units(
            doc,
            ChunkerTool(chunk_config=config),
            config,
            PrepareOptions(section_schema_id="academic", **options),
        )
    )


def _unit_with(document: PreparedDocument, needle: str) -> PreparedChunk:
    return next(unit for unit in document.routed if needle in unit.text)


# --- the decision rule ----------------------------------------------------------------


def test_strong_evidence_beats_weak() -> None:
    inherited = _evidence("results", S.HEADING_INHERITED, WEAK)
    density = _evidence("acknowledgements", S.CONTENT_DENSITY, STRONG)
    assert decide([inherited, density]) == density


def test_between_equals_the_heading_wins() -> None:
    heading = _evidence("methods", S.HEADING_PATTERN, STRONG)
    density = _evidence("acknowledgements", S.CONTENT_DENSITY, STRONG)
    assert decide([density, heading]) == heading
    keyword = _evidence("methods", S.HEADING_KEYWORD, WEAK)
    llm = _evidence("results", S.LLM, WEAK)
    assert decide([llm, keyword]) == keyword


def test_among_weak_votes_the_model_outranks_the_embedding_tier() -> None:
    """The model saw the heading, the opening text and the outline; the
    embedding tier saw a few words."""
    embedding = _evidence("back_matter", S.HEADING_EMBEDDING, WEAK)
    llm = _evidence("results", S.LLM, WEAK)
    keyword = _evidence("methods", S.HEADING_KEYWORD, WEAK)
    assert decide([embedding, llm]) == llm
    assert decide([llm, keyword]) == keyword


def test_no_evidence_no_label() -> None:
    assert decide([]) is None


@pytest.mark.parametrize(
    ("label", "strength"),
    [("acknowledgements", STRONG), ("references", STRONG), ("methods", WEAK)],
)
def test_only_the_conservative_content_tier_is_strong(
    monkeypatch: pytest.MonkeyPatch, label: str, strength: LabelStrength
) -> None:
    """The aggressive tier guesses methods or results; a guess is weak."""
    monkeypatch.setattr(
        assignment, "classify_by_density", lambda *_a, **_k: (label, 0.9)
    )
    evidence = content_evidence("x" * 300, ACADEMIC, aggressive=True)
    assert evidence == _evidence(label, S.CONTENT_DENSITY, strength)


# --- strength of heading evidence ---------------------------------------------------


def test_heading_pattern_is_strong_and_a_keyword_in_a_phrase_weak() -> None:
    nodes = build_document_outline(
        "## Methods\n\nBody.\n\n## Funding Mechanisms\n\nBody.\n", ACADEMIC
    ).nodes
    assert (nodes[0].source, nodes[0].strength) == (S.HEADING_PATTERN, STRONG)
    assert (nodes[1].label, nodes[1].strength) == ("acknowledgements", WEAK)


def test_inheritance_through_a_numbered_edge_from_a_strong_parent_is_strong() -> None:
    nodes = build_document_outline(
        "# 2 Methods\n\nBody.\n\n"
        "# 2.1 Spin coating of the precursor solution on glass\n\nBody.\n",
        ACADEMIC,
    ).nodes
    child = nodes[1]
    assert (child.label, child.source, child.strength) == (
        "methods",
        S.HEADING_INHERITED,
        STRONG,
    )


def test_inheritance_from_a_weak_parent_is_weak() -> None:
    nodes = build_document_outline(
        "# 2 Funding Mechanisms\n\nBody.\n\n"
        "# 2.1 Twelve agencies across five countries and two decades\n\nBody.\n",
        ACADEMIC,
    ).nodes
    assert (nodes[1].label, nodes[1].strength) == ("acknowledgements", WEAK)


def test_inheritance_guessed_from_wording_is_weak() -> None:
    nodes = build_document_outline(
        "## Results\n\nBody.\n\n"
        "## Stability of films under humid air for weeks\n\nBody.\n",
        ACADEMIC,
    ).nodes
    assert (nodes[1].label, nodes[1].strength) == ("results", WEAK)


# --- end to end -------------------------------------------------------------------------


def test_exclude_sections_reaches_numbered_subsections() -> None:
    document = _prepare(
        f"# 1 Introduction\n{_FINDING}\n"
        f"# 2 Methods\n"
        f"# 2.1 Spin coating of the precursor solution on glass\n{_SPIN}\n"
        f"# 3 Results\n{_FINDING}\n",
        exclude_sections=["methods"],
    )
    spin = _unit_with(document, "spin coated")
    assert spin.route_reason is RouteReason.EXCLUDED_BY_REQUEST
    assert spin.section_label_strength is STRONG


def test_strong_content_overrides_a_weak_heading_label() -> None:
    document = _prepare(
        f"## Results\n{_FINDING}\n"
        f"## Help received from colleagues and agencies during this work\n{_THANKS}\n"
    )
    thanks = _unit_with(document, "beamline staff")
    assert thanks.section_label == "acknowledgements"
    assert thanks.section_label_source is S.CONTENT_DENSITY
    assert thanks.section_label_strength is STRONG
    assert {(e.label, e.source) for e in thanks.section_label_evidence} == {
        ("results", S.HEADING_INHERITED),
        ("acknowledgements", S.CONTENT_DENSITY),
    }
    assert thanks.route_reason is RouteReason.BOILERPLATE


def test_a_strong_heading_keeps_its_label_against_content() -> None:
    document = _prepare(f"## Methods\n{_THANKS}\n")
    unit = _unit_with(document, "beamline staff")
    assert (unit.section_label, unit.section_label_strength) == ("methods", STRONG)
    assert ("acknowledgements", S.CONTENT_DENSITY) in {
        (e.label, e.source) for e in unit.section_label_evidence
    }


def test_unheaded_text_is_typed_per_segment() -> None:
    document = _prepare(
        f"{_FINDING} {_FINDING} {_FINDING}\n\n{_THANKS}", plain=True, max_size=500
    )
    thanks = _unit_with(document, "beamline staff")
    finding = _unit_with(document, "encapsulation explains")
    assert thanks is not finding
    assert (thanks.section_label, thanks.section_label_strength) == (
        "acknowledgements",
        STRONG,
    )
    assert finding.section_label is None
    assert finding.section_label_strength is None


def test_an_unresolved_section_stays_unlabeled() -> None:
    """No tier spreads a neighbour's label into a section the outline saw."""
    document = _prepare(f"## Results\n{_FINDING}\n## Miscellany\n{_SPIN}\n")
    spin = _unit_with(document, "spin coated")
    assert spin.section_label is None
    assert spin.section_label_evidence == ()


# --- the unit's record ------------------------------------------------------------------


def test_a_content_unit_records_strength_and_evidence_not_a_confidence() -> None:
    fields = ContentUnit.model_fields
    assert "section_label_confidence" not in fields
    assert {"section_label_strength", "section_label_evidence"} <= set(fields)
