"""Outline detection, span closing and heading genericity.

The regression these tests guard: an unrecognised heading used to leave the
preceding section span open until the next *recognised* heading, so one label
smeared across the rest of the document -- and because the label was stamped on
segments at split time, no later tier could correct it.
"""

import pytest

from ontocast.config.section_labels import load_section_label_schema
from ontocast.onto.enum import SectionLabelSource
from ontocast.tool.chunk.outline import (
    build_document_outline,
    heading_is_sectionlike,
    markdown_headings,
    outline_to_spans,
    text_headings,
)
from ontocast.tool.chunk.sections import detect_section_spans

pytestmark = pytest.mark.unit

SCHEMA = load_section_label_schema("academic")

# Headings the anchored patterns cannot match, between headings they can.
_UNRECOGNISED_MIDDLE_DOC = """## Introduction

Perovskite solar cells have attracted attention for a decade.

## Experimental Section

Films were deposited by spin coating from a precursor solution.

## Results and Discussion

The efficiency reached 24.1% after 500 hours of continuous operation.

## Conclusions and Outlook

We demonstrated an approach to improved operational stability.

## References

[1] Foo, B. et al. J. Chem. 2020.
"""

# A descriptive subsection title inside Results, as emitted by docling for
# Nature-family papers.
_DESCRIPTIVE_SUBSECTION_DOC = """## Results

The assemblies show a superlinear emission onset.

## Cooperative ensemble breaks the population-inversion limitation

Time-resolved traces reveal a threshold at low fluence.

## Discussion

The mechanism is consistent with cooperative emission.
"""

# Back matter that is section-like but has no schema pattern.
_BACK_MATTER_DOC = """## Results

The measured yield was stable across the series.

## Notes

The authors declare no competing financial interest.

## References

[1] Bar, Q. Nature 2021.
"""


def _labels(spans):
    return [span.label for span in spans]


def test_unrecognized_heading_closes_previous_span():
    """Each heading ends the previous section, recognised or not."""
    spans = detect_section_spans(_UNRECOGNISED_MIDDLE_DOC, SCHEMA)

    assert len(spans) == 5
    introduction = next(span for span in spans if span.label == "introduction")
    covered = _UNRECOGNISED_MIDDLE_DOC[introduction.start : introduction.end]
    assert "Experimental Section" not in covered
    assert "Results and Discussion" not in covered
    assert "Conclusions and Outlook" not in covered


def test_outline_spans_partition_document():
    """Spans tile the document with no gaps and no overlaps."""
    spans = detect_section_spans(_UNRECOGNISED_MIDDLE_DOC, SCHEMA)

    assert spans[0].start == 0
    assert spans[-1].end == len(_UNRECOGNISED_MIDDLE_DOC)
    for left, right in zip(spans, spans[1:]):
        assert left.end == right.start


def test_descriptive_subsection_inherits_parent_label():
    """A descriptive subheading stays inside its section instead of splitting it off."""
    spans = detect_section_spans(_DESCRIPTIVE_SUBSECTION_DOC, SCHEMA)

    assert _labels(spans) == ["results", "results", "discussion"]
    inherited = spans[1]
    assert inherited.source is SectionLabelSource.HEADING_INHERITED
    assert (
        "Time-resolved traces"
        in _DESCRIPTIVE_SUBSECTION_DOC[inherited.start : inherited.end]
    )


def test_sectionlike_unrecognized_heading_stays_unresolved():
    """An unnamed back-matter section is explicitly unresolved, not inherited."""
    spans = detect_section_spans(_BACK_MATTER_DOC, SCHEMA)

    notes = next(
        span
        for span in spans
        if "Notes" in _BACK_MATTER_DOC[span.start : span.end][:20]
    )
    assert notes.label is None
    assert notes.source is SectionLabelSource.OUTLINE_UNRESOLVED


def test_front_matter_labeled_abstract_when_first_section_is_not_imrad():
    """Unheaded front matter is recovered even when the paper opens with Results."""
    doc = "A study of assemblies. " * 8 + "\n\n## Results\n\nThe yield was high.\n"
    spans = detect_section_spans(doc, SCHEMA)

    assert spans[0].label == "abstract"
    assert spans[0].start == 0
    assert spans[0].source is SectionLabelSource.FRONT_MATTER


class TestHeadingGenericity:
    """``heading_is_sectionlike`` separates section names from descriptive titles."""

    def test_generic_section_names(self):
        for heading in (
            "Results",
            "Results and Discussion",
            "Experimental Section",
            "Materials and Methods",
            "Synthesis of thin films",
            "Data availability",
        ):
            assert heading_is_sectionlike(heading), heading

    def test_descriptive_titles_and_document_titles(self):
        for heading in (
            "Cooperative ensemble breaks the population-inversion limitation",
            "Halide Perovskite Artificial Solids as a New Platform to Simulate "
            "Collective Phenomena",
            "Investigation into the Photoluminescence Red Shift in Nanocrystals",
        ):
            assert not heading_is_sectionlike(heading), heading

    def test_empty_heading(self):
        assert not heading_is_sectionlike("")


class TestHeadingDetection:
    def test_markdown_headings_carry_exact_offsets(self):
        nodes = markdown_headings(_UNRECOGNISED_MIDDLE_DOC)

        assert [node.text for node in nodes][:2] == [
            "Introduction",
            "Experimental Section",
        ]
        for node in nodes:
            assert (
                _UNRECOGNISED_MIDDLE_DOC[node.start :]
                .lstrip("#")
                .startswith(" " + node.text)
            )

    def test_text_headings_for_documents_without_markdown(self):
        doc = (
            "RESULTS AND DISCUSSION\n\n"
            "The measured efficiency was stable.\n\n"
            "REFERENCES\n\n"
            "[1] Foo, B. Nature 2020.\n"
        )
        assert markdown_headings(doc) == []

        nodes = text_headings(doc)
        assert [node.text for node in nodes] == [
            "RESULTS AND DISCUSSION",
            "REFERENCES",
        ]

    def test_text_headings_ignore_prose_lines(self):
        doc = "The measured efficiency was stable across the series.\n\nWe conclude.\n"
        assert text_headings(doc) == []

    def test_text_heading_fallback_is_opt_in(self):
        doc = (
            "RESULTS\n\nThe yield was high enough to matter.\n\nREFERENCES\n\n[1] X.\n"
        )
        without = detect_section_spans(doc, SCHEMA, include_text_headings=False)
        with_headings = detect_section_spans(doc, SCHEMA, include_text_headings=True)
        assert _labels(without) == [None]
        assert "results" in _labels(with_headings)


def test_outline_without_headings_yields_single_unresolved_span():
    outline = build_document_outline("Body text with no headings at all.", SCHEMA)
    spans = outline_to_spans(outline)

    assert len(spans) == 1
    assert spans[0].label is None
    assert spans[0].start == 0
