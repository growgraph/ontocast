"""Heading normalisation and the keyword recall tier.

The positive table is drawn from headings measured on real docling conversions
(ACS and Nature-family papers) plus common variants the anchored patterns miss.
The negative table pins publisher banners that must stay unlabeled -- labeling
them would attach body text to the wrong section.
"""

import pytest

from ontocast.config.section_labels import (
    load_section_label_schema,
    match_heading_line,
    normalise_heading_line,
    normalise_user_section_label,
    resolve_heading_label,
)

pytestmark = pytest.mark.unit

SCHEMA = load_section_label_schema("academic")

RECOGNISED = [
    ("## Abstract", "abstract"),
    ("## Introduction", "introduction"),
    ("## Results", "results"),
    ("## Results and Discussion", "results"),
    ("## RESULTS AND DISCUSSION", "results"),
    ("## 3. Results and Discussion", "results"),
    ("## Methods", "methods"),
    ("## Materials and Methods", "methods"),
    ("## Experimental", "methods"),
    ("## Experimental Section", "methods"),
    ("## 2.1 Synthesis of MAPbI3 films", "methods"),
    ("## Device fabrication", "methods"),
    ("## Characterization", "methods"),
    ("## 4 Discussion and Conclusions", "discussion"),
    ("## Conclusions and Outlook", "conclusion"),
    ("## Data Availability", "data"),
    ("## Supporting Information", "appendix"),
    ("## *sı Supporting Information", "appendix"),
    ("## ■ ASSOCIATED CONTENT", "appendix"),
    ("## Author Contributions", "acknowledgements"),
    ("## Competing Interests", "acknowledgements"),
    ("## ■ ACKNOWLEDGMENTS", "acknowledgements"),
    ("## References", "references"),
    ("## ■ REFERENCES", "references"),
]

UNRECOGNISED = [
    "## ARTICLE",
    "## ORCID",
    "## Notes",
    "## ACCESS",
    "## Cooperative ensemble breaks the population-inversion limitation",
]


@pytest.mark.parametrize(("heading", "expected"), RECOGNISED)
def test_headings_resolve_to_expected_label(heading: str, expected: str):
    resolved = resolve_heading_label(heading, SCHEMA)

    assert resolved is not None, heading
    assert resolved[0] == expected


@pytest.mark.parametrize("heading", UNRECOGNISED)
def test_publisher_banners_stay_unlabeled(heading: str):
    assert resolve_heading_label(heading, SCHEMA) is None


class TestNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("## ■ REFERENCES", "REFERENCES"),
            ("**Results and Discussion**", "Results and Discussion"),
            ("2.1 Synthesis of films", "Synthesis of films"),
            ("1 Introduction", "Introduction"),
            ("Chapter 3: Methods", "Methods"),
            ("Results:", "Results"),
        ],
    )
    def test_decoration_and_numbering_are_stripped(self, raw: str, expected: str):
        assert normalise_heading_line(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["I Introduction", "A Framework for Analysis"],
    )
    def test_bare_letters_are_not_mistaken_for_numbering(self, raw: str):
        """Stripping "I"/"A" without a separator would eat the first word."""
        assert normalise_heading_line(raw) == raw

    def test_appendix_letter_survives(self):
        assert normalise_heading_line("Appendix A") == "Appendix A"


class TestPrecisionBoundary:
    """``match_heading_line`` stays exact; recall lives in the keyword tier."""

    def test_compound_heading_is_not_pattern_matched(self):
        assert match_heading_line("Results and Discussion", SCHEMA) is None

    def test_compound_heading_is_keyword_matched(self):
        resolved = resolve_heading_label("Results and Discussion", SCHEMA)
        assert resolved is not None
        assert resolved[2] == "heading_keyword"

    def test_exact_heading_prefers_the_pattern_tier(self):
        resolved = resolve_heading_label("Results", SCHEMA)
        assert resolved is not None
        assert resolved[2] == "heading_pattern"
        assert resolved[1] > 0.9


class TestCompoundHeadingOrdering:
    """Compound headings resolve to their leading component."""

    @pytest.mark.parametrize(
        ("heading", "expected"),
        [
            ("Results and Discussion", "results"),
            ("Discussion and Conclusions", "discussion"),
            ("Conclusions and Outlook", "conclusion"),
            ("Introduction and Background", "introduction"),
        ],
    )
    def test_leading_component_wins(self, heading: str, expected: str):
        resolved = resolve_heading_label(heading, SCHEMA)
        assert resolved is not None
        assert resolved[0] == expected


class TestUserLabelNormalisationUnaffected:
    """The permissive tier must not leak into user-supplied label parsing."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("results", "results"),
            ("Results", "results"),
            ("methods", "methods"),
            ("*", "*"),
            ("not a section at all", None),
        ],
    )
    def test_user_labels_resolve_predictably(self, raw: str, expected: str | None):
        assert normalise_user_section_label(raw) == expected
