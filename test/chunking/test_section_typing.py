"""Heading typing: terminal labels, head-noun keywords, subtitled headings, one vocabulary.

A label is evidence for routing and the key for ``target_sections``, so a wrong
label misdirects selection even when routing keeps the text.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from ontocast.config import ChunkConfig
from ontocast.config.section_labels import (
    RouteClass,
    label_route_class,
    load_section_label_schema,
    resolve_heading_label,
)
from ontocast.onto.enum import RouteReason, SectionLabelSource, UnitRoute
from ontocast.tool.chunk.non_content import is_non_content_unit
from ontocast.tool.chunk.outline import build_document_outline
from ontocast.tool.chunk.prepare import PreparedChunk
from ontocast.tool.chunk.routing import route_unit
from ontocast.tool.chunk.section_llm import fragment_for_text

pytestmark = pytest.mark.unit

ACADEMIC = load_section_label_schema("academic")
MANUAL = load_section_label_schema("manual")


def _label(heading: str, schema=ACADEMIC) -> str | None:
    resolved = resolve_heading_label(heading, schema)
    return resolved[0] if resolved else None


# --- terminal labels ------------------------------------------------------------


@pytest.mark.parametrize(
    "heading",
    ["References", "## ■ REFERENCES", "References and Notes", "Notes and References"],
)
def test_reference_headings_resolve(heading: str) -> None:
    assert _label(heading) == "references"


@pytest.mark.parametrize(
    "heading",
    [
        "Comparison with literature references",
        "Benchmark references",
        "Cross-references",
    ],
)
def test_a_heading_that_only_mentions_references_is_not_a_reference_list(
    heading: str,
) -> None:
    """F2: a terminal label matches by keyword only as a whole conjunct."""
    assert _label(heading) != "references"


def test_descriptive_headings_after_references_do_not_inherit_it() -> None:
    """F1: a reference list has no subsections."""
    outline = build_document_outline(
        "# Results\n\nBody.\n\n# References\n\n1. A ref.\n\n"
        "# Stability of CsPbBr3 films under humid air exposure\n\nBody.\n",
        ACADEMIC,
    )
    labels = [(node.text, node.label) for node in outline.nodes]
    assert labels[1] == ("References", "references")
    assert labels[2][1] is None
    assert outline.nodes[2].source is SectionLabelSource.OUTLINE_UNRESOLVED


def test_descriptive_headings_still_inherit_from_other_sections() -> None:
    outline = build_document_outline(
        "# Methods\n\nBody.\n\n# Spin coating of the precursor solution on glass\n\n",
        ACADEMIC,
    )
    assert outline.nodes[1].label == "methods"
    assert outline.nodes[1].source is SectionLabelSource.HEADING_INHERITED


# --- subtitled headings -----------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "label"),
    [
        ("Appendix A: Derivation of the degradation rate", "appendix"),
        ("Supplementary Note 2: XRD", "appendix"),
        ("Methods: spin coating, annealing and encapsulation of devices", "methods"),
    ],
)
def test_the_head_of_a_subtitled_heading_is_typed(heading: str, label: str) -> None:
    """F6: "Head: subtitle" opens the head's section instead of inheriting."""
    outline = build_document_outline(
        f"# Results\n\nBody.\n\n# {heading}\n\nBody.\n", ACADEMIC
    )
    node = outline.nodes[1]
    assert node.label == label
    assert node.sectionlike
    assert node.source in (
        SectionLabelSource.HEADING_PATTERN,
        SectionLabelSource.HEADING_KEYWORD,
    )


def test_a_subtitled_descriptive_heading_still_inherits() -> None:
    outline = build_document_outline(
        "# Results\n\nBody.\n\n# Aging study: films stored in humid air for weeks\n\n",
        ACADEMIC,
    )
    assert outline.nodes[1].label == "results"
    assert outline.nodes[1].source is SectionLabelSource.HEADING_INHERITED


# --- head-noun keyword resolution ---------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "label"),
    [
        ("Characterization Results", "results"),  # F8
        ("Summary of Results", "conclusion"),  # a summary, of results
        ("Materials Characterization", "methods"),
        ("Device Fabrication", "methods"),
        ("Results and Discussion", "results"),
        ("Experimental Section", "methods"),
        ("Conclusions and Outlook", "conclusion"),
        ("4 Discussion and Conclusions", "discussion"),
        ("2.1 Synthesis of MAPbI3 films", "methods"),
        ("Simulation Results", "results"),
    ],
)
def test_keywords_resolve_by_head_noun(heading: str, label: str) -> None:
    assert _label(heading) == label


# --- one vocabulary for front and back matter ---------------------------------------


@pytest.mark.parametrize(
    "heading",
    ["ORCID", "Notes", "Authors", "Affiliations", "Data Availability", "Open Access"],
)
def test_academic_back_matter_headings_are_non_content_labels(heading: str) -> None:
    label = _label(heading)
    assert label == "back_matter"
    assert label_route_class(label, ACADEMIC) is RouteClass.NON_CONTENT


def test_a_manual_has_no_back_matter_vocabulary() -> None:
    """F11: "Notes" in a manual is content."""
    text = (
        "## Notes\nTighten the bolts before powering the unit on, never past the mark."
    )
    assert not is_non_content_unit(text, None, MANUAL)
    assert is_non_content_unit(text, None, ACADEMIC)


def test_heading_rule_needs_a_schema() -> None:
    text = "## Notes\nThe authors declare no competing financial interest."
    assert not is_non_content_unit(text, None, None)


def _route(text: str, label: str | None, source, *, schema=ACADEMIC, **config: Any):
    decision = route_unit(
        PreparedChunk(
            text=text, headings=None, section_label=label, section_label_source=source
        ),
        schema=schema,
        schema_fallback=config.pop("schema_fallback", False),
        exclude_sections=None,
        config=ChunkConfig(**cast(Any, config)),
    )
    return decision.route, decision.reason


def test_strong_back_matter_label_takes_the_non_content_route() -> None:
    text = "John Doe: 0000-0001-2345-6789. Jane Roe contributed the analysis."
    strong = SectionLabelSource.HEADING_PATTERN
    assert _route(text, "back_matter", strong) == (
        UnitRoute.NON_CONTENT,
        RouteReason.NON_CONTENT,
    )
    assert _route(text, "back_matter", strong, non_content_mode="skip")[0] is (
        UnitRoute.SKIP
    )


def test_back_matter_with_a_measurement_is_extracted() -> None:
    text = "Data availability: films annealed at 100 °C for 10 min are archived."
    assert _route(text, "back_matter", SectionLabelSource.HEADING_PATTERN) == (
        UnitRoute.EXTRACT,
        RouteReason.LABEL_CONTRADICTED,
    )


def test_fallback_schema_has_no_heading_vocabulary() -> None:
    text = "## Notes\nThe authors declare no competing financial interest."
    assert _route(
        text, "back_matter", SectionLabelSource.HEADING_PATTERN, schema_fallback=True
    ) == (UnitRoute.EXTRACT, RouteReason.FALLBACK_SCHEMA)


# --- the LLM excerpt ------------------------------------------------------------------


def test_llm_excerpt_is_the_leading_heading_only() -> None:
    """F10: a heading later in the segment does not name the segment."""
    prose = "Encapsulated devices kept 92 % efficiency after 1000 h."
    segment = f"{prose}\n\n## References\n\n1. Smith 2020"
    assert fragment_for_text(segment).startswith("Encapsulated devices")
    assert fragment_for_text("## Methods\n\nFilms were annealed.") == "## Methods"


# --- coreference ------------------------------------------------------------------------


def test_citation_units_are_not_coreference_candidates() -> None:
    from rdflib import URIRef

    from ontocast.onto.constants import DEFAULT_IRI
    from ontocast.onto.content_unit import ContentUnit, OutputType
    from ontocast.onto.rdfgraph import RDFGraph
    from ontocast.tool.agg.coreference import coreference_candidates

    cd = f"{DEFAULT_IRI}/"

    def unit(index: int, ttl: str, *, citation: bool = False) -> ContentUnit:
        graph = RDFGraph()
        graph.parse(
            data=f"@prefix cd: <{cd}> .\n@prefix q: <https://x.org/q#> .\n{ttl}",
            format="turtle",
        )
        return ContentUnit(
            text=f"unit {index}",
            index=index,
            doc_iri=URIRef("https://x.org/doc/1"),
            graph=graph,
            type=OutputType.FACTS,
            is_citation_metadata=citation,
        )

    units = [
        unit(0, "cd:smith__u0 a q:Person ; q:authorOf cd:work__u0 ."),
        unit(1, "cd:j_smith__u1 a q:Person ; q:authorOf cd:work__u1 .", citation=True),
    ]
    final = {cd + "work__u0": cd + "Work", cd + "work__u1": cd + "Work"}
    merges = [(cd + "smith__u0", cd + "j_smith__u1", 0.6)]

    assert (
        coreference_candidates(
            unconfirmed_merges=merges,
            lexical_near_misses=[],
            units=units,
            final_uri=final,
            max_pairs=10,
        )
        == []
    )
