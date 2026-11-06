"""Section routing: a drop needs a strong label and agreeing content, or content alone.

Covers the block-level citation classifier, trimming of citation runs, the
routing rule itself, and the end-to-end prepare behaviour on documents whose
headings or contents used to silence content sections.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ontocast.config import ChunkConfig
from ontocast.config.section_labels import load_section_label_schema
from ontocast.onto.enum import LabelStrength, SectionLabelSource
from ontocast.tool.chunk.bibliography import (
    citation_share,
    is_citation_entry,
    looks_like_bibliography,
    split_blocks,
    trim_citation_runs,
)
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.prepare import (
    PreparedChunk,
    PreparedDocument,
    PrepareOptions,
    prepare_content_units,
)
from ontocast.tool.chunk.routing import (
    RouteReason,
    UnitRoute,
    route_unit,
)
from ontocast.toolbox import ToolBox
from test.docling_test_helpers import doc_from_markdown_lines

pytestmark = pytest.mark.unit

_REFERENCE_LIST = """[1] J. Smith, A. Jones, Perovskite degradation under humidity, Nature Mater.
14, 193-198 (2015). doi:10.1038/nmat4150
[2] L. Chen et al., Photoluminescence of CsPbBr3 nanocrystals, Nano Lett. 15,
3692-3696 (2015). doi:10.1021/nl5048779
[3] M. Garcia, Stability of halide perovskites, Chem. Rev. 119, 3036 (2019).
doi:10.1021/acs.chemrev.8b00539
[4] K. Tanaka, Exciton dynamics in lead halide perovskites, Phys. Rev. B 92,
045414 (2015). doi:10.1103/PhysRevB.92.045414
[5] R. Novak, Encapsulation strategies for perovskite devices, Adv. Mater. 30,
1806702 (2018). doi:10.1002/adma.201806702
[6] P. Ivanov, Ion migration in mixed halide films, J. Phys. Chem. Lett. 9,
1234-1240 (2018). doi:10.1021/acs.jpclett.8b00123"""

_PROSE = (
    "In conclusion, encapsulated MAPbI3 devices retained 92 % of their initial "
    "efficiency after 1000 h at 85 °C and 85 % relative humidity, while "
    "unencapsulated devices failed within 200 h. The degradation proceeds through "
    "PbI2 formation at grain boundaries, which the encapsulant slows by excluding "
    "moisture from the film."
)

# A table that cites its sources holds data: literature values compared side by side.
_DOI_TABLE = (
    "| Material | PCE (%) | Stability (h) | Ref |\n|---|---|---|---|\n"
    + "\n".join(
        f"| MAPbI3 variant {i} | {18 + i}.{i} | {500 + 100 * i} "
        f"| doi:10.1021/acsenergylett.{i}b0{i}123 |"
        for i in range(8)
    )
)

# Related work written as an enumerated discussion: entries state measurements.
_RELATED_WORK_LIST = """Prior studies differ in their aging protocols.
(1) Smith et al. (2019) stored films at 85 °C in nitrogen and reported a 20 % efficiency loss, Adv. Mater. 31, 1802.
(2) Jones et al. (2020) used 60 % relative humidity for 500 h, see doi 10.1039/c9ee01234a.
(3) Lee et al. (2021) combined light soaking at 1 sun with heat at 65 °C, Joule 5, 12."""

_ACKNOWLEDGEMENTS = (
    "We thank the beamline staff for their help and colleagues for helpful "
    "discussions of this work. The authors acknowledge support from the national "
    "research agency."
)


# --- block classifier ---------------------------------------------------------


def test_reference_list_entries_are_one_block_each() -> None:
    blocks = [block for block in split_blocks(_REFERENCE_LIST) if block.kind == "text"]
    assert len(blocks) == 6
    assert all(is_citation_entry(block.text) for block in blocks)


def test_reference_list_is_a_bibliography() -> None:
    assert citation_share(_REFERENCE_LIST) == 1.0
    assert looks_like_bibliography(_REFERENCE_LIST)


def test_table_rows_never_count_as_citations() -> None:
    """F3: a literature-comparison table with a DOI column is data."""
    assert citation_share(_DOI_TABLE) == 0.0
    assert not looks_like_bibliography(_DOI_TABLE)


def test_entries_stating_measurements_are_not_citations() -> None:
    """F3: an enumerated related-work discussion is not a reference list."""
    assert not looks_like_bibliography(_RELATED_WORK_LIST)


def test_prose_is_not_a_citation_entry() -> None:
    assert not is_citation_entry(_PROSE)
    assert citation_share(_PROSE) == 0.0


# --- trimming -----------------------------------------------------------------


def test_trailing_reference_run_is_trimmed_from_prose() -> None:
    """F4: the prose is kept; only the reference run is cut out."""
    pieces = trim_citation_runs(_PROSE + "\n\n" + _REFERENCE_LIST)
    assert [is_run for _, is_run in pieces] == [False, True]
    assert pieces[0][0].startswith("In conclusion")
    assert pieces[1][0].startswith("[1]")
    assert "[6]" in pieces[1][0]


def test_leading_reference_run_is_trimmed() -> None:
    pieces = trim_citation_runs(_REFERENCE_LIST + "\n\n" + _PROSE)
    assert [is_run for _, is_run in pieces] == [True, False]


def test_heading_before_a_trailing_run_goes_with_the_run() -> None:
    pieces = trim_citation_runs(_PROSE + "\n\n## References\n\n" + _REFERENCE_LIST)
    assert [is_run for _, is_run in pieces] == [False, True]
    assert pieces[1][0].startswith("## References")
    assert "References" not in pieces[0][0]


def test_short_runs_and_runs_inside_prose_are_left_alone() -> None:
    three = "\n".join(_REFERENCE_LIST.splitlines()[:6])
    assert trim_citation_runs(_PROSE + "\n\n" + three) == [
        (_PROSE + "\n\n" + three, False)
    ]
    middle = _PROSE + "\n\n" + _REFERENCE_LIST + "\n\n" + _PROSE
    assert [is_run for _, is_run in trim_citation_runs(middle)] == [False]


def test_a_run_beside_placeholders_only_is_not_split() -> None:
    """An image placeholder is not prose: it must not become a unit of its own."""
    text = _REFERENCE_LIST + "\n\n<!-- image -->"
    assert trim_citation_runs(text) == [(text, True)]


def test_a_unit_that_is_all_citations_is_not_split() -> None:
    assert trim_citation_runs(_REFERENCE_LIST) == [(_REFERENCE_LIST, True)]


# --- the routing rule ---------------------------------------------------------

_ACADEMIC = load_section_label_schema("academic")


# The strength each source gives when nothing else sets it (an inherited label
# is strong only through a numbered edge from a strong parent).
_STRONG_SOURCES = {
    SectionLabelSource.HEADING_PATTERN,
    SectionLabelSource.CONTENT_DENSITY,
    SectionLabelSource.CONTENT_BLOCK,
}


def _chunk(
    text: str,
    label: str | None = None,
    source: SectionLabelSource | None = None,
) -> PreparedChunk:
    strength = None
    if label is not None:
        strength = (
            LabelStrength.STRONG if source in _STRONG_SOURCES else LabelStrength.WEAK
        )
    return PreparedChunk(
        text=text,
        headings=None,
        section_label=label,
        section_label_source=source,
        section_label_strength=strength,
    )


def _route(
    chunk: PreparedChunk,
    *,
    schema_fallback: bool = False,
    exclude_sections: list[str] | None = None,
    bibliography_mode: str = "skip",
    non_content_mode: str = "extract",
    min_unit_chars: int = 0,
    schema=_ACADEMIC,
):
    config = ChunkConfig(
        bibliography_mode=cast(Any, bibliography_mode),
        non_content_mode=cast(Any, non_content_mode),
        min_unit_chars=min_unit_chars,
    )
    decision = route_unit(
        chunk,
        schema=schema,
        schema_fallback=schema_fallback,
        exclude_sections=exclude_sections,
        config=config,
    )
    return decision.route, decision.reason


@pytest.mark.parametrize(
    "source",
    [
        SectionLabelSource.HEADING_KEYWORD,
        SectionLabelSource.HEADING_INHERITED,
        SectionLabelSource.LLM,
        SectionLabelSource.FRONT_MATTER,
    ],
)
def test_weak_references_label_never_drops_prose(source) -> None:
    """F1, F2, F5, F10: a weak label alone is not evidence for a drop."""
    assert _route(_chunk(_PROSE, "references", source)) == (
        UnitRoute.EXTRACT,
        RouteReason.WEAK_LABEL,
    )


def test_strong_references_label_contradicted_by_content_is_extracted() -> None:
    chunk = _chunk(_PROSE, "references", SectionLabelSource.HEADING_PATTERN)
    assert _route(chunk) == (UnitRoute.EXTRACT, RouteReason.LABEL_CONTRADICTED)


def test_strong_references_label_over_a_reference_list_is_skipped() -> None:
    chunk = _chunk(_REFERENCE_LIST, "references", SectionLabelSource.HEADING_PATTERN)
    assert _route(chunk)[0] is UnitRoute.SKIP
    assert _route(chunk, bibliography_mode="citations_only") == (
        UnitRoute.CITATIONS,
        RouteReason.CITATION_LIST,
    )


def test_label_agreement_routes_a_list_with_a_preamble() -> None:
    """A strong label needs less content evidence than content alone."""
    text = "\n\n".join(
        [_PROSE, _PROSE.replace("In conclusion", "Moreover"), _REFERENCE_LIST]
    )
    assert not looks_like_bibliography(text)
    chunk = _chunk(text, "references", SectionLabelSource.HEADING_PATTERN)
    assert _route(chunk) == (UnitRoute.SKIP, RouteReason.CITATION_LABEL)


def test_unlabeled_reference_list_is_skipped_on_content_alone() -> None:
    assert _route(_chunk(_REFERENCE_LIST)) == (
        UnitRoute.SKIP,
        RouteReason.CITATION_LIST,
    )


def test_domain_facts_mode_disables_citation_routing() -> None:
    chunk = _chunk(_REFERENCE_LIST, "references", SectionLabelSource.HEADING_PATTERN)
    assert _route(chunk, bibliography_mode="domain_facts")[0] is UnitRoute.EXTRACT


def test_reference_table_is_extracted() -> None:
    assert _route(_chunk(_DOI_TABLE)) == (UnitRoute.EXTRACT, RouteReason.CONTENT)


def test_strong_acknowledgements_without_measurement_is_boilerplate() -> None:
    chunk = _chunk(
        _ACKNOWLEDGEMENTS, "acknowledgements", SectionLabelSource.HEADING_PATTERN
    )
    assert _route(chunk) == (UnitRoute.SKIP, RouteReason.BOILERPLATE)


def test_boilerplate_label_with_a_measurement_is_extracted() -> None:
    text = _ACKNOWLEDGEMENTS + " Samples were annealed at 100 °C for 10 min."
    chunk = _chunk(text, "acknowledgements", SectionLabelSource.HEADING_PATTERN)
    assert _route(chunk) == (UnitRoute.EXTRACT, RouteReason.LABEL_CONTRADICTED)


def test_keyword_boilerplate_label_is_extracted() -> None:
    """F7: "Funding Mechanisms" is a keyword match, not a boilerplate section."""
    text = "Grant schemes allocate funds by lottery among eligible proposals."
    chunk = _chunk(text, "acknowledgements", SectionLabelSource.HEADING_KEYWORD)
    assert _route(chunk) == (UnitRoute.EXTRACT, RouteReason.WEAK_LABEL)


def test_fallback_schema_applies_no_default_exclusions() -> None:
    """F9: a schema nobody chose cannot decide what is boilerplate."""
    chunk = _chunk(
        _ACKNOWLEDGEMENTS, "acknowledgements", SectionLabelSource.HEADING_PATTERN
    )
    assert _route(chunk, schema_fallback=True) == (
        UnitRoute.EXTRACT,
        RouteReason.FALLBACK_SCHEMA,
    )


def test_appendix_is_content() -> None:
    text = "Supplementary derivations: the rate constant is 0.02 per hour at 85 °C."
    chunk = _chunk(text, "appendix", SectionLabelSource.HEADING_PATTERN)
    assert _route(chunk) == (UnitRoute.EXTRACT, RouteReason.CONTENT)


def test_explicit_exclusion_needs_a_strong_label() -> None:
    strong = _chunk(_PROSE, "results", SectionLabelSource.HEADING_PATTERN)
    weak = _chunk(_PROSE, "results", SectionLabelSource.LLM)
    assert _route(strong, exclude_sections=["results"]) == (
        UnitRoute.SKIP,
        RouteReason.EXCLUDED_BY_REQUEST,
    )
    assert _route(weak, exclude_sections=["results"]) == (
        UnitRoute.EXTRACT,
        RouteReason.WEAK_LABEL,
    )


def test_explicit_exclusion_list_replaces_the_boilerplate_default() -> None:
    chunk = _chunk(
        _ACKNOWLEDGEMENTS, "acknowledgements", SectionLabelSource.HEADING_PATTERN
    )
    for request in ([], ["results"]):
        assert _route(chunk, exclude_sections=request) == (
            UnitRoute.EXTRACT,
            RouteReason.EXCLUSION_OVERRIDDEN,
        )


def test_undersized_units_are_skipped() -> None:
    assert _route(_chunk("Too short."), min_unit_chars=40) == (
        UnitRoute.SKIP,
        RouteReason.UNDERSIZED,
    )


def test_non_content_heuristic_flags_or_skips() -> None:
    orcid = (
        "J. Smith https://orcid.org/0000-0002-1825-0097 jsmith@example.org "
        "A. B. Jones https://orcid.org/0000-0001-5109-3700 ajones@example.org"
    )
    assert _route(_chunk(orcid)) == (UnitRoute.NON_CONTENT, RouteReason.NON_CONTENT)
    assert _route(_chunk(orcid), non_content_mode="skip") == (
        UnitRoute.SKIP,
        RouteReason.NON_CONTENT,
    )


def test_without_a_schema_only_content_rules_apply() -> None:
    chunk = _chunk(
        _ACKNOWLEDGEMENTS, "acknowledgements", SectionLabelSource.HEADING_PATTERN
    )
    assert _route(chunk, schema=None)[0] is UnitRoute.EXTRACT
    assert _route(_chunk(_REFERENCE_LIST), schema=None)[0] is UnitRoute.SKIP


# --- prepare end to end -------------------------------------------------------


def _tools(**config: Any) -> ToolBox:
    chunk_config = ChunkConfig.model_validate(
        {"min_size": 40, "max_size": 4000, **config}
    )

    async def no_llm(_prompt):
        raise AssertionError("LLM should not be called")

    return cast(
        ToolBox,
        SimpleNamespace(
            chunker=ChunkerTool(chunk_config=chunk_config),
            config=SimpleNamespace(
                chunk_config=chunk_config,
                server=SimpleNamespace(parallel_workers=2),
            ),
            llm=no_llm,
        ),
    )


def _prepare(markdown: str, options: PrepareOptions | None = None, **config: Any):
    tools = _tools(**config)
    return asyncio.run(
        prepare_content_units(
            doc_from_markdown_lines(markdown),
            tools.chunker,
            tools.chunker.config,
            options or PrepareOptions(document_type_hint="research article"),
            tools,
        )
    )


def _kept_text(document: PreparedDocument) -> str:
    return "\n".join(unit.text for unit in document.units)


def test_sections_after_references_are_kept() -> None:
    """F1: descriptive headings after References used to inherit it and vanish."""
    document = _prepare(
        "# Results\n\nFilms kept at 85 °C lost 20 % efficiency after 500 h.\n\n"
        f"# References\n\n{_REFERENCE_LIST}\n\n"
        "# Stability of CsPbBr3 films under humid air exposure\n\n"
        "Films stored at 60 % relative humidity retained 80 % efficiency after 1000 h.\n\n"
        "# Supplementary Note 2: XRD\n\n"
        "The PbI2 reflection at 12.7 degrees grows over 200 h of storage.\n"
    )
    kept = _kept_text(document)
    assert "retained 80 % efficiency" in kept
    assert "12.7 degrees" in kept
    assert "Nature Mater." not in kept
    assert document.dropped
    assert {unit.route_reason for unit in document.dropped} <= {
        RouteReason.CITATION_LABEL,
        RouteReason.CITATION_LIST,
    }


def test_conclusion_with_an_unheaded_reference_list_keeps_its_prose() -> None:
    """F4: the prose survives, the reference run alone is dropped."""
    document = _prepare(
        f"# Conclusion\n\n{_PROSE}\n\n{_REFERENCE_LIST}\n",
        section_classifier="heading",
    )
    kept = _kept_text(document)
    assert "retained 92 %" in kept
    assert "Nature Mater." not in kept
    assert len(document.dropped) == 1
    assert document.dropped[0].text.startswith("[1]")


def test_keyword_and_supporting_information_sections_are_kept() -> None:
    """F7: keyword labels no longer reach the default exclusions."""
    document = _prepare(
        "# Introduction\n\nThe study compares research grant schemes in Europe.\n\n"
        "# Funding Mechanisms\n\nGrant schemes allocate funds by lottery among "
        "eligible proposals and by peer review among the rest.\n\n"
        "# Supporting Information\n\nThe XRD patterns show a PbI2 reflection at "
        "12.7 degrees after 200 h at 85 °C.\n\n"
        f"# Acknowledgements\n\n{_ACKNOWLEDGEMENTS}\n"
    )
    kept = _kept_text(document)
    assert "lottery" in kept
    assert "12.7 degrees" in kept
    assert "beamline staff" not in kept
    assert [unit.route_reason for unit in document.dropped] == [RouteReason.BOILERPLATE]


def test_fallback_schema_keeps_boilerplate() -> None:
    document = _prepare(
        "# Overview\n\nThe valve assembly ships pre-calibrated.\n\n"
        f"# Acknowledgements\n\n{_ACKNOWLEDGEMENTS}\n",
        options=PrepareOptions(),
    )
    assert document.schema_source == "default"
    assert "beamline staff" in _kept_text(document)
    assert document.dropped == []


def test_summarize_sections_does_not_filter_extraction() -> None:
    """F12: summarize_sections chooses summaries, never what is extracted."""
    document = _prepare(
        "# Introduction\n\nWe survey prior work on perovskite aging.\n\n"
        "# Methods\n\nFilms were annealed at 100 °C for 10 min in nitrogen.\n",
        options=PrepareOptions(
            document_type_hint="research article", summarize_sections=["methods"]
        ),
    )
    assert {unit.section_label for unit in document.units} == {
        "introduction",
        "methods",
    }


def test_allowlist_reports_what_it_excluded() -> None:
    """F13: a strict allowlist records the unlabeled text it removed."""
    unheaded = "A title block with no heading and some unlabeled prose about films."
    document = _prepare(
        f"{unheaded}\n\n# Methods\n\nFilms were annealed at 100 °C for 10 min.\n\n"
        "# Results\n\nEfficiency fell by 20 % after 500 h at 85 °C.\n",
        options=PrepareOptions(
            document_type_hint="research article", target_sections=["methods"]
        ),
    )
    assert {unit.section_label for unit in document.units} == {"methods"}
    assert document.allowlist_excluded_chars > 0
    assert document.allowlist_excluded_unlabeled_chars >= len(unheaded) // 2


def test_every_unit_carries_a_route_and_reason() -> None:
    document = _prepare(
        "# Methods\n\nFilms were annealed at 100 °C for 10 min in nitrogen.\n"
    )
    assert document.units
    assert all(unit.route is UnitRoute.EXTRACT for unit in document.units)
    assert all(unit.route_reason is RouteReason.CONTENT for unit in document.units)


# --- the home-ontology excerpt -------------------------------------------------


def test_document_excerpt_skips_citation_and_non_content_units() -> None:
    from rdflib import URIRef

    from ontocast.onto.content_unit import ContentUnit
    from ontocast.onto.state import AgentState
    from ontocast.stategraph.helpers import build_document_excerpt

    doc_iri = URIRef("https://example.org/doc/1")

    def unit(
        index: int, text: str, *, citation: bool = False, non_content: bool = False
    ) -> ContentUnit:
        return ContentUnit(
            text=text,
            index=index,
            doc_iri=doc_iri,
            is_citation_metadata=citation,
            is_non_content=non_content,
        )

    state = AgentState()
    state.content_units = [
        unit(0, "Intro about perovskites."),
        unit(1, "Methods with annealing."),
        unit(2, "Results with efficiency."),
        unit(3, "Author block.", non_content=True),
        unit(4, _REFERENCE_LIST, citation=True),
    ]
    excerpt = build_document_excerpt(state)
    assert "Nature Mater." not in excerpt
    assert "Author block." not in excerpt
    assert "Results with efficiency." in excerpt


def test_unit_route_values_are_stable_wire_names() -> None:
    assert {route.value for route in UnitRoute} == {
        "extract",
        "citations",
        "non_content",
        "skip",
    }


def test_normative_references_are_a_citations_section() -> None:
    standard = load_section_label_schema("standard")
    chunk = _chunk(
        _REFERENCE_LIST, "normative_references", SectionLabelSource.HEADING_PATTERN
    )
    assert _route(chunk, schema=standard) == (UnitRoute.SKIP, RouteReason.CITATION_LIST)
    preamble = "The following documents are referred to in this text.\n\n"
    labelled = _chunk(
        preamble + "\n".join(_REFERENCE_LIST.splitlines()[:4]) + "\n\n" + _PROSE,
        "normative_references",
        SectionLabelSource.HEADING_PATTERN,
    )
    assert _route(labelled, schema=standard) == (
        UnitRoute.SKIP,
        RouteReason.CITATION_LABEL,
    )


def test_a_trimmed_run_keeps_its_units_citations_label() -> None:
    from ontocast.tool.chunk.prepare import _trim_citation_runs

    standard = load_section_label_schema("standard")
    unit = _chunk(
        _PROSE + "\n\n" + _REFERENCE_LIST,
        "normative_references",
        SectionLabelSource.HEADING_PATTERN,
    )
    other = _chunk(
        _PROSE + "\n\n" + _REFERENCE_LIST, "scope", SectionLabelSource.HEADING_PATTERN
    )
    kept, run = _trim_citation_runs([unit], standard)
    assert (kept.section_label, run.section_label) == (
        "normative_references",
        "normative_references",
    )
    _, other_run = _trim_citation_runs([other], standard)
    assert other_run.section_label == "normative_references"
    assert other_run.section_label_source is SectionLabelSource.CONTENT_BLOCK


def test_size_merging_never_joins_a_weak_vote_to_a_strong_section() -> None:
    from ontocast.tool.chunk.prepare import _merge_prepared_chunks

    strong = _chunk(_REFERENCE_LIST, "references", SectionLabelSource.HEADING_PATTERN)
    weak = _chunk(_PROSE, "references", SectionLabelSource.LLM)
    merged = _merge_prepared_chunks([strong, weak], min_size=10_000, max_size=20_000)
    assert [chunk.section_label_strength for chunk in merged] == [
        LabelStrength.STRONG,
        LabelStrength.WEAK,
    ]
    assert _route(merged[0]) == (UnitRoute.SKIP, RouteReason.CITATION_LIST)
    assert _route(merged[1])[0] is UnitRoute.EXTRACT
    pair = _merge_prepared_chunks([strong, strong], min_size=10_000, max_size=20_000)
    assert len(pair) == 1
