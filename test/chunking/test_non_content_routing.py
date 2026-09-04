"""Tests for non-content (front/back matter) detection and routing."""

import importlib
import logging
from types import SimpleNamespace
from typing import Literal, cast

import pytest

from ontocast.config import ChunkConfig
from ontocast.onto.docling_helpers import plain_text_to_docling_doc
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.non_content import (
    NON_CONTENT_TOKEN_SHARE,
    has_non_content_heading,
    is_non_content_unit,
    metadata_token_share,
)
from ontocast.tool.chunk.prepare import PreparedChunk
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

_ORCID_BLOCK = """## ORCID
John Doe: 0000-0001-2345-6789
Jane Roe: https://orcid.org/0000-0002-3456-789X
"""

_NOTES_BLOCK = """## Notes
The authors declare no competing financial interest.
"""

_AUTHOR_BLOCK = """## Authors
J. Smith,1 A. B. Jones,2 and C. Lee1
1 Department of Chemistry, University X; 2 Institute Y
Email: j.smith@x.edu, a.jones@y.org
"""

_DATA_AVAILABILITY = """## Data availability
The data supporting this study are available from the corresponding author
upon reasonable request.
"""

_LICENCE_NOTICE = (
    "This article is licensed under a Creative Commons Attribution 4.0 "
    "International License, which permits use, sharing, adaptation, "
    "distribution and reproduction in any medium. "
    "http://creativecommons.org/licenses/by/4.0/."
)

_GLYPH_AUTHOR_INFORMATION = """## ■ AUTHOR INFORMATION
## Corresponding Author
Jane Roe − Department of Chemistry; orcid.org/0000-0002-3456-789X;
Email: jane@x.edu
"""

_RESULTS_PROSE = """## Results
The photoluminescence peak shifted by 10-15 meV after aging for 4 days at
77 K, and the quality factor reached 1200 at 77 K. These changes are
consistent with the formation of larger emissive domains in the film bulk.
"""

_NOTES_WITH_MEASUREMENTS = """## Notes
Samples were stored at 25 °C for 30 days before measurement.
"""

_HEADER_FRAGMENT = "1234567890():,; <!-- image --> ## ARTICLE OPEN"

_LICENCE_DISCUSSION = (
    "We discuss Creative Commons licences as a model for data sharing in "
    "materials science, where reuse terms shape what gets published and how "
    "results are reused by later studies."
)

_REFERENCE_LIST = """
[1] J. Smith, A. Jones, Perovskite degradation under humidity, Nature Mater.
14, 193-198 (2015). doi:10.1038/nmat4150
[2] L. Chen et al., Photoluminescence of CsPbBr3 nanocrystals, Nano Lett. 15,
3692-3696 (2015). doi:10.1021/nl5048779
[3] M. Garcia, Stability of halide perovskites, Chem. Rev. 119, 3036 (2019).
doi:10.1021/acs.chemrev.8b00539
[4] K. Tanaka, Exciton dynamics in lead halide perovskites, Phys. Rev. B 92,
045414 (2015). doi:10.1103/PhysRevB.92.045414
[5] R. Novak, Encapsulation strategies for perovskite devices, Adv. Mater. 30,
1806702 (2018). doi:10.1002/adma.201806702
"""


@pytest.mark.parametrize(
    "text",
    [
        _ORCID_BLOCK,
        _NOTES_BLOCK,
        _AUTHOR_BLOCK,
        _DATA_AVAILABILITY,
        _LICENCE_NOTICE,
        _GLYPH_AUTHOR_INFORMATION,
    ],
    ids=["orcid", "notes", "authors", "data_availability", "licence", "glyph"],
)
def test_front_matter_units_detected(text: str) -> None:
    assert is_non_content_unit(text, None, None)


@pytest.mark.parametrize(
    "text",
    [_RESULTS_PROSE, _NOTES_WITH_MEASUREMENTS, _HEADER_FRAGMENT, _LICENCE_DISCUSSION],
    ids=["results", "notes_with_measurements", "header_fragment", "licence_prose"],
)
def test_content_units_kept(text: str) -> None:
    """A measurement anywhere keeps a unit; prose about licences is prose."""
    assert not is_non_content_unit(text, None, None)


def test_heading_matches_via_breadcrumb_when_text_has_no_heading_line() -> None:
    body = "Jane Roe and John Doe contributed equally to this work."
    assert not has_non_content_heading(body, None)
    assert has_non_content_heading(body, ["Results", "Author Contributions"])
    assert is_non_content_unit(body, ["Author Contributions"], None)


def test_heading_normalisation_strips_glyphs_and_numbering() -> None:
    assert has_non_content_heading("## ■ AUTHOR INFORMATION\nbody", None)
    assert has_non_content_heading("7. Conflicts of Interest\nbody", None)
    assert has_non_content_heading("**Publisher's Note**\nbody", None)
    assert not has_non_content_heading("## Results and Discussion\nbody", None)


def test_acknowledgements_label_counts_as_front_matter_heading() -> None:
    body = "We thank the beamline staff for their support during the campaign."
    assert is_non_content_unit(body, None, "acknowledgements")
    assert not is_non_content_unit(body, None, "results")


def test_metadata_token_share() -> None:
    dense = "J. Smith j.smith@x.edu 0000-0001-2345-6789 https://orcid.org/x A. B."
    assert metadata_token_share(dense) >= NON_CONTENT_TOKEN_SHARE
    assert metadata_token_share(_RESULTS_PROSE) < NON_CONTENT_TOKEN_SHARE
    assert metadata_token_share("") == 0.0
    # Punctuation-only tokens do not dilute the share.
    assert metadata_token_share("■ ## j@x.edu -->") == 1.0


def test_metadata_heavy_unit_with_a_measurement_is_kept() -> None:
    """A high metadata-token share must not override a real stated measurement.

    The heading and licence-boilerplate branches already check
    ``states_measurement`` before routing a unit as non-content; the
    metadata-token-share branch has to make the same check, or an author
    block that happens to also report a value (a device spec, a calibration
    temperature) is dropped alongside genuine front matter.
    """
    text = (
        "J. Smith j.smith@uni.edu https://orcid.org/0000-0002-1825-0097 "
        "A. B. Jones measured 300 K samples."
    )
    assert metadata_token_share(text) >= NON_CONTENT_TOKEN_SHARE
    assert not is_non_content_unit(text, None, None)


def test_empty_text_is_not_non_content() -> None:
    assert not is_non_content_unit("   ", None, None)


def test_non_content_mode_default_is_extract() -> None:
    assert ChunkConfig().non_content_mode == "extract"


# --- routing through chunk_text -----------------------------------------------


async def _run_chunk_text(
    mode: Literal["extract", "skip"],
    chunks: list[PreparedChunk],
    monkeypatch: pytest.MonkeyPatch,
    *,
    bibliography_mode: Literal["domain_facts", "citations_only", "skip"] = "skip",
) -> AgentState:
    """Drive chunk_text with prepared chunks, bypassing docling conversion."""
    chunk_text_module = importlib.import_module("ontocast.agent.chunk_text")

    async def fake_prepare(*_args, **_kwargs):
        return list(chunks)

    monkeypatch.setattr(chunk_text_module, "prepare_content_units", fake_prepare)
    tools = SimpleNamespace(
        chunker=SimpleNamespace(
            config=ChunkConfig(
                non_content_mode=mode, bibliography_mode=bibliography_mode
            )
        )
    )
    state = AgentState(docling_doc=plain_text_to_docling_doc("x", "doc"))
    state.current_domain = "https://example.org"
    state.doc_hid = "1"
    logging.getLogger("ontocast.agent.chunk_text").setLevel(logging.INFO)
    return await chunk_text_module.chunk_text(state, cast(ToolBox, tools))


def _chunk(text: str, label: str | None = None) -> PreparedChunk:
    return PreparedChunk(text=text, headings=None, section_label=label)


@pytest.mark.anyio
async def test_extract_keeps_unit_flagged_and_logs(monkeypatch, caplog) -> None:
    chunks = [_chunk(_RESULTS_PROSE, "results"), _chunk(_NOTES_BLOCK)]
    with caplog.at_level("INFO"):
        state = await _run_chunk_text("extract", chunks, monkeypatch)

    assert [unit.is_non_content for unit in state.content_units] == [False, True]
    assert [unit.index for unit in state.content_units] == [0, 1]
    assert "routed as non-content" in caplog.text
    assert "kept, flagged" in caplog.text
    assert state.non_content_units_skipped == 0


@pytest.mark.anyio
async def test_skip_drops_unit_renumbers_and_counts(monkeypatch, caplog) -> None:
    chunks = [
        _chunk(_ORCID_BLOCK),
        _chunk(_RESULTS_PROSE, "results"),
        _chunk(_NOTES_BLOCK),
        _chunk(_RESULTS_PROSE, "results"),
    ]
    with caplog.at_level("INFO"):
        state = await _run_chunk_text("skip", chunks, monkeypatch)

    assert len(state.content_units) == 2
    # Dropping chunks must not leave holes in the index sequence.
    assert [unit.index for unit in state.content_units] == [0, 1]
    assert not any(unit.is_non_content for unit in state.content_units)
    assert state.non_content_units_skipped == 2
    assert "Dropped 2 non-content chunk(s)" in caplog.text


@pytest.mark.anyio
async def test_bibliography_route_takes_precedence(monkeypatch) -> None:
    """A reference list is routed as bibliography, never as non-content."""
    chunks = [_chunk(_REFERENCE_LIST, "references"), _chunk(_RESULTS_PROSE)]
    state = await _run_chunk_text(
        "skip", chunks, monkeypatch, bibliography_mode="citations_only"
    )

    assert [unit.is_citation_metadata for unit in state.content_units] == [
        True,
        False,
    ]
    assert not any(unit.is_non_content for unit in state.content_units)
    assert state.bibliography_units_skipped == 0
    assert state.non_content_units_skipped == 0


@pytest.mark.anyio
async def test_routing_counters_reach_the_manifest_selection_block(
    monkeypatch, tmp_path
) -> None:
    from ontocast.api.process_helpers import _selection_manifest
    from ontocast.config import Config, PathConfig, ToolConfig

    chunks = [_chunk(_NOTES_BLOCK), _chunk(_REFERENCE_LIST, "references")]
    state = await _run_chunk_text("skip", chunks, monkeypatch)
    ontology_dir = tmp_path / "ontologies"
    ontology_dir.mkdir()
    config = Config(
        tool_config=ToolConfig(
            path_config=PathConfig(ontology_directory=ontology_dir),
            chunk_config=ChunkConfig(non_content_mode="skip"),
        )
    )

    selection = _selection_manifest(state, config)

    assert selection.non_content_mode == "skip"
    assert selection.non_content_units_skipped == 1
    assert selection.bibliography_units_skipped == 1
    assert selection.undersized_units_skipped == 0
