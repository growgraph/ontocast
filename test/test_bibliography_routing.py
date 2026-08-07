"""Tests for bibliography detection and citation-metadata routing."""

from typing import Literal, cast

import pytest
from rdflib import URIRef

from ontocast.config import ChunkConfig
from ontocast.onto.constants import DEFAULT_DOMAIN
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.model import FactsUnitFindingKind
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState
from ontocast.stategraph.atomic import _collect_facts_findings
from ontocast.tool.chunk.bibliography import (
    is_bibliography_unit,
    looks_like_bibliography,
)
from ontocast.toolbox import ToolBox

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

_RESULTS_PROSE = """
The photoluminescence spectra were recorded after aging the films for 4-15
days at 10 C under vacuum. We observe a red shift of 10-15 meV for the thin
films and up to 96 meV for the thick films, with an excitation fluence of
230 uJ/cm2. The quality factor reached 1200 at 77 K, and the emission peak
narrowed from 85 meV to 42 meV over the aging period. These changes are
consistent with the formation of larger emissive domains in the film bulk,
as discussed in the context of halide migration and surface passivation.
"""


def test_reference_list_detected() -> None:
    assert looks_like_bibliography(_REFERENCE_LIST)


def test_results_prose_not_detected() -> None:
    assert not looks_like_bibliography(_RESULTS_PROSE)


def test_short_text_not_detected() -> None:
    assert not looks_like_bibliography("[1] one citation (2015). doi:10.1/x")


def test_section_label_overrides_heuristics() -> None:
    assert is_bibliography_unit(_RESULTS_PROSE, "References")
    assert is_bibliography_unit(_RESULTS_PROSE, "bibliography")
    assert not is_bibliography_unit(_RESULTS_PROSE, "results")
    assert not is_bibliography_unit(_RESULTS_PROSE, None)


def test_bibliography_mode_default_is_skip() -> None:
    assert ChunkConfig().bibliography_mode == "skip"


def _unit(text: str, *, citation: bool) -> ContentUnit:
    return ContentUnit(
        text=text,
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        is_citation_metadata=citation,
    )


def test_numeric_coverage_suppressed_for_citation_units() -> None:
    state = UnitFactsState(content_unit=_unit(_REFERENCE_LIST, citation=True))
    findings = _collect_facts_findings(state)
    assert not [
        finding
        for finding in findings
        if finding.kind == FactsUnitFindingKind.NUMERIC_COVERAGE
    ]

    prose_state = UnitFactsState(content_unit=_unit(_RESULTS_PROSE, citation=False))
    prose_findings = _collect_facts_findings(prose_state)
    assert [
        finding
        for finding in prose_findings
        if finding.kind == FactsUnitFindingKind.NUMERIC_COVERAGE
    ]


# --- routing through chunk_text (previously untested) ------------------------


async def _run_chunk_text(
    mode: Literal["domain_facts", "citations_only", "skip"], chunks, monkeypatch
):
    """Drive chunk_text with prepared chunks, bypassing docling conversion."""
    import importlib
    import logging
    from types import SimpleNamespace

    from ontocast.onto.docling_helpers import plain_text_to_docling_doc
    from ontocast.onto.state import AgentState

    # ontocast.agent re-exports the function under this name, so import the
    # module explicitly to patch its prepare_content_units dependency.
    chunk_text_module = importlib.import_module("ontocast.agent.chunk_text")

    async def fake_prepare(*_args, **_kwargs):
        return list(chunks)

    monkeypatch.setattr(chunk_text_module, "prepare_content_units", fake_prepare)
    tools = SimpleNamespace(
        chunker=SimpleNamespace(config=ChunkConfig(bibliography_mode=mode))
    )
    # doc_iri is a derived property; it follows from current_domain + doc_hid.
    state = AgentState(docling_doc=plain_text_to_docling_doc("x", "doc"))
    state.current_domain = "https://example.org"
    state.doc_hid = "1"
    logging.getLogger("ontocast.agent.chunk_text").setLevel(logging.INFO)
    return await chunk_text_module.chunk_text(state, cast(ToolBox, tools))


def _chunk(text: str, label: str | None = None):
    from ontocast.tool.chunk.prepare import PreparedChunk

    return PreparedChunk(text=text, headings=None, section_label=label)


@pytest.mark.anyio
async def test_citations_only_marks_unit_and_keeps_it(monkeypatch, caplog) -> None:
    chunks = [_chunk(_RESULTS_PROSE), _chunk(_REFERENCE_LIST, "references")]
    with caplog.at_level("INFO"):
        state = await _run_chunk_text("citations_only", chunks, monkeypatch)

    assert [unit.is_citation_metadata for unit in state.content_units] == [False, True]
    # Indices stay contiguous and the routing decision leaves a trace.
    assert [unit.index for unit in state.content_units] == [0, 1]
    assert "routed as bibliography" in caplog.text


@pytest.mark.anyio
async def test_skip_drops_unit_and_renumbers(monkeypatch) -> None:
    chunks = [
        _chunk(_REFERENCE_LIST, "references"),
        _chunk(_RESULTS_PROSE),
        _chunk(_RESULTS_PROSE),
    ]
    state = await _run_chunk_text("skip", chunks, monkeypatch)

    assert len(state.content_units) == 2
    # Dropping chunk 0 must not leave a hole in the index sequence.
    assert [unit.index for unit in state.content_units] == [0, 1]
    assert not any(unit.is_citation_metadata for unit in state.content_units)


@pytest.mark.anyio
async def test_domain_facts_disables_detection_entirely(monkeypatch) -> None:
    chunks = [_chunk(_REFERENCE_LIST, "references")]
    state = await _run_chunk_text("domain_facts", chunks, monkeypatch)

    assert len(state.content_units) == 1
    assert state.content_units[0].is_citation_metadata is False


def test_current_domain_constructor_argument_is_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom __init__ used to overwrite the caller's value from the env."""
    monkeypatch.setenv("CURRENT_DOMAIN", "https://from-environment.example")

    assert AgentState().current_domain == "https://from-environment.example"
    assert (
        AgentState(current_domain="https://explicit.example").current_domain
        == "https://explicit.example"
    )


def test_current_domain_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CURRENT_DOMAIN", raising=False)

    assert AgentState().current_domain == DEFAULT_DOMAIN
