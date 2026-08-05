"""Deterministic repair loop tests for the per-unit facts loop.

At the default MAX_VISITS=1 the LLM critic never runs; the repair loop is
what guarantees machine-found violations (quarantined literals, unknown
terms) still get a bounded fix chance, with findings injected as MANDATORY
prompt items.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import Literal, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.model import FactsUnitFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import _run_deterministic_repair
from ontocast.tool.atomic import AtomicToolBox

_EX_PREDICATE = URIRef("http://example.org/redShiftContribution")
_GOOD_PREDICATE = URIRef("https://schema.org/measurement")


def _unit_state_with_violation() -> UnitFactsState:
    graph = RDFGraph()
    subject = URIRef(f"{DEFAULT_IRI}sample_1")
    graph.add((subject, _EX_PREDICATE, URIRef(f"{DEFAULT_IRI}value_1")))
    graph.add(
        (
            subject,
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
            Literal("sample"),
        )
    )
    unit = ContentUnit(
        text="a shift of 96 meV",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=graph,
    )
    state = UnitFactsState(content_unit=unit)
    state.status = Status.SUCCESS
    return state


def _atomic_tools(repair_visits: int = 1) -> AtomicToolBox:
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            facts_repair_visits=repair_visits,
            additional_standard_namespaces=(),
        ),
    )


@pytest.mark.anyio
async def test_repair_fires_and_findings_reach_render(monkeypatch) -> None:
    seen_findings: list = []

    async def fake_render(state, tools, supplemental_ontologies=None):
        seen_findings.append(list(state.deterministic_findings))
        # The "LLM" fixes the violation: replace the ex: predicate.
        for s, p, o in list(
            state.content_unit.graph.triples((None, _EX_PREDICATE, None))
        ):
            state.content_unit.graph.remove((s, p, o))
            state.content_unit.graph.add((s, _GOOD_PREDICATE, o))
        state.deterministic_findings = []
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", fake_render)
    state = _unit_state_with_violation()

    result = await _run_deterministic_repair(
        state, _atomic_tools(), [], render_attempt=1
    )

    assert len(seen_findings) == 1
    kinds = {finding.kind for finding in seen_findings[0]}
    assert FactsUnitFindingKind.UNKNOWN_TERM in kinds
    # Coverage finding for "96" rides along as advisory.
    assert FactsUnitFindingKind.NUMERIC_COVERAGE in kinds
    # Post-repair validation is clean except coverage (96 still missing).
    residual_kinds = {finding.kind for finding in result.deterministic_findings}
    assert FactsUnitFindingKind.UNKNOWN_TERM not in residual_kinds
    repair_attempts = [a for a in result.attempt_log if a.kind == "repair"]
    assert len(repair_attempts) == 1
    # The record carries the residual AFTER the repair render, not the
    # pre-render count the repair was asked to fix.
    assert repair_attempts[0].n_deterministic_findings == len(
        result.deterministic_findings
    )
    assert repair_attempts[0].n_deterministic_findings < len(seen_findings[0])
    assert repair_attempts[0].n_mandatory_findings == 0


@pytest.mark.anyio
async def test_repair_skipped_when_no_findings(monkeypatch) -> None:
    calls = {"render": 0}

    async def fake_render(state, tools, supplemental_ontologies=None):
        calls["render"] += 1
        return state

    monkeypatch.setattr(atomic_module, "render_facts", fake_render)
    graph = RDFGraph()
    graph.add(
        (
            URIRef(f"{DEFAULT_IRI}v"),
            URIRef("http://qudt.org/schema/qudt/numericValue"),
            Literal("96"),
        )
    )
    unit = ContentUnit(
        text="a shift of 96 meV",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=graph,
    )
    state = UnitFactsState(content_unit=unit)
    state.status = Status.SUCCESS

    result = await _run_deterministic_repair(
        state, _atomic_tools(), [], render_attempt=1
    )

    assert calls["render"] == 0
    assert result.attempt_log == []


@pytest.mark.anyio
async def test_failed_repair_render_keeps_graph_and_success(monkeypatch) -> None:
    async def fake_render(state, tools, supplemental_ontologies=None):
        state.status = Status.FAILED
        return state

    monkeypatch.setattr(atomic_module, "render_facts", fake_render)
    state = _unit_state_with_violation()
    triples_before = len(state.content_unit.graph)

    result = await _run_deterministic_repair(
        state, _atomic_tools(), [], render_attempt=1
    )

    assert result.status == Status.SUCCESS
    assert len(result.content_unit.graph) == triples_before


@pytest.mark.anyio
async def test_repair_budget_bounds_visits(monkeypatch) -> None:
    calls = {"render": 0}

    async def fake_render(state, tools, supplemental_ontologies=None):
        # Never fixes anything: findings persist each iteration.
        calls["render"] += 1
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", fake_render)
    state = _unit_state_with_violation()

    result = await _run_deterministic_repair(
        state, _atomic_tools(repair_visits=2), [], render_attempt=1
    )

    assert calls["render"] == 2
    assert any(
        finding.kind == FactsUnitFindingKind.UNKNOWN_TERM
        for finding in result.deterministic_findings
    )
    # A repair that fixed nothing records a non-zero mandatory residual.
    repair_attempts = [a for a in result.attempt_log if a.kind == "repair"]
    assert repair_attempts
    assert all(a.n_mandatory_findings > 0 for a in repair_attempts)
