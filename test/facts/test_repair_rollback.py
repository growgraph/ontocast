"""A delete-only repair render is rolled back, not kept.

The findings prompt orders every MANDATORY item to be fixed by rewriting in
place. A model that answers it by deleting the offending statements resolves
nothing and destroys extracted data -- and the 2026-08 matsci runs measured that
happening at scale: of 58 cached repair responses carrying the false
``qudt:numericValue`` finding, 25 deleted valid values outright and 28
re-encoded scalars as equal-bound fake ranges.

The loop already detected the signature (graph shrank, mandatory count did not
fall) and kept the shrunken graph anyway, logging a warning. That made the
detector a bystander to the loss. It now restores the pre-repair graph, which by
construction has the same mandatory-finding count and strictly more data.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import Literal, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import _run_finding_driven_repair
from ontocast.tool.atomic import AtomicToolBox

pytestmark = pytest.mark.unit

_EX_PREDICATE = URIRef("http://example.org/redShiftContribution")
_LABEL = URIRef("http://www.w3.org/2000/01/rdf-schema#label")


def _unit_state_with_violation() -> UnitFactsState:
    """A unit carrying one mandatory UNKNOWN_TERM plus recoverable data."""
    graph = RDFGraph()
    subject = URIRef(f"{DEFAULT_IRI}sample_1")
    graph.add((subject, _EX_PREDICATE, URIRef(f"{DEFAULT_IRI}value_1")))
    graph.add((subject, _LABEL, Literal("sample")))
    graph.add((URIRef(f"{DEFAULT_IRI}value_1"), _LABEL, Literal("96 meV")))
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
            facts_llm_repair_visits=repair_visits,
            additional_standard_namespaces=(),
            validation_policy=None,
            acceptance_policy=None,
        ),
    )


@pytest.mark.anyio
async def test_delete_only_repair_is_rolled_back(monkeypatch) -> None:
    """Deleting the offending triple resolves the finding by destroying it."""

    async def deleting_render(state, tools, supplemental_ontologies=None):
        for triple in list(state.content_unit.graph.triples((None, None, None))):
            state.content_unit.graph.remove(triple)
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", deleting_render)
    state = _unit_state_with_violation()
    triples_before = len(state.content_unit.graph)

    result = await _run_finding_driven_repair(
        state, _atomic_tools(), [], render_attempt=1
    )

    assert len(result.content_unit.graph) == triples_before, (
        "a delete-only repair must leave the pre-repair graph in place"
    )
    assert (None, _LABEL, Literal("96 meV")) in result.content_unit.graph, (
        "the extracted value must survive a repair that only deleted"
    )
    repairs = [a for a in result.attempt_log if a.kind == "llm_repair"]
    assert len(repairs) == 1
    assert repairs[0].repair_delete_only is True
    assert repairs[0].triple_count == triples_before


@pytest.mark.anyio
async def test_a_rewrite_that_shrinks_the_graph_is_kept(
    monkeypatch,
) -> None:
    """The guard keys on whether anything was written back, not on size.

    A rewrite that collapses two statements into one legitimately shrinks the
    graph. It must not be mistaken for the delete-only pathology, or every real
    repair that removes a duplicate would be reverted.
    """

    async def collapsing_rewrite(state, tools, supplemental_ontologies=None):
        graph = state.content_unit.graph
        good = URIRef("https://schema.org/measurement")
        subject = URIRef(f"{DEFAULT_IRI}sample_1")
        for s, _p, o in list(graph.triples((None, _EX_PREDICATE, None))):
            graph.remove((s, _EX_PREDICATE, o))
            graph.add((s, good, o))
        # Also drop a duplicate label, so the net triple count falls while the
        # render still wrote the corrected statement.
        graph.remove((subject, _LABEL, Literal("sample")))
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", collapsing_rewrite)
    state = _unit_state_with_violation()
    triples_before = len(state.content_unit.graph)

    result = await _run_finding_driven_repair(
        state, _atomic_tools(), [], render_attempt=1
    )

    assert len(result.content_unit.graph) < triples_before
    repairs = [a for a in result.attempt_log if a.kind == "llm_repair"]
    assert repairs[0].repair_delete_only is False, (
        "the render wrote the corrected statement back, so this is a rewrite "
        "that happens to shrink the graph, not data destruction"
    )


@pytest.mark.anyio
async def test_deleting_the_flagged_triple_is_still_a_deletion(monkeypatch) -> None:
    """The case the previous detector structurally could not see.

    Removing the statement the finding points at makes the mandatory count
    *fall*, so a guard keyed on ``mandatory_after >= mandatory_before`` scores
    it as a successful repair. It is the dominant failure mode on record.
    """

    async def delete_the_flagged_triple(state, tools, supplemental_ontologies=None):
        graph = state.content_unit.graph
        for triple in list(graph.triples((None, _EX_PREDICATE, None))):
            graph.remove(triple)
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", delete_the_flagged_triple)
    state = _unit_state_with_violation()
    triples_before = len(state.content_unit.graph)

    result = await _run_finding_driven_repair(
        state, _atomic_tools(), [], render_attempt=1
    )

    assert len(result.content_unit.graph) == triples_before
    assert (None, _EX_PREDICATE, None) in result.content_unit.graph
    repairs = [a for a in result.attempt_log if a.kind == "llm_repair"]
    assert repairs[0].repair_delete_only is True


@pytest.mark.anyio
async def test_a_growing_repair_is_never_flagged(monkeypatch) -> None:
    """The guard must not fire on a repair that adds the corrected statement."""

    async def rewriting_render(state, tools, supplemental_ontologies=None):
        graph = state.content_unit.graph
        good = URIRef("https://schema.org/measurement")
        for s, _p, o in list(graph.triples((None, _EX_PREDICATE, None))):
            graph.remove((s, _EX_PREDICATE, o))
            graph.add((s, good, o))
            graph.add((s, _LABEL, Literal("rewritten")))
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", rewriting_render)
    state = _unit_state_with_violation()

    result = await _run_finding_driven_repair(
        state, _atomic_tools(), [], render_attempt=1
    )

    repairs = [a for a in result.attempt_log if a.kind == "llm_repair"]
    assert repairs[0].repair_delete_only is False
