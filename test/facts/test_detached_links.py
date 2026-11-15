"""The gap pass asks for a link for each instance the render linked to nothing."""

from dataclasses import replace

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SDO

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.model import FactsUnitFinding, TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.prompt.complete_facts import build_detached_chapter
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import FACTS_PHASE, _run_completion_passes
from ontocast.tool.facts_validation.terms import link_properties
from test.facts.test_complete_facts import _atomic, _unit_state

pytestmark = pytest.mark.unit

_EVENT = URIRef(f"{DEFAULT_IRI}championships_2020")
_SERIES = URIRef(f"{DEFAULT_IRI}series")


def _facts() -> RDFGraph:
    graph = RDFGraph()
    graph.bind("cd", DEFAULT_IRI)
    graph.bind("schema", str(SDO))
    graph.add((_EVENT, RDF.type, SDO.SportsEvent))
    graph.add((_EVENT, RDFS.label, Literal("National Championships 2020")))
    graph.add((_EVENT, SDO.eventStatus, Literal("cancelled")))
    graph.add((_SERIES, RDF.type, SDO.SportsEvent))
    graph.add((_SERIES, RDFS.label, Literal("championship series")))
    graph.add((_SERIES, SDO.location, URIRef(f"{DEFAULT_IRI}sydney")))
    graph.add((URIRef(f"{DEFAULT_IRI}sydney"), RDF.type, SDO.Place))
    return graph


def _no_findings(_state, _atomic) -> list[FactsUnitFinding]:
    return []


def _link(subject: URIRef, obj: URIRef) -> TripleFix:
    return TripleFix(
        text_fragment="part of the championship series",
        action="ADD",
        severity="minor",
        correct_value=f'{{"@id": "{subject}", "{SDO.superEvent}": {{"@id": "{obj}"}}}}',
        explanation="the link the text states",
    )


@pytest.mark.anyio
async def test_an_instance_linked_to_nothing_is_offered_and_its_link_counted(
    monkeypatch,
) -> None:
    state = _unit_state(
        text="The 2020 championships, part of the series.", graph=_facts()
    )
    seen: list[list[URIRef]] = []

    async def fake_complete(_state, _atomic, _inventory, _gaps=(), detached=()):
        seen.append(list(detached))
        return [_link(_EVENT, _SERIES)]

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    atomic = _atomic()
    atomic.facts_completion_detached = True  # type: ignore[attr-defined]
    atomic.facts_completion_measurements = False  # type: ignore[attr-defined]

    await _run_completion_passes(
        state,
        atomic,
        replace(FACTS_PHASE, collect_findings=_no_findings),
        render_attempt=1,
    )

    assert seen == [[_EVENT]]
    assert (_EVENT, SDO.superEvent, _SERIES) in state.content_unit.graph
    assert state.attempt_log[-1].n_detached_linked == 1


@pytest.mark.anyio
async def test_no_detached_trigger_when_it_is_off(monkeypatch) -> None:
    state = _unit_state(text="The 2020 championships.", graph=_facts())
    calls: list[int] = []

    async def fake_complete(*_args):
        calls.append(1)
        return []

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    atomic = _atomic()
    atomic.facts_completion_measurements = False  # type: ignore[attr-defined]

    await _run_completion_passes(
        state,
        atomic,
        replace(FACTS_PHASE, collect_findings=_no_findings),
        render_attempt=1,
    )

    assert calls == []


def test_candidate_properties_follow_the_side_the_instance_fits() -> None:
    facts = _facts()
    onto = RDFGraph()
    onto.add((SDO.SportsEvent, RDF.type, OWL.Class))
    onto.add((SDO.Place, RDF.type, OWL.Class))
    onto.add((SDO.superEvent, RDFS.domain, SDO.SportsEvent))
    onto.add((SDO.superEvent, RDFS.range, SDO.SportsEvent))
    onto.add((SDO.location, RDFS.domain, SDO.SportsEvent))
    onto.add((SDO.location, RDFS.range, SDO.Place))
    onto.add(
        (SDO.eventStatus, RDFS.range, URIRef("http://www.w3.org/2001/XMLSchema#string"))
    )

    assert link_properties(_EVENT, facts, onto) == [
        (SDO.location, "from"),
        (SDO.superEvent, "from"),
        (SDO.superEvent, "to"),
    ]

    chapter = build_detached_chapter(facts, [(_EVENT, [(SDO.superEvent, "from")])])
    assert (
        'cd:championships_2020  a schema:SportsEvent  rdfs:label "National Championships 2020"'
        in chapter
    )
    assert "candidate properties: schema:superEvent (from)" in chapter
