"""Date, time and duration literals: strict lexical forms, ranges, quarantine.

A literal on a date property must be a value of its datatype as XSD defines
it. Text that is not one is set aside for the critic, from a render or from a
patch, and what is still set aside when the unit leaves its loop is put back
on its subject as a comment.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDFS, Literal, URIRef
from rdflib.namespace import PROV, XSD

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import RenderMode, Status
from ontocast.onto.model import FactsGateRepairKind, FactsUnitFindingKind
from ontocast.onto.prefixes import PrefixTable
from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState
from ontocast.onto.xsd_lexical import is_valid_temporal_lexical
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import _repair_facts_patch, facts_loop
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import (
    CriticPatchPolicy,
    literal_ranges,
    normalize_literals_against_schema,
    partition_literals_outside_temporal_ranges,
    unresolved_quarantine,
    write_back_temporal_quarantine,
)
from ontocast.tool.facts_validation.terms import CatalogRoles
from ontocast.tool.ontology_validation import CatalogSurfaceIndex
from ontocast.toolbox import ToolBox
from test.snapshot_helpers import empty_snapshot

pytestmark = pytest.mark.unit

EV = "https://ex.org/events#"
FACTS = "https://growgraph.dev/facts/"
REVIEW = URIRef(f"{FACTS}review")
ON = URIRef(f"{EV}on")
COUNT = URIRef(f"{EV}count")


def _ranges(**declared: URIRef) -> RDFGraph:
    graph = RDFGraph()
    for name, datatype in declared.items():
        graph.add((URIRef(EV + name), RDFS.range, datatype))
    return graph


# -- strict lexical forms ----------------------------------------------------


@pytest.mark.parametrize(
    ("lexical", "datatype", "valid"),
    [
        ("2021-06-29T00:00:00", XSD.dateTime, True),
        ("2021-06-29T12:00:00.5+02:00", XSD.dateTime, True),
        ("2021-06-29", XSD.dateTime, False),
        ("20210629", XSD.dateTime, False),
        ("2021-06-29 12:00", XSD.dateTime, False),
        ("2021-W26-2", XSD.dateTime, False),
        ("2021-06-29", XSD.date, True),
        ("2021-06-29T00:00:00", XSD.date, False),
        ("2021-02-30", XSD.date, False),
        ("2020-02-29", XSD.date, True),
        ("2021-02-29", XSD.date, False),
        ("--02-29", XSD.gMonthDay, True),
        ("2021-06", XSD.gYearMonth, True),
        ("21", XSD.gYear, False),
        ("P5Y", XSD.duration, True),
        ("P", XSD.duration, False),
        ("every 5 years", XSD.duration, False),
    ],
)
def test_temporal_lexical_forms_are_those_xsd_defines(
    lexical: str, datatype: URIRef, valid: bool
) -> None:
    assert is_valid_temporal_lexical(lexical, datatype) is valid


def test_other_datatypes_are_not_decided_here() -> None:
    assert is_valid_temporal_lexical("x", XSD.string) is None


def test_a_typed_literal_that_is_no_date_is_quarantined() -> None:
    graph = RDFGraph()
    graph.add((REVIEW, ON, Literal("June 2021", datatype=XSD.dateTime)))
    graph.add((REVIEW, ON, Literal("2021-06-29T00:00:00", datatype=XSD.dateTime)))
    clean, rejected = RDFGraph.partition_invalid_typed_literals(graph)
    assert [item.object_lexical for item in rejected] == ["June 2021"]
    assert len(clean) == 1


# -- retyping ----------------------------------------------------------------


def test_a_date_on_a_date_time_range_takes_the_form_a_typed_one_gets() -> None:
    ontology = _ranges(on=XSD.dateTime)
    graph = RDFGraph()
    graph.add((REVIEW, ON, Literal("2021-06-29")))
    graph.add((REVIEW, ON, Literal("every year")))
    assert normalize_literals_against_schema(graph, ontology) == 1
    assert (REVIEW, ON, Literal("2021-06-29T00:00:00", datatype=XSD.dateTime)) in graph
    clean, rejected = partition_literals_outside_temporal_ranges(graph, ontology)
    assert [item.object_lexical for item in rejected] == ["every year"]
    assert len(clean) == 1


def test_a_date_with_a_zone_is_not_widened_to_a_date_time() -> None:
    ontology = _ranges(on=XSD.dateTime)
    graph = RDFGraph()
    graph.add((REVIEW, ON, Literal("2021-06-29+02:00")))
    assert normalize_literals_against_schema(graph, ontology) == 0


def test_calendar_text_is_read_as_the_iso_value_it_names() -> None:
    ontology = _ranges(on=XSD.date, at=XSD.dateTime)
    at = URIRef(EV + "at")
    graph = RDFGraph()
    graph.add((REVIEW, ON, Literal("29 June 2021")))
    graph.add((REVIEW, at, Literal("June 2021")))
    assert normalize_literals_against_schema(graph, ontology) == 1
    assert (REVIEW, ON, Literal("2021-06-29", datatype=XSD.date)) in graph
    assert (REVIEW, at, Literal("June 2021")) in graph, "names no day: no date-time"


def test_a_date_rdflib_cannot_read_is_never_written_typed() -> None:
    ontology = _ranges(on=XSD.date)
    graph = RDFGraph()
    graph.add((REVIEW, ON, Literal("20210629")))
    assert normalize_literals_against_schema(graph, ontology) == 0


# -- where ranges come from ----------------------------------------------------


def test_a_range_the_snapshot_lacks_is_read_from_the_whole_catalog() -> None:
    roles = CatalogRoles.of([_ranges(on=XSD.dateTime)])
    graph = RDFGraph()
    graph.add((REVIEW, ON, Literal("every year")))
    _, rejected = partition_literals_outside_temporal_ranges(graph, RDFGraph(), roles)
    assert [item.object_lexical for item in rejected] == ["every year"]


def test_a_well_known_date_property_is_checked_without_any_catalog() -> None:
    graph = RDFGraph()
    graph.add((REVIEW, PROV.startedAtTime, Literal("every 5 years or more often")))
    _, rejected = partition_literals_outside_temporal_ranges(graph, None)
    assert [item.expected_datatype for item in rejected] == [str(XSD.dateTime)]


def test_a_property_declared_with_two_datatypes_is_not_checked() -> None:
    ontology = _ranges(on=XSD.dateTime)
    ontology.add((ON, RDFS.range, XSD.date))
    assert ON not in literal_ranges(ontology)


# -- resolution and write-back -------------------------------------------------


def _set_aside(text: str = "every 5 years") -> RejectedLiteralTriple:
    return RejectedLiteralTriple(
        subject=str(REVIEW),
        predicate=str(ON),
        object_lexical=text,
        reason="not a value of the property's datatype",
        expected_datatype=str(XSD.dateTime),
    )


def test_a_quarantined_text_is_resolved_by_a_valid_value_or_by_moving_it() -> None:
    item = _set_aside()
    graph = RDFGraph()
    graph.add((REVIEW, RDFS.label, Literal("review")))
    assert unresolved_quarantine(graph, [item]) == [item]
    graph.add((REVIEW, ON, Literal("June 2021", datatype=XSD.dateTime)))
    assert unresolved_quarantine(graph, [item]) == [item], "ill-formed: no fix"
    graph.add((REVIEW, ON, Literal("2021-06-29T00:00:00", datatype=XSD.dateTime)))
    assert unresolved_quarantine(graph, [item]) == []
    moved = RDFGraph()
    moved.add((REVIEW, RDFS.comment, Literal("every 5 years")))
    assert unresolved_quarantine(moved, [item]) == []


def test_a_string_on_an_object_property_is_resolved_by_an_iri() -> None:
    item = RejectedLiteralTriple(
        subject=str(REVIEW),
        predicate=str(ON),
        object_lexical="nm",
        reason="object property expects an IRI",
    )
    graph = RDFGraph()
    graph.add((REVIEW, ON, URIRef(FACTS + "nm")))
    assert unresolved_quarantine(graph, [item]) == []


def test_date_text_still_set_aside_is_written_back_as_a_comment() -> None:
    bad_number = RejectedLiteralTriple(
        subject=str(REVIEW),
        predicate=str(COUNT),
        object_lexical="10^-15",
        datatype=str(XSD.decimal),
    )
    typed = RejectedLiteralTriple(
        subject=str(REVIEW),
        predicate=str(ON),
        object_lexical="June 2021",
        datatype=str(XSD.dateTime),
    )
    graph = RDFGraph()
    graph.add((REVIEW, RDFS.label, Literal("review")))
    records = write_back_temporal_quarantine(graph, [_set_aside(), typed, bad_number])
    assert set(graph.objects(REVIEW, RDFS.comment)) == {
        Literal("every 5 years"),
        Literal("June 2021"),
    }
    assert [record.kind for record in records] == [
        FactsGateRepairKind.QUARANTINE_AS_COMMENT
    ] * 2
    assert write_back_temporal_quarantine(graph, [_set_aside()]) == []


# -- patches and the unit loop ---------------------------------------------------


def _unit_state() -> UnitFactsState:
    unit = ContentUnit(
        text="The plans are reviewed every 5 years or more often.",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=RDFGraph(),
    )
    return UnitFactsState(content_unit=unit, ontology_snapshot=empty_snapshot())


def _atomic(roles: CatalogRoles, critic_passes: int = 0) -> AtomicToolBox:
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            facts_critic_passes=critic_passes,
            facts_patch_policy=CriticPatchPolicy(),
            quantity_fallback_vocabulary=None,
            catalog_surface_index=CatalogSurfaceIndex,
            additional_standard_namespaces=(),
            validation_policy=None,
            acceptance_policy=None,
            numeric_coverage_limit=30,
            numeric_coverage_mandatory=False,
            facts_critic_min_triples=0,
            facts_completion_passes=0,
            catalog_terms=lambda: set(),
            catalog_roles=lambda: roles,
        ),
    )


def test_a_patch_gets_the_retype_and_the_date_check() -> None:
    state = _unit_state()
    graph = state.content_unit.graph
    graph.add((REVIEW, PROV.startedAtTime, Literal("every 5 years")))
    graph.add((REVIEW, COUNT, Literal("5")))
    roles = CatalogRoles.of([_ranges(count=XSD.decimal)])

    _repair_facts_patch(state, _atomic(roles))

    assert (REVIEW, PROV.startedAtTime, Literal("every 5 years")) not in graph
    assert [item.object_lexical for item in state.quarantined_literal_triples] == [
        "every 5 years"
    ]
    assert (REVIEW, COUNT, Literal("5", datatype=XSD.decimal)) in graph


def test_a_rollback_restores_the_quarantine_with_the_graph() -> None:
    state = _unit_state()
    token = state.snapshot_for_rollback()
    state.quarantined_literal_triples.append(_set_aside())
    state.content_unit.graph.add((REVIEW, RDFS.label, Literal("review")))
    state.restore(token)
    assert state.quarantined_literal_triples == []
    assert len(state.content_unit.graph) == 0


@pytest.mark.anyio
async def test_with_no_critic_the_text_leaves_the_loop_as_a_comment(
    monkeypatch,
) -> None:
    atomic = _atomic(CatalogRoles())

    async def render(state, tools, **kwargs):
        graph = state.content_unit.graph
        graph.add((REVIEW, RDFS.label, Literal("review of the plans")))
        graph.add((REVIEW, PROV.startedAtTime, Literal("every 5 years")))
        clean, rejected = partition_literals_outside_temporal_ranges(graph, None)
        state.content_unit.graph = clean
        state.quarantined_literal_triples = rejected
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", render)
    tools = cast(
        ToolBox,
        SimpleNamespace(
            prefix_table=lambda: PrefixTable({}, {}),
            get_atomic_tools=lambda: atomic,
        ),
    )
    result = await facts_loop(
        _unit_state(),
        tools,
        UnitLoopContext.from_agent_state(AgentState(render_mode=RenderMode.FACTS)),
        pre_resolved_context=UnitOntologyContext(
            snapshot=empty_snapshot(), writable_iris=[], confidence=1.0
        ),
    )

    graph = result.content_unit.graph
    assert (REVIEW, RDFS.comment, Literal("every 5 years")) in graph
    assert (REVIEW, PROV.startedAtTime, Literal("every 5 years")) not in graph
    assert [repair.kind for repair in result.applied_repairs] == [
        FactsGateRepairKind.QUARANTINE_AS_COMMENT
    ]
    assert not [
        finding
        for finding in result.deterministic_findings
        if finding.kind == FactsUnitFindingKind.QUARANTINED_LITERAL
    ]
