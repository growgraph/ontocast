"""A statement's role link held by a node that is not a statement, and empty values.

A result, subject or property link runs from a statement. Held by anything
else (a place with ``hasResult 5 m``), it states a value that names no
quantity. The unit validator reports it, the gap pass is asked to restate a
misplaced result as a statement about its holder, and the exit removes the
link. A value node carrying nothing but its type and label is removed at exit.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef
from rdflib.namespace import RDF, SOSA

from ontocast.check import check_facts
from ontocast.check.report import FactsCheckKind
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode, Status
from ontocast.onto.model import (
    FactsGateRepairKind,
    FactsUnitFinding,
    FactsUnitFindingKind,
    TripleFix,
)
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.prefixes import PrefixTable
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import FactScope
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.complete_facts import (
    build_misplaced_chapter,
    build_task_chapter,
    completion_guidelines,
)
from ontocast.prompt.facts_guidelines import DEFAULT_QUANTITY_FALLBACK_VOCABULARY
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import FACTS_PHASE, _run_completion_passes
from ontocast.stategraph.unit_exit import EXIT_REPAIR_KINDS, exit_unit
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import CriticPatchPolicy, ValidationPolicy
from ontocast.tool.facts_validation.mandate import GapMandate
from ontocast.tool.facts_validation.terms import CatalogRoles
from ontocast.tool.facts_validation.unit_findings import (
    empty_value_findings,
    role_outside_statement_findings,
)

pytestmark = pytest.mark.unit

CD = DEFAULT_IRI
EX = "https://ex.org/lab#"
QUDT = "http://qudt.org/schema/qudt/"
UNIT = "http://qudt.org/vocab/unit/"
PREFIXES = f"""@prefix cd: <{CD}> . @prefix ex: <{EX}> . @prefix sosa: <{SOSA}> .
@prefix qudt: <{QUDT}> . @prefix unit: <{UNIT}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <https://schema.org/> .
"""
CATALOG_TTL = """
sosa:Observation a owl:Class . sosa:FeatureOfInterest a owl:Class .
ex:Place a owl:Class . ex:Sample a owl:Class . qudt:QuantityValue a owl:Class .
sosa:hasResult a owl:ObjectProperty ; schema:domainIncludes sosa:Observation .
sosa:isResultOf a owl:ObjectProperty ; schema:rangeIncludes sosa:Observation .
sosa:hasFeatureOfInterest a owl:ObjectProperty ;
    schema:domainIncludes sosa:Observation .
ex:measuredValue a owl:ObjectProperty ; rdfs:subPropertyOf sosa:hasResult ;
    rdfs:domain ex:Sample .
"""
LOC, CITY, VALUE = URIRef(CD + "station"), URIRef(CD + "city"), URIRef(CD + "elev")


def _graph(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


CATALOG = _graph(CATALOG_TTL)
CONTRACT = CatalogRoles.of([CATALOG]).statement_contract(
    frozenset({QUDT + "QuantityValue"})
)


def _outside(body: str) -> list[FactsUnitFinding]:
    return role_outside_statement_findings(
        FactScope.of(_graph(body), [CD]), CONTRACT, CATALOG
    )


_MISPLACED = """
cd:station a ex:Place ; rdfs:label "station" ;
    sosa:hasResult [ a qudt:QuantityValue ; qudt:numericValue 5 ; qudt:unit unit:M ] .
"""


# -- the finding -------------------------------------------------------------


def test_a_result_held_by_a_place_is_reported() -> None:
    (finding,) = _outside(_MISPLACED)
    assert finding.kind == FactsUnitFindingKind.ROLE_OUTSIDE_STATEMENT
    assert finding.mandatory
    assert finding.facet == "result"
    assert finding.subject == str(LOC)
    assert finding.predicate == str(SOSA.hasResult)
    assert finding.value.startswith("_:")
    assert str(SOSA.Observation) in finding.suggestions


def test_a_result_held_by_an_observation_is_not_reported() -> None:
    assert not _outside(
        "cd:obs a sosa:Observation ; sosa:hasResult [ qudt:numericValue 5 ] ."
    )


def test_an_untyped_or_undecidable_holder_is_not_reported() -> None:
    assert not _outside("cd:x sosa:hasResult [ qudt:numericValue 5 ] .")
    assert not _outside("cd:x a ex:Unplaced ; sosa:hasResult [ qudt:numericValue 5 ] .")


def test_a_catalog_sub_property_is_judged_by_its_own_side() -> None:
    """A result link the catalog declares on its own class admits that class."""
    assert not _outside("cd:s a ex:Sample ; ex:measuredValue [ qudt:numericValue 5 ] .")
    (finding,) = _outside(
        "cd:s a ex:Place ; ex:measuredValue [ qudt:numericValue 5 ] ."
    )
    assert finding.facet == "result"


def test_an_inverse_link_is_judged_at_its_object() -> None:
    (finding,) = _outside(
        "cd:elev a qudt:QuantityValue ; qudt:numericValue 5 ; "
        "sosa:isResultOf cd:station . cd:station a ex:Place ."
    )
    assert finding.subject == str(VALUE)
    assert finding.value == str(LOC)


def test_a_subject_link_held_by_a_feature_is_reported_as_subject() -> None:
    (finding,) = _outside(
        "cd:region a sosa:FeatureOfInterest ; sosa:hasFeatureOfInterest cd:city ."
    )
    assert finding.facet == "subject"


# -- empty values ------------------------------------------------------------


def _empty(body: str) -> list[FactsUnitFinding]:
    return empty_value_findings(FactScope.of(_graph(body), [CD]), CONTRACT)


def test_a_value_with_only_a_label_is_empty() -> None:
    (finding,) = _empty(
        'cd:obs a sosa:Observation ; sosa:hasResult [ a qudt:QuantityValue ; rdfs:label "records" ] .'
    )
    assert finding.kind == FactsUnitFindingKind.EMPTY_VALUE
    assert finding.subject.startswith("_:")


@pytest.mark.parametrize(
    "component",
    ["qudt:numericValue 5", "qudt:unit unit:M", 'qudt:value "dry"'],
)
def test_a_value_stating_anything_is_not_empty(component: str) -> None:
    assert not _empty(
        f"cd:obs a sosa:Observation ; sosa:hasResult [ a qudt:QuantityValue ; {component} ] ."
    )


# -- the exit ----------------------------------------------------------------


def test_the_exit_removes_the_link_and_a_blank_value_only_it_reached() -> None:
    graph = _graph(_MISPLACED)
    findings = role_outside_statement_findings(
        FactScope.of(graph, [CD]), CONTRACT, CATALOG
    )
    result = exit_unit(graph, findings, [], CONTRACT)
    assert (LOC, SOSA.hasResult, None) not in graph
    assert not list(graph.subjects(RDF.type, URIRef(QUDT + "QuantityValue")))
    assert (LOC, RDF.type, URIRef(EX + "Place")) in graph
    (record,) = result.records
    assert record.kind == FactsGateRepairKind.ROLE_OUTSIDE_STATEMENT_DROPPED
    assert record.kind in EXIT_REPAIR_KINDS


def test_the_exit_keeps_a_restated_value() -> None:
    """A named value a statement also holds keeps that statement's link."""
    graph = _graph(
        "cd:station a ex:Place ; sosa:hasResult cd:elev . "
        "cd:elev a qudt:QuantityValue ; qudt:numericValue 5 . "
        "cd:obs a sosa:Observation ; sosa:hasFeatureOfInterest cd:station ; "
        "sosa:hasResult cd:elev ."
    )
    findings = role_outside_statement_findings(
        FactScope.of(graph, [CD]), CONTRACT, CATALOG
    )
    exit_unit(graph, findings, [], CONTRACT)
    assert (LOC, SOSA.hasResult, VALUE) not in graph
    assert (URIRef(CD + "obs"), SOSA.hasResult, VALUE) in graph
    assert (VALUE, URIRef(QUDT + "numericValue"), None) in graph


def test_the_exit_removes_an_empty_value_and_keeps_its_carrier() -> None:
    graph = _graph(
        'cd:obs a sosa:Observation ; rdfs:label "records" ; '
        'sosa:hasResult [ a qudt:QuantityValue ; rdfs:label "records" ] .'
    )
    findings = empty_value_findings(FactScope.of(graph, [CD]), CONTRACT)
    result = exit_unit(graph, findings, [], CONTRACT)
    obs = URIRef(CD + "obs")
    assert (obs, SOSA.hasResult, None) not in graph
    assert (obs, RDF.type, SOSA.Observation) in graph
    assert result.records[0].kind == FactsGateRepairKind.EMPTY_VALUE_DROPPED
    assert result.records[0].kind in EXIT_REPAIR_KINDS


# -- the gap pass ------------------------------------------------------------


def _mandate(graph: RDFGraph) -> GapMandate:
    findings = role_outside_statement_findings(
        FactScope.of(graph, [CD]), CONTRACT, CATALOG
    )
    return GapMandate.of(findings, [], GapMandate().inventory, CONTRACT, [CD])


def _add(turtle: str) -> TripleFix:
    return TripleFix(
        text_fragment="5 m",
        action="ADD",
        severity="minor",
        correct_value=turtle,
        explanation="restated",
    )


_RESTATED = (
    f"@prefix cd: <{CD}> . @prefix sosa: <{SOSA}> . @prefix qudt: <{QUDT}> . "
    f"@prefix unit: <{UNIT}> . @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> . "
    'cd:elevation_obs a sosa:Observation ; rdfs:label "station elevation" ; '
    "sosa:hasFeatureOfInterest cd:station ; sosa:observedProperty cd:elevation ; "
    "sosa:hasResult [ a qudt:QuantityValue ; qudt:numericValue {n} ; qudt:unit unit:M ] . "
    'cd:elevation a sosa:ObservableProperty ; rdfs:label "elevation" .'
)


def test_a_misplaced_result_calls_the_gap_pass() -> None:
    mandate = _mandate(_graph(_MISPLACED))
    assert mandate.calls
    assert len(mandate.misplaced) == 1


def test_a_misplaced_subject_link_is_not_restated() -> None:
    graph = _graph(
        "cd:region a sosa:FeatureOfInterest ; sosa:hasFeatureOfInterest cd:city ."
    )
    mandate = _mandate(graph)
    assert not mandate.misplaced
    assert not mandate.calls


def test_the_restatement_with_the_same_numbers_is_admitted() -> None:
    graph = _graph(_MISPLACED)
    mandate = _mandate(graph)
    assert mandate.admits(_add(_RESTATED.format(n=5)), graph) is None


def test_a_restatement_changing_a_number_is_refused() -> None:
    graph = _graph(_MISPLACED)
    mandate = _mandate(graph)
    assert mandate.admits(_add(_RESTATED.format(n=6)), graph) == "value_not_listed"


def test_a_named_misplaced_value_may_be_linked_from_a_new_statement() -> None:
    graph = _graph(
        "cd:station a ex:Place ; sosa:hasResult cd:elev . "
        "cd:elev a qudt:QuantityValue ; qudt:numericValue 5 ."
    )
    mandate = _mandate(graph)
    fix = _add(
        f"@prefix cd: <{CD}> . @prefix sosa: <{SOSA}> . "
        "cd:obs a sosa:Observation ; sosa:hasFeatureOfInterest cd:station ; "
        "sosa:hasResult cd:elev ."
    )
    assert mandate.admits(fix, graph) is None


def test_the_prompt_lists_the_value_and_asks_for_a_restatement() -> None:
    graph = _graph(_MISPLACED)
    mandate = _mandate(graph)
    chapter = build_misplaced_chapter(graph, mandate.misplaced_values(graph))
    assert "VALUES OUTSIDE A STATEMENT" in chapter
    assert "cd:station" in chapter
    assert "qudt:numericValue" in chapter and "unit:M" in chapter
    assert "VALUES OUTSIDE A STATEMENT" in completion_guidelines(mandate)
    assert "not a statement" in build_task_chapter(mandate)
    assert "VALUES OUTSIDE A STATEMENT" not in completion_guidelines(GapMandate())


def _unit_state(graph: RDFGraph) -> UnitFactsState:
    unit = ContentUnit(
        text="The station sits at 5 m.",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=graph,
    )
    snapshot = OntologySnapshot.from_graph(
        CATALOG,
        source_iris=["https://example.com/onto/lab"],
        assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
    )
    state = UnitFactsState(content_unit=unit, ontology_snapshot=snapshot)
    state.status = Status.SUCCESS
    return state


@pytest.mark.anyio
async def test_a_misplaced_result_is_asked_for_once(monkeypatch) -> None:
    """An insert cannot remove the link, so a second pass does not ask again."""
    graph = _graph(_MISPLACED)
    state = _unit_state(graph)
    misplaced = role_outside_statement_findings(
        FactScope.of(graph, [CD]), CONTRACT, CATALOG
    )
    calls = {"n": 0}

    async def fake_complete(_state, _atomic, _mandate):
        calls["n"] += 1
        return []

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    phase = replace(FACTS_PHASE, collect_findings=lambda _s, _a: list(misplaced))
    atomic = cast(
        AtomicToolBox,
        SimpleNamespace(
            prefix_table=lambda: PrefixTable({}, {}),
            facts_completion_passes=2,
            facts_completion_measurements=False,
            facts_completion_detached=False,
            facts_patch_policy=CriticPatchPolicy(),
            quantity_fallback_vocabulary=dict(DEFAULT_QUANTITY_FALLBACK_VOCABULARY),
            additional_standard_namespaces=(),
            validation_policy=ValidationPolicy(),
            acceptance_policy=None,
            numeric_coverage_limit=30,
            numeric_coverage_mandatory="off",
            catalog_terms=lambda: set(),
            catalog_roles=lambda: CatalogRoles(),
            statement_contract=lambda: CONTRACT,
        ),
    )

    await _run_completion_passes(state, atomic, phase, render_attempt=1)

    assert calls["n"] == 1


# -- the public check --------------------------------------------------------


def test_the_facts_check_reports_both() -> None:
    graph = _graph(
        _MISPLACED
        + 'cd:obs a sosa:Observation ; sosa:hasResult [ a qudt:QuantityValue ; rdfs:label "x" ] .'
    )
    report = check_facts(graph, CATALOG)
    kinds = {(f.kind, f.severity) for f in report.findings}
    assert (FactsCheckKind.ROLE_OUTSIDE_STATEMENT, "error") in kinds
    assert (FactsCheckKind.EMPTY_VALUE, "warning") in kinds
