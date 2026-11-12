"""A statement names what it is about: the unit finding, the gap pass, the prompt.

An observation, or a value outside the property carrying it, is a fact only
with its subject. The unit validator reports one linked to nothing; the gap
pass is then asked for the link; the prompt states the rule and the term
sheet shows the properties that make it.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SOSA

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode, Status
from ontocast.onto.model import FactsUnitFinding, FactsUnitFindingKind, TripleFix
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.prefixes import PrefixTable
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.complete_facts import (
    build_subject_gaps_chapter,
    build_term_sheet,
)
from ontocast.prompt.facts_guidelines import (
    DEFAULT_QUANTITY_FALLBACK_VOCABULARY,
    format_facts_operational_guidelines,
)
from ontocast.prompt.term_sheet import build_ontology_term_sheet
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import FACTS_PHASE, _run_completion_passes
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import CriticPatchPolicy, ValidationPolicy
from ontocast.tool.facts_validation.terms import CatalogRoles, SubjectContract
from ontocast.tool.facts_validation.unit_findings import (
    collect_unit_findings,
    statement_without_subject_findings,
)

pytestmark = pytest.mark.unit

EX = "https://ex.org/lab#"
CD = DEFAULT_IRI
OBS, FILM, LONE = URIRef(CD + "obs"), URIRef(CD + "film"), URIRef(CD + "lone")
CONTRACT = SubjectContract(
    statement_classes=frozenset({str(SOSA.Observation)}),
    forward=frozenset({str(SOSA.hasFeatureOfInterest)}),
    inverse=frozenset({EX + "hasObservation"}),
)


def _facts(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(
        data=f"""@prefix cd: <{CD}> . @prefix ex: <{EX}> .
        @prefix sosa: <{SOSA}> . @prefix rdfs: <{RDFS}> .
        {body}""",
        format="turtle",
    )
    return graph


_LAPSED = """
cd:obs a sosa:Observation ; rdfs:label "thickness" ;
    sosa:hasResult [ ex:amount 5 ] .
cd:film a ex:Film ; rdfs:label "film" .
"""


# -- the finding -------------------------------------------------------------


def test_an_observation_linked_to_nothing_is_reported_with_candidates() -> None:
    findings = statement_without_subject_findings(_facts(_LAPSED), [CD], CONTRACT)
    assert [(f.kind, f.subject, f.mandatory) for f in findings] == [
        (FactsUnitFindingKind.STATEMENT_WITHOUT_SUBJECT, str(OBS), True)
    ]
    assert findings[0].suggestions == [str(FILM)]


@pytest.mark.parametrize(
    "link",
    [
        "cd:obs sosa:hasFeatureOfInterest cd:film .",
        "cd:film ex:hasObservation cd:obs .",
    ],
)
def test_an_observation_linked_in_either_direction_is_not_reported(link) -> None:
    graph = _facts(_LAPSED + link)
    assert statement_without_subject_findings(graph, [CD], CONTRACT) == []


def test_an_instance_stating_values_about_itself_is_not_reported() -> None:
    graph = _facts('cd:lone a ex:Film ; rdfs:label "film" ; ex:thickness 5 .')
    assert statement_without_subject_findings(graph, [CD], CONTRACT) == []


def test_an_observation_stating_nothing_is_not_this_finding() -> None:
    graph = _facts('cd:obs a sosa:Observation ; rdfs:label "an observation" .')
    assert statement_without_subject_findings(graph, [CD], CONTRACT) == []


def test_the_finding_is_not_raised_on_citation_units() -> None:
    def kinds(citation: bool) -> set[str]:
        return {
            str(f.kind)
            for f in collect_unit_findings(
                graph=_facts(_LAPSED),
                ontology_graph=RDFGraph(),
                quarantined=[],
                extraction_text="",
                fact_namespaces=[CD],
                coverage_limit=0,
                is_citation_metadata=citation,
                subject_contract=CONTRACT,
            )
        }

    assert "statement_without_subject" in kinds(False)
    assert "statement_without_subject" not in kinds(True)


def test_a_link_to_its_observed_property_is_no_subject() -> None:
    graph = _facts(
        _LAPSED
        + """
        cd:prop a ex:Property ; rdfs:label "thickness" .
        cd:obs sosa:observedProperty cd:prop ."""
    )
    findings = statement_without_subject_findings(graph, [CD], CONTRACT)
    assert [f.subject for f in findings] == [str(OBS)]


def test_the_contract_reads_classes_and_subject_properties_from_the_catalog() -> None:
    catalog = RDFGraph()
    reading, has_reading = URIRef(EX + "Reading"), URIRef(EX + "hasReading")
    catalog.add((reading, RDF.type, OWL.Class))
    catalog.add((reading, RDFS.subClassOf, SOSA.Observation))
    catalog.add((has_reading, RDFS.subPropertyOf, SOSA.isFeatureOfInterestOf))
    catalog.add((URIRef(EX + "about"), OWL.inverseOf, has_reading))
    contract = CatalogRoles.of([catalog]).subject_contract(frozenset())
    assert EX + "Reading" in contract.statement_classes
    assert EX + "hasReading" in contract.inverse
    assert EX + "about" in contract.forward


# -- the gap pass ---------------------------------------------------------------


def _state(graph: RDFGraph, text: str = "The film was examined.") -> UnitFactsState:
    unit = ContentUnit(
        text=text, index=0, doc_iri=URIRef("https://ex.org/doc/1"), graph=graph
    )
    snapshot = OntologySnapshot.from_graph(
        RDFGraph(),
        source_iris=["https://ex.org/onto"],
        assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
    )
    state = UnitFactsState(content_unit=unit, ontology_snapshot=snapshot)
    state.status = Status.SUCCESS
    return state


def _atomic() -> AtomicToolBox:
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            prefix_table=lambda: PrefixTable({}, {}),
            facts_completion_passes=1,
            facts_patch_policy=CriticPatchPolicy(),
            quantity_fallback_vocabulary=dict(DEFAULT_QUANTITY_FALLBACK_VOCABULARY),
            additional_standard_namespaces=(),
            validation_policy=ValidationPolicy(),
            acceptance_policy=None,
            numeric_coverage_limit=30,
            numeric_coverage_mandatory="off",
            catalog_terms=lambda: set(),
            catalog_roles=lambda: CatalogRoles(),
            subject_contract=lambda: CONTRACT,
        ),
    )


@pytest.mark.anyio
async def test_the_gap_pass_fires_on_a_subject_gap_alone_and_links_it(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def fake_complete(_state, _atomic, inventory, subject_gaps=()):
        seen["measurements"] = list(inventory.measurements)
        seen["gaps"] = [gap.subject for gap in subject_gaps]
        return [
            TripleFix(
                text_fragment="The film",
                action="ADD",
                severity="important",
                correct_value=f"<{OBS}> <{SOSA.hasFeatureOfInterest}> <{FILM}> .",
                explanation="subject",
            )
        ]

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    state = _state(_facts(_LAPSED))

    await _run_completion_passes(state, _atomic(), FACTS_PHASE, render_attempt=1)

    assert seen == {"measurements": [], "gaps": [str(OBS)]}
    assert (OBS, SOSA.hasFeatureOfInterest, FILM) in state.content_unit.graph
    assert state.attempt_log[-1].n_subjects_linked == 1
    assert not [
        f
        for f in state.deterministic_findings
        if f.kind == FactsUnitFindingKind.STATEMENT_WITHOUT_SUBJECT
    ]


@pytest.mark.anyio
async def test_the_gap_pass_makes_no_call_without_a_gap(monkeypatch) -> None:
    async def refuse(*_args, **_kwargs):  # pragma: no cover - never reached
        raise AssertionError("no gap, no call")

    monkeypatch.setattr(atomic_module, "complete_facts", refuse)
    state = _state(_facts(_LAPSED + "cd:obs sosa:hasFeatureOfInterest cd:film ."))

    await _run_completion_passes(state, _atomic(), FACTS_PHASE, render_attempt=1)

    assert state.attempt_log == []


@pytest.mark.anyio
async def test_a_fix_that_adds_a_mandatory_finding_is_rolled_back(
    monkeypatch,
) -> None:
    def findings(state, atomic):
        found = FACTS_PHASE.collect_findings(state, atomic)
        if (OBS, URIRef(EX + "bad"), Literal("x")) in state.content_unit.graph:
            found = [
                *found,
                FactsUnitFinding(
                    kind=FactsUnitFindingKind.UNKNOWN_TERM, message="unknown term"
                ),
            ]
        return found

    async def fake_complete(_state, _atomic, _inventory, _subject_gaps=()):
        return [
            TripleFix(
                text_fragment="x",
                action="ADD",
                severity="important",
                correct_value=f'<{OBS}> <{EX}bad> "x" .',
                explanation="wrong",
            )
        ]

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    state = _state(_facts(_LAPSED))

    await _run_completion_passes(
        state,
        _atomic(),
        replace(FACTS_PHASE, collect_findings=findings),
        render_attempt=1,
    )

    assert (OBS, URIRef(EX + "bad"), Literal("x")) not in state.content_unit.graph
    assert state.attempt_log[-1].n_fixes_rolled_back == 1


def test_the_gap_chapter_lists_statements_and_candidates() -> None:
    graph = _facts(_LAPSED)
    findings = statement_without_subject_findings(graph, [CD], CONTRACT)
    chapter = build_subject_gaps_chapter(graph, findings)
    assert "# SUBJECT GAPS" in chapter
    assert '"thickness"' in chapter and '"film"' in chapter
    assert build_subject_gaps_chapter(graph, []) == ""


def test_the_gap_term_sheet_lists_the_subject_properties() -> None:
    sheet = build_term_sheet(RDFGraph(), contract=CONTRACT)
    assert "hasFeatureOfInterest" in sheet and "hasObservation" in sheet
    assert "Properties a statement takes" not in sheet


def test_the_gap_term_sheet_shows_statement_properties_beside_a_statement_class() -> (
    None
):
    catalog = RDFGraph()
    catalog.bind("sosa", SOSA)
    catalog.add((SOSA.Observation, RDF.type, OWL.Class))
    sheet = build_term_sheet(catalog, contract=CONTRACT)
    assert 'sosa:hasResult  "has result"' in sheet


# -- the prompt -----------------------------------------------------------------


def test_the_render_prompt_states_the_rule_and_checks_it() -> None:
    text = format_facts_operational_guidelines(
        facts_namespace=CD, domain_ontologies_clause="", jsonld=False
    )
    assert "3c. EVERY STATEMENT NAMES ITS SUBJECT" in text
    checklist = text.split("FINAL STRUCTURAL VALIDATION CHECKLIST", 1)[1]
    assert "rule 3c" in checklist


def test_the_term_sheet_shows_statement_properties_beside_an_observation_class() -> (
    None
):
    catalog = RDFGraph()
    reading = URIRef(EX + "Reading")
    catalog.add((reading, RDF.type, OWL.Class))
    catalog.add((reading, RDFS.label, Literal("reading")))
    catalog.add((reading, RDFS.subClassOf, SOSA.Observation))
    assert "has feature of interest" in build_ontology_term_sheet(catalog)

    plain = RDFGraph()
    plain.add((URIRef(EX + "Film"), RDF.type, OWL.Class))
    assert "has feature of interest" not in build_ontology_term_sheet(plain)
