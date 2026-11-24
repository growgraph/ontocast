"""The rules a value and a statement obey when a unit leaves its loop.

Each section is one rule: the gap pass adds only what it was asked for; a
literal is a value of its property's declared range; a closed vocabulary is
closed in every position; a statement carries the roles its class requires; a
value's numbers and individuals agree with the text; and the exit acts on
what is left. Test data is neutral: ``ex:`` items and generic properties, with
the well-known vocabularies as the registry ships them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS, SOSA, XSD

from ontocast.check import check_facts
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode, Status
from ontocast.onto.model import (
    FactsGateRepairKind,
    FactsRenderReport,
    FactsUnitFinding,
    FactsUnitFindingKind,
    FactsValidationFindingKind,
    TripleFix,
)
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.prefixes import PrefixTable
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.complete_facts import build_task_chapter, completion_guidelines
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import (
    FACTS_PHASE,
    _repair_facts_patch,
    _run_completion_passes,
)
from ontocast.stategraph.unit_exit import (
    ABSENCE_KINDS,
    EXIT_ACTIONS,
    EXIT_HANDLERS,
    NOT_FINDINGS,
    ExitAction,
    exit_unit,
)
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import (
    CriticPatchPolicy,
    ValidationPolicy,
    validate_aggregated_facts,
)
from ontocast.tool.facts_validation.literal_repair import (
    range_mismatches,
    unresolved_quarantine,
    write_back_quarantine,
)
from ontocast.tool.facts_validation.mandate import GapMandate
from ontocast.tool.facts_validation.statements import StatementContract
from ontocast.tool.facts_validation.terms import CatalogRoles, value_class_base
from ontocast.tool.facts_validation.unit_findings import (
    collect_unit_findings,
    missing_role_findings,
)
from ontocast.util.measurement_lexicon import Mention
from ontocast.util.numeric_inventory import (
    NumericInventory,
    stated_numbers,
    worded_numbers,
)
from test.snapshot_helpers import snapshot_from_ontology
from test.test_agent_facts import _build_ontology, _build_tools, render_facts_module

pytestmark = pytest.mark.unit

EX = "https://ex.org/v#"
CD = DEFAULT_IRI
QUDT = "http://qudt.org/schema/qudt/"
UNIT = "http://qudt.org/vocab/unit/"

_PREFIXES = f"""
@prefix ex:   <{EX}> .
@prefix cd:   <{CD}> .
@prefix sosa: <{SOSA}> .
@prefix qudt: <{QUDT}> .
@prefix unit: <{UNIT}> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
"""

#: A neutral catalog: a statement class, a value class with a decimal number
#: and a closed qualifier class whose individuals carry cue labels.
CATALOG = """
ex:Reading a owl:Class ; rdfs:subClassOf sosa:Observation .
ex:Change a owl:Class ; rdfs:subClassOf sosa:Actuation .
ex:Draw a owl:Class ; rdfs:subClassOf sosa:Sampling .
ex:Item a owl:Class .
ex:Amount a owl:Class ; rdfs:subClassOf qudt:QuantityValue .
ex:Qualifier a owl:Class .
ex:measures a owl:ObjectProperty ; rdfs:subPropertyOf sosa:observedProperty .
ex:number a owl:DatatypeProperty ; rdfs:range xsd:decimal .
ex:flag a owl:DatatypeProperty ; rdfs:range xsd:boolean .
ex:name a owl:DatatypeProperty ; rdfs:range xsd:string .
ex:qualifier a owl:ObjectProperty ; rdfs:range ex:Qualifier .
ex:result a owl:ObjectProperty .
ex:partOf a owl:ObjectProperty .
ex:around a ex:Qualifier ; rdfs:label "around" ; skos:altLabel "about" , "~" .
ex:atLeast a ex:Qualifier ; rdfs:label "at least" ; skos:altLabel "at least" .
ex:precise a ex:Qualifier ; rdfs:label "precise" ; skos:altLabel "exactly" .
"""


def _graph(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=_PREFIXES + body, format="turtle")
    return graph


def _catalog(body: str = CATALOG) -> RDFGraph:
    return _graph(body)


def _contract(catalog: RDFGraph | None = None) -> StatementContract:
    roles = CatalogRoles.of([catalog if catalog is not None else _catalog()])
    return roles.statement_contract(value_class_base(None, {}))


def _findings(
    graph: RDFGraph,
    text: str = "",
    *,
    catalog: RDFGraph | None = None,
    contract: StatementContract | None = None,
) -> list[FactsUnitFinding]:
    catalog = catalog if catalog is not None else _catalog()
    return collect_unit_findings(
        graph=graph,
        ontology_graph=catalog,
        quarantined=[],
        extraction_text=text,
        fact_namespaces=[CD],
        coverage_limit=0,
        policy=ValidationPolicy(vocabulary="open"),
        contract=contract if contract is not None else _contract(catalog),
    )


def _kinds(findings: list[FactsUnitFinding], kind: FactsUnitFindingKind) -> list:
    return [finding for finding in findings if finding.kind == kind]


# -- the gap pass adds only what it was asked for ---------------------------


_GAPPED = """
cd:r1 a ex:Reading ; rdfs:label "reading" ; ex:measures cd:size ;
    ex:result [ a ex:Amount ; rdfs:label "5 units" ; ex:number 5 ] .
cd:size a ex:Item ; rdfs:label "size" .
cd:item a ex:Item ; rdfs:label "item" .
cd:other a ex:Item ; rdfs:label "other item" .
"""


def _mandate(
    graph: RDFGraph,
    *,
    detached: tuple[URIRef, ...] = (),
    measurements: tuple[str, ...] = (),
) -> GapMandate:
    contract = _contract()
    return GapMandate.of(
        missing_role_findings(graph, contract, [CD]),
        list(detached),
        NumericInventory(
            measurements=[
                Mention(value=value, unit="u", start=0, end=0, context=value)
                for value in measurements
            ]
        ),
        contract,
        [CD],
    )


def _fix(turtle: str) -> TripleFix:
    return TripleFix(
        text_fragment="x",
        action="ADD",
        severity="minor",
        correct_value=_PREFIXES + turtle,
        explanation="proposed",
    )


def test_a_listed_role_link_is_admitted() -> None:
    graph = _graph(_GAPPED)
    mandate = _mandate(graph)

    assert [gap.facet for gap in mandate.role_gaps] == ["subject"]
    assert (
        mandate.admits(_fix("cd:r1 sosa:hasFeatureOfInterest cd:item ."), graph) is None
    )


def test_a_new_subject_a_link_introduces_takes_its_type_and_label() -> None:
    graph = _graph(_GAPPED)
    fix = _fix(
        'cd:r1 sosa:hasFeatureOfInterest cd:new . cd:new a ex:Item ; rdfs:label "new" .'
    )
    assert _mandate(graph).admits(fix, graph) is None


def test_a_value_on_a_carrier_outside_the_mandate_is_refused() -> None:
    graph = _graph(_GAPPED)
    fix = _fix("cd:item ex:result [ a ex:Amount ; ex:number 5 ] .")

    assert _mandate(graph).admits(fix, graph) == "value_not_listed"


def test_a_value_whose_number_is_listed_is_admitted() -> None:
    graph = _graph(_GAPPED)
    fix = _fix("cd:item ex:result [ a ex:Amount ; ex:number 7 ; rdfs:label '7 u' ] .")

    assert _mandate(graph, measurements=("7",)).admits(fix, graph) is None


def test_a_statement_beyond_the_listed_link_is_refused() -> None:
    graph = _graph(_GAPPED)
    fix = _fix("cd:r1 sosa:hasFeatureOfInterest cd:item . cd:item ex:partOf cd:other .")

    assert _mandate(graph).admits(fix, graph) == "outside_mandate"


def test_a_link_with_a_listed_detached_instance_is_admitted() -> None:
    graph = _graph(_GAPPED)
    mandate = _mandate(graph, detached=(URIRef(CD + "item"), URIRef(CD + "other")))

    assert mandate.admits(_fix("cd:item ex:partOf cd:other ."), graph) is None
    assert mandate.admits(_fix("cd:size ex:partOf cd:r1 ."), graph) == (
        "outside_mandate"
    )


@pytest.mark.anyio
async def test_refused_fixes_are_counted_by_reason(monkeypatch) -> None:
    graph = _graph(_GAPPED)
    state = UnitFactsState(
        content_unit=ContentUnit(
            text="A reading of 5 units for the item.",
            index=0,
            doc_iri=URIRef("https://ex.org/doc/1"),
            graph=graph,
        ),
        ontology_snapshot=OntologySnapshot.from_graph(
            _catalog(),
            source_iris=["https://ex.org/v"],
            assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
        ),
    )
    state.status = Status.SUCCESS
    contract = _contract()
    atomic = cast(
        AtomicToolBox,
        SimpleNamespace(
            prefix_table=lambda: PrefixTable({}, {}),
            facts_completion_passes=1,
            facts_completion_measurements=False,
            facts_completion_detached=False,
            facts_patch_policy=CriticPatchPolicy(),
            quantity_fallback_vocabulary=None,
            additional_standard_namespaces=(),
            validation_policy=ValidationPolicy(vocabulary="open"),
            acceptance_policy=None,
            numeric_coverage_limit=30,
            numeric_coverage_mandatory="off",
            catalog_terms=lambda: set(),
            catalog_roles=lambda: CatalogRoles(),
            statement_contract=lambda: contract,
        ),
    )

    async def fake_complete(_state, _atomic, mandate):
        assert [gap.subject for gap in mandate.role_gaps] == [CD + "r1"]
        return [
            _fix("cd:r1 sosa:hasFeatureOfInterest cd:item ."),
            _fix("cd:item ex:result [ a ex:Amount ; ex:number 5 ] ."),
            _fix("cd:item ex:partOf cd:other ."),
        ]

    monkeypatch.setattr(atomic_module, "complete_facts", fake_complete)
    await _run_completion_passes(state, atomic, FACTS_PHASE, render_attempt=1)

    attempt = state.attempt_log[-1]
    assert attempt.n_fixes_refused_by_reason == {
        "value_not_listed": 1,
        "outside_mandate": 1,
    }
    assert attempt.n_fixes_applied == 1
    assert attempt.n_roles_filled == 1
    assert (URIRef(CD + "r1"), SOSA.hasFeatureOfInterest, URIRef(CD + "item")) in (
        state.content_unit.graph
    )


def test_the_task_and_guidelines_ask_only_for_the_listed_gaps() -> None:
    graph = _graph(_GAPPED)
    links_only = _mandate(graph)

    task = build_task_chapter(links_only)
    assert "statements that lack a role" in task
    assert "measurements" not in task
    guidelines = completion_guidelines(links_only)
    assert "A recovered measurement" not in guidelines
    assert "recovers a listed measurement" not in guidelines

    with_values = _mandate(graph, measurements=("7",))
    assert "measurements the source text states" in build_task_chapter(with_values)
    assert "A recovered measurement" in completion_guidelines(with_values)


# -- a literal is a value of its property's declared range ------------------


def test_prose_on_a_declared_numeric_range_is_set_aside() -> None:
    graph = _graph(
        """
        cd:a ex:number "about five" ; ex:number "1e3" ; ex:number 7 ;
            ex:name "about five" ; ex:flag "yes" ; ex:flag "true" ."""
    )
    found = {record.object_lexical for _, record in range_mismatches(graph, _catalog())}

    assert found == {"about five", "yes"}


def test_the_well_known_numeric_union_needs_no_catalog_declaration() -> None:
    graph = _graph(
        """
        cd:a ex:result [ a qudt:QuantityValue ; qudt:numericValue "a few" ;
            qudt:numericValue "1e3" ; qudt:numericValue "2"^^xsd:integer ] ."""
    )
    (record,) = [record for _, record in range_mismatches(graph, None)]

    assert record.object_lexical == "a few"
    assert record.expected_datatypes == tuple(
        sorted(str(XSD[name]) for name in ("decimal", "double", "float", "integer"))
    )


def test_a_property_without_a_declared_range_is_never_judged() -> None:
    graph = _graph('cd:a ex:partOf "about five" ; ex:size "about five" .')
    assert range_mismatches(graph, _catalog()) == []


def test_a_patch_sets_prose_on_a_numeric_range_aside() -> None:
    state = UnitFactsState(
        content_unit=ContentUnit(
            text="about five",
            index=0,
            doc_iri=URIRef("https://ex.org/doc/1"),
            graph=_graph('cd:a ex:number "about five" ; ex:number "1e3" .'),
        ),
        ontology_snapshot=OntologySnapshot.from_graph(
            _catalog(),
            source_iris=["https://ex.org/v"],
            assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
        ),
    )
    atomic = cast(
        AtomicToolBox,
        SimpleNamespace(
            catalog_roles=lambda: CatalogRoles.of([_catalog()]),
            catalog_terms=lambda: set(),
            quantity_fallback_vocabulary=None,
        ),
    )

    _repair_facts_patch(state, atomic)

    graph = state.content_unit.graph
    assert (URIRef(CD + "a"), URIRef(EX + "number"), Literal("about five")) not in graph
    assert (URIRef(CD + "a"), URIRef(EX + "number"), Literal("1e3")) in graph
    assert [item.object_lexical for item in state.quarantined_literal_triples] == [
        "about five"
    ]


@pytest.mark.anyio
async def test_the_render_sets_prose_on_a_numeric_range_aside(monkeypatch) -> None:
    async def fake_call_llm_with_retry(**kwargs):
        return FactsRenderReport(
            semantic_graph=_graph(
                """
                cd:a a ex:Item ; ex:result [ a qudt:QuantityValue ;
                    rdfs:label "a few hundred" ; qudt:numericValue "a few hundred" ] ."""
            )
        )

    monkeypatch.setattr(
        render_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )
    state = UnitFactsState(
        content_unit=ContentUnit(
            text="a few hundred", index=0, doc_iri=URIRef("https://ex.org/doc/1")
        ),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )

    result = await render_facts_module.render_facts_fresh(state, tools=_build_tools())

    assert result.status == Status.SUCCESS
    assert not list(
        result.content_unit.graph.triples((None, URIRef(QUDT + "numericValue"), None))
    )
    assert [item.object_lexical for item in result.quarantined_literal_triples] == [
        "a few hundred"
    ]
    # The label already states the text: nothing is left to resolve.
    assert (
        unresolved_quarantine(
            result.content_unit.graph, result.quarantined_literal_triples
        )
        == []
    )


def test_check_facts_reports_prose_on_a_numeric_range_and_accepts_1e3() -> None:
    graph = _graph('cd:a a ex:Item ; ex:number "about five" ; ex:number "1e3" .')
    report = check_facts(graph, _catalog())
    invalid = [
        finding.value
        for finding in report.findings
        if finding.kind.value == "invalid_literal"
    ]

    assert invalid == ["about five"]


def test_a_literal_set_aside_is_written_back_only_when_no_annotation_holds_it() -> None:
    graph = _graph(
        """
        cd:a a ex:Item ; rdfs:label "item, about five" .
        cd:b a ex:Item ; ex:number "about six" ."""
    )
    found = range_mismatches(graph, _catalog())
    for triple, _ in found:
        graph.remove(triple)
    graph.add((URIRef(CD + "a"), URIRef(EX + "number"), Literal("about five")))
    found_a = range_mismatches(graph, _catalog())
    for triple, _ in found_a:
        graph.remove(triple)

    records, _ = write_back_quarantine(
        graph, [record for _, record in (*found, *found_a)]
    )

    assert [record.target for record in records] == [CD + "b"]
    assert (URIRef(CD + "b"), RDFS.comment, Literal("about six")) in graph


# -- a closed vocabulary is closed in every position ------------------------


_CASE_UNIT = """
cd:a a ex:Item ; rdfs:label "item" ;
    ex:result [ a qudt:QuantityValue ; rdfs:label "3 million" ;
        qudt:numericValue 3 ; qudt:unit unit:Million ] .
"""


def test_a_unit_differing_by_case_is_unknown_in_the_unit_loop_and_never_applied() -> (
    None
):
    graph = _graph(_CASE_UNIT)
    before = set(graph)
    (finding,) = [
        finding
        for finding in _findings(graph, "3 million")
        if finding.kind == FactsUnitFindingKind.UNKNOWN_TERM
    ]

    assert finding.predicate == UNIT + "Million"
    assert finding.suggestions[0] == UNIT + "MILLION"
    assert set(graph) == before


def test_a_unit_differing_by_case_is_unknown_at_the_gate() -> None:
    report = validate_aggregated_facts(
        _graph(_CASE_UNIT), _catalog(), fact_namespaces=[CD], contract=_contract()
    )
    (finding,) = [
        finding
        for finding in report.findings
        if finding.kind == FactsValidationFindingKind.UNKNOWN_TERM
    ]

    assert finding.predicate == UNIT + "Million"
    assert finding.severity == "error"
    assert finding.suggestions[0] == UNIT + "MILLION"


def test_a_unit_differing_by_case_is_unknown_in_check_facts() -> None:
    graph = _graph(_CASE_UNIT)
    before = set(graph)
    report = check_facts(graph, _catalog())
    (finding,) = [
        finding for finding in report.findings if finding.kind.value == "unknown_term"
    ]

    assert finding.predicate == UNIT + "Million"
    assert finding.suggestions[0] == UNIT + "MILLION"
    assert set(graph) == before


def test_an_invented_individual_of_a_value_is_caught_by_its_role() -> None:
    graph = _graph(
        """
        cd:a a ex:Item ; ex:partOf <https://open.example.net/u/furlong> ;
            ex:result [ a qudt:QuantityValue ; qudt:numericValue 3 ;
                qudt:unit <https://open.example.net/u/furlong> ;
                ex:qualifier <https://open.example.net/q/roughly> ] ."""
    )
    unknown = sorted(
        (finding.predicate, finding.facet)
        for finding in _findings(graph, "3")
        if finding.kind == FactsUnitFindingKind.UNKNOWN_TERM
    )

    assert unknown == [
        ("https://open.example.net/q/roughly", "closed_range_missing"),
        ("https://open.example.net/u/furlong", "closed_range_missing"),
    ]


def test_the_quantity_unit_property_is_a_term_of_its_vocabulary() -> None:
    graph = _graph(_CASE_UNIT.replace("unit:Million", "unit:MILLION"))
    assert not _kinds(_findings(graph, "3 million"), FactsUnitFindingKind.UNKNOWN_TERM)


# -- a statement carries the roles its class requires ---------------------


def _roles(body: str) -> list[tuple[str, str]]:
    return sorted(
        (finding.subject, finding.facet)
        for finding in missing_role_findings(_graph(body), _contract(), [CD])
    )


def test_an_observation_without_a_property_is_reported() -> None:
    assert _roles(
        "cd:r a ex:Reading ; sosa:hasFeatureOfInterest cd:x ; ex:result [ ex:number 5 ] ."
        " cd:x a ex:Item ."
    ) == [(CD + "r", "property")]


def test_a_catalog_sub_property_satisfies_the_role() -> None:
    assert (
        _roles(
            "cd:r a ex:Reading ; sosa:hasFeatureOfInterest cd:x ; ex:measures cd:p ;"
            " ex:result [ ex:number 5 ] . cd:x a ex:Item . cd:p a ex:Item ."
        )
        == []
    )


def test_a_link_serves_only_the_classes_its_domain_admits() -> None:
    body = (
        "cd:c a ex:Change ; sosa:hasFeatureOfInterest cd:x ; {link} cd:p ;"
        " ex:result [ ex:number 5 ] . cd:x a ex:Item . cd:p a ex:Item ."
    )
    assert _roles(body.format(link="sosa:observedProperty")) == [(CD + "c", "property")]
    assert _roles(body.format(link="sosa:actsOnProperty")) == []


def test_sampling_needs_only_a_subject() -> None:
    assert (
        _roles(
            "cd:d a ex:Draw ; sosa:hasFeatureOfInterest cd:x ; ex:result [ ex:number 5 ] ."
            " cd:x a ex:Item ."
        )
        == []
    )


def test_a_link_a_restriction_requires_is_a_role() -> None:
    catalog = _catalog(
        CATALOG
        + "ex:Reading rdfs:subClassOf [ a owl:Restriction ; owl:onProperty ex:partOf ;"
        " owl:someValuesFrom ex:Item ] ."
    )
    findings = missing_role_findings(
        _graph(
            "cd:r a ex:Reading ; sosa:hasFeatureOfInterest cd:x ; ex:measures cd:p ;"
            " ex:result [ ex:number 5 ] . cd:x a ex:Item . cd:p a ex:Item ."
        ),
        _contract(catalog),
        [CD],
    )

    assert [(f.facet, f.predicate) for f in findings] == [("link", EX + "partOf")]
    assert "partOf" in findings[0].message


# -- a value's numbers and individuals agree with the text -----------------


def _value(number: str, label: str, extra: str = "") -> RDFGraph:
    return _graph(
        f"""
        cd:a a ex:Item ; rdfs:label "item" ;
            ex:result [ a ex:Amount ; rdfs:label "{label}" ; ex:number {number} {extra} ] ."""
    )


def test_a_number_absent_from_label_and_text_is_ungrounded() -> None:
    graph = _value("4", "several")
    (finding,) = _kinds(
        _findings(graph, "The item lasted several rounds."),
        FactsUnitFindingKind.UNGROUNDED_NUMBER,
    )
    assert finding.predicate == EX + "number" and finding.value == "4"
    assert finding.mandatory


@pytest.mark.parametrize(
    ("number", "text"),
    [
        ('"1e3"', "about 1000 of them"),
        ("1000", "about 1,000 of them"),
        ("15", "between 10-15 of them"),
        ("10", "between 10–15 of them"),
        ("1400", "about 1.4 × 10^3 of them"),
    ],
)
def test_a_number_the_text_writes_is_grounded(number: str, text: str) -> None:
    graph = _value(number, "some")
    assert not _kinds(_findings(graph, text), FactsUnitFindingKind.UNGROUNDED_NUMBER)


@pytest.mark.parametrize(
    ("number", "label", "text"),
    [
        ("5", "five parts", "The item has Five  parts."),
        ("25", "twenty-five parts", "The item has twenty-five parts."),
        (
            "325000",
            "three hundred and twenty-five thousand parts",
            "It has three hundred and twenty-five thousand parts.",
        ),
        ("5", "fifth place", "The item finished fifth place."),
        ("2", "two-month window", "It opened a two-month window."),
        ("5", "cinq parties", "L'objet a cinq parties."),
    ],
)
def test_number_words_the_label_quotes_from_the_text_ground_the_number(
    number: str, label: str, text: str
) -> None:
    graph = _value(number, label)
    assert not _kinds(_findings(graph, text), FactsUnitFindingKind.UNGROUNDED_NUMBER)


@pytest.mark.parametrize(
    ("number", "label", "text"),
    [
        # the words are in the text, but the label does not quote them
        ("5", "parts", "The item has five parts."),
        # the label writes the number, but the text does not
        ("5", "5 parts", "The item has several parts."),
        # vague and fractional words are no numbers
        ("3", "several parts", "The item has several parts."),
        ("2.5", "two and a half years", "It lasted two and a half years."),
    ],
)
def test_a_number_no_quoted_mention_states_is_ungrounded(
    number: str, label: str, text: str
) -> None:
    graph = _value(number, label)
    assert _kinds(_findings(graph, text), FactsUnitFindingKind.UNGROUNDED_NUMBER)


def test_running_text_is_read_in_digits_only() -> None:
    # A word of one language can be a number in another ("once" is 11 in
    # Spanish): words are read only inside a value's quoted mention.
    assert stated_numbers("five parts, 2 more") == {"2"}
    assert "11" in worded_numbers("once")


def test_a_cue_of_another_individual_contradicts_the_qualifier() -> None:
    graph = _value("5", "about 5 units", "; ex:qualifier ex:precise")
    (finding,) = _kinds(
        _findings(graph, "The item weighs about 5 units."),
        FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL,
    )
    assert finding.value == EX + "precise"
    assert finding.suggestions == [EX + "around"]


def test_without_the_number_in_its_label_the_text_is_read() -> None:
    graph = _value("5", "weight", "; ex:qualifier ex:precise")
    assert _kinds(
        _findings(graph, "The item weighs about 5 units."),
        FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL,
    )


def test_a_label_writing_the_number_is_the_value_s_own_mention() -> None:
    """Elsewhere in the text the same number may belong to another value."""
    graph = _value("5", "5 units", "; ex:qualifier ex:precise")
    assert not _kinds(
        _findings(graph, "The item weighs 5 units; the other about 5 units."),
        FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL,
    )


def test_a_cue_of_the_qualifier_itself_agrees() -> None:
    graph = _value("50", "50", "; ex:qualifier ex:atLeast")
    assert not _kinds(
        _findings(graph, "The item holds at least 50 parts."),
        FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL,
    )


def test_a_value_naming_no_qualifier_is_never_given_one() -> None:
    graph = _value("5", "5")
    assert not _kinds(
        _findings(graph, "The item weighs about 5 units."),
        FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL,
    )


def test_a_catalog_without_cue_labels_gives_no_finding() -> None:
    catalog = _catalog(CATALOG.replace("skos:altLabel", "rdfs:comment"))
    graph = _value("5", "5", "; ex:qualifier ex:precise")
    assert not _kinds(
        _findings(graph, "The item weighs about 5 units.", catalog=catalog),
        FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL,
    )


# -- the exit acts on what the loop left ------------------------------------


def test_every_finding_kind_about_a_present_statement_has_an_exit_action() -> None:
    for kind in FactsUnitFindingKind:
        if kind in NOT_FINDINGS:
            continue
        assert kind in EXIT_ACTIONS, f"{kind} has no exit action"
        if kind not in ABSENCE_KINDS:
            assert EXIT_ACTIONS[kind] is not ExitAction.REPORT, kind
    assert {
        kind for kind, action in EXIT_ACTIONS.items() if action is ExitAction.REPORT
    } == ABSENCE_KINDS


def test_every_exit_action_but_report_has_a_handler() -> None:
    for action in ExitAction:
        if action is ExitAction.REPORT:
            assert action not in EXIT_HANDLERS
            continue
        assert action in EXIT_HANDLERS, f"{action} has no handler"
    for kind, action in EXIT_ACTIONS.items():
        assert action is ExitAction.REPORT or action in EXIT_HANDLERS, kind


def test_the_exit_runs_what_the_table_says(monkeypatch) -> None:
    called: list[tuple[ExitAction, list[str]]] = []
    for action in EXIT_HANDLERS:
        monkeypatch.setitem(
            EXIT_HANDLERS,
            action,
            lambda state, findings, action=action: called.append(
                (action, [f.kind.value for f in findings])
            ),
        )
    monkeypatch.setitem(
        EXIT_ACTIONS, FactsUnitFindingKind.UNGROUNDED_NUMBER, ExitAction.DROP_TRIPLE
    )
    finding = FactsUnitFinding(
        kind=FactsUnitFindingKind.UNGROUNDED_NUMBER, message="m", subject=CD + "v"
    )
    exit_unit(_graph(""), [finding], [], _contract())

    assert (ExitAction.DROP_TRIPLE, ["ungrounded_number"]) in called
    assert (ExitAction.DROP_NUMBER, []) in called


def _exit(graph: RDFGraph, text: str, catalog: RDFGraph | None = None):
    findings = _findings(graph, text, catalog=catalog)
    result = exit_unit(graph, findings, [], _contract(catalog))
    return findings, result


def test_the_exit_drops_an_unknown_closed_vocabulary_term() -> None:
    graph = _graph(_CASE_UNIT)
    _, result = _exit(graph, "3 million")

    assert not list(graph.triples((None, URIRef(QUDT + "unit"), None)))
    assert list(graph.triples((None, URIRef(QUDT + "numericValue"), None)))
    assert [record.kind for record in result.records] == [
        FactsGateRepairKind.UNKNOWN_TERM_DROPPED
    ]


def test_the_exit_drops_an_ungrounded_number_and_keeps_the_label() -> None:
    graph = _value("4", "several", "; ex:number 7")
    _, result = _exit(graph, "Several rounds, 7 of them long.")

    value = graph.value(URIRef(CD + "a"), URIRef(EX + "result"))
    assert isinstance(value, BNode)
    assert set(graph.objects(value, URIRef(EX + "number"))) == {Literal(7)}
    assert (value, RDFS.label, Literal("several")) in graph
    assert [record.kind for record in result.records] == [
        FactsGateRepairKind.UNGROUNDED_NUMBER_DROPPED
    ]


def test_a_value_left_with_no_number_goes_and_its_carrier_stays() -> None:
    graph = _value("4", "several")
    _, result = _exit(graph, "Several rounds.")

    assert graph.value(URIRef(CD + "a"), URIRef(EX + "result")) is None
    assert (URIRef(CD + "a"), RDFS.label, Literal("item")) in graph
    assert not list(graph.triples((None, URIRef(EX + "number"), None)))
    assert [record.kind for record in result.records] == [
        FactsGateRepairKind.UNGROUNDED_NUMBER_DROPPED,
        FactsGateRepairKind.VALUE_WITHOUT_NUMBER_DROPPED,
    ]
    assert result.records[-1].target == CD + "a"


def test_the_exit_removes_a_contradicted_qualifier_and_reports_it() -> None:
    graph = _value("5", "about 5 units", "; ex:qualifier ex:precise")
    _, result = _exit(graph, "The item weighs about 5 units.")

    assert not list(graph.triples((None, URIRef(EX + "qualifier"), None)))
    assert [(record.kind, record.target) for record in result.records] == [
        (FactsGateRepairKind.CONTRADICTED_INDIVIDUAL_DROPPED, EX + "precise")
    ]


def test_a_missing_role_is_carried_not_removed() -> None:
    graph = _graph(
        "cd:r a ex:Reading ; rdfs:label 'reading' ; ex:measures cd:p ;"
        " ex:result [ a ex:Amount ; rdfs:label '5' ; ex:number 5 ] . cd:p a ex:Item ."
    )
    before = set(graph)
    findings, result = _exit(graph, "a reading of 5")

    assert _kinds(findings, FactsUnitFindingKind.MISSING_ROLE)
    assert result.records == []
    assert set(graph) == before


@pytest.mark.anyio
async def test_a_unit_leaving_the_loop_applies_the_exit(monkeypatch) -> None:
    from ontocast.onto.enum import RenderMode
    from ontocast.onto.state import AgentState
    from ontocast.stategraph.atomic import facts_loop
    from ontocast.stategraph.context_resolver import UnitOntologyContext
    from ontocast.stategraph.unit_context import UnitLoopContext
    from ontocast.tool.ontology_validation import CatalogSurfaceIndex
    from ontocast.toolbox import ToolBox

    catalog = _catalog()
    contract = _contract()
    atomic = cast(
        AtomicToolBox,
        SimpleNamespace(
            facts_critic_passes=0,
            facts_patch_policy=CriticPatchPolicy(),
            quantity_fallback_vocabulary=None,
            catalog_surface_index=CatalogSurfaceIndex,
            additional_standard_namespaces=(),
            validation_policy=ValidationPolicy(vocabulary="open"),
            acceptance_policy=None,
            numeric_coverage_limit=0,
            numeric_coverage_mandatory=False,
            facts_critic_min_triples=0,
            facts_completion_passes=0,
            catalog_terms=lambda: set(),
            catalog_roles=lambda: CatalogRoles.of([catalog]),
            statement_contract=lambda: contract,
        ),
    )

    async def render(state, tools, **kwargs):
        state.content_unit.graph = _graph(
            _CASE_UNIT + 'cd:b a ex:Item ; ex:result [ a ex:Amount ; rdfs:label "x" ;'
            " ex:number 9 ] ."
        )
        state.status = Status.SUCCESS
        return state

    snapshot = OntologySnapshot.from_graph(
        catalog,
        source_iris=[],
        assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
    )
    monkeypatch.setattr(atomic_module, "render_facts", render)
    state = UnitFactsState(
        content_unit=ContentUnit(
            text="3 million items.", index=0, doc_iri=URIRef("https://ex.org/doc/1")
        ),
        ontology_snapshot=snapshot,
    )
    result = await facts_loop(
        state,
        cast(
            ToolBox,
            SimpleNamespace(
                prefix_table=lambda: PrefixTable({}, {}),
                get_atomic_tools=lambda: atomic,
            ),
        ),
        UnitLoopContext.from_agent_state(AgentState(render_mode=RenderMode.FACTS)),
        pre_resolved_context=UnitOntologyContext(
            snapshot=snapshot, writable_iris=[], confidence=1.0
        ),
    )

    graph = result.content_unit.graph
    assert not list(graph.triples((None, URIRef(QUDT + "unit"), None)))
    assert graph.value(URIRef(CD + "b"), URIRef(EX + "result")) is None
    assert sorted(str(record.kind) for record in result.applied_repairs) == [
        "ungrounded_number_dropped",
        "unknown_term_dropped",
        "value_without_number_dropped",
    ]
    assert not [
        finding
        for finding in result.deterministic_findings
        if finding.kind
        in (FactsUnitFindingKind.UNKNOWN_TERM, FactsUnitFindingKind.UNGROUNDED_NUMBER)
    ]


# -- domain balance: no values, no new finding ----------------------------------


def test_plain_entity_relations_get_no_new_finding_and_lose_nothing() -> None:
    """A catalog with no quantity vocabulary, numbers without any unit."""
    catalog = _graph(
        """
        ex:Item a owl:Class .
        ex:partOf a owl:ObjectProperty .
        ex:count a owl:DatatypeProperty .
        ex:year a owl:DatatypeProperty .
        ex:note a owl:DatatypeProperty ."""
    )
    graph = _graph(
        """
        cd:a a ex:Item ; rdfs:label "item 3" ; ex:partOf cd:b ;
            ex:count 3 ; ex:year 2021 ; ex:note "about five" .
        cd:b a ex:Item ; rdfs:label "item 4" ; ex:count 4 ."""
    )
    before = set(graph)
    contract = _contract(catalog)
    new_kinds = {
        FactsUnitFindingKind.UNGROUNDED_NUMBER,
        FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL,
        FactsUnitFindingKind.MISSING_ROLE,
        FactsUnitFindingKind.UNKNOWN_TERM,
        FactsUnitFindingKind.QUARANTINED_LITERAL,
    }

    findings = _findings(
        graph, "Item 3 of 2021 has 3 parts.", catalog=catalog, contract=contract
    )

    assert not [finding for finding in findings if finding.kind in new_kinds]
    assert range_mismatches(graph, catalog, CatalogRoles.of([catalog])) == []
    assert exit_unit(graph, findings, [], contract).records == []
    assert set(graph) == before


def test_plain_entity_relations_call_no_gap_pass() -> None:
    """Without a statement or a measurement there is no gap to call for."""
    graph = _graph(
        'cd:a a ex:Item ; rdfs:label "item 3" ; ex:partOf cd:b . cd:b a ex:Item .'
    )
    contract = _contract()
    mandate = GapMandate.of(
        missing_role_findings(graph, contract, [CD]),
        [],
        NumericInventory(),
        contract,
        [CD],
    )
    assert not mandate.calls


# -- numbers are read as the text writes them ------------------------------------


@pytest.mark.parametrize(
    ("number", "text"),
    [
        ("5", "The item weighs 5kg."),
        ("3.5", "The item weighs 3.5kg."),
        ("100", "It moved at 100km/h."),
        ("20", "It lasted 20s."),
        ("12", "It ran 12h."),
        ("1.5", "A cell of 1.5V."),
        ("5", "It cost $5m."),
        ("5", "An area of 5m²."),
        ('"0.0015"^^xsd:decimal', "about 1.5 × 10−3 of them"),
        ('"1234.5"^^xsd:decimal', "about 1.234,5 of them"),
        ("1000", "about 1'000 of them"),
    ],
)
def test_a_number_written_against_its_unit_or_grouped_is_grounded(
    number: str, text: str
) -> None:
    graph = _value(number, "value")
    findings, result = _exit(graph, text)

    assert not _kinds(findings, FactsUnitFindingKind.UNGROUNDED_NUMBER)
    assert result.records == []
    assert graph.value(URIRef(CD + "a"), URIRef(EX + "result")) is not None


@pytest.mark.parametrize(
    ("number", "text"),
    [
        ("5000000", "about 5 million of them"),
        ('"0.05"^^xsd:decimal', "a share of 5%"),
        ("2500", "a mass of 2.5 thousand units"),
    ],
)
def test_a_number_rescaled_by_a_power_of_ten_is_grounded(
    number: str, text: str
) -> None:
    graph = _value(number, "value")
    assert not _kinds(_findings(graph, text), FactsUnitFindingKind.UNGROUNDED_NUMBER)


@pytest.mark.parametrize(
    ("number", "text"),
    [
        ('"0.5"^^xsd:decimal', "1/2 of them"),
        ('"5.08"^^xsd:decimal', "a length of 2 inches"),
        ("15", "only 5 of them"),
    ],
)
def test_a_fraction_or_a_conversion_is_not_grounded(number: str, text: str) -> None:
    graph = _value(number, "value")
    assert _kinds(_findings(graph, text), FactsUnitFindingKind.UNGROUNDED_NUMBER)


@pytest.mark.parametrize(
    "text",
    [
        "The item weighs 5kg; a cell of 1.5V ran 12h at 100km/h.",
        "about 1.5 × 10−3 of them, 1.234,5 and 1'000 more",
        "shifts of 10-15 u and 8.5 ± 0.5 u, 2,000 items",
    ],
)
def test_every_number_the_inventory_lists_is_one_grounding_reads(text: str) -> None:
    """One reader: a measurement the gap pass is asked to recover is never a
    number the exit then removes as unstated."""
    from ontocast.util.numeric_inventory import (
        canonical_number,
        inventory_numeric_mentions,
    )

    inventory = inventory_numeric_mentions(text, unit_surfaces={"u"})
    stated = stated_numbers(text)
    listed = {canonical_number(m.value) for m in inventory.measurements}
    assert listed | set(inventory.unclassified)
    assert listed <= stated
    assert set(inventory.unclassified) <= stated


# -- identifiers are text, not numbers -------------------------------------------


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ('"1234567"', "Indexed as AB1234567."),
        ('"9783161484100"', "Catalogued as 978-3-16-148410-0."),
        ('"0042"', "Listed as item A0042."),
    ],
)
def test_an_untyped_digit_string_off_a_numeric_range_is_no_number(
    value: str, text: str
) -> None:
    graph = _graph(
        "@prefix schema: <https://schema.org/> ."
        ' cd:a a ex:Item ; rdfs:label "item" ; schema:identifier'
        f' [ a schema:PropertyValue ; schema:propertyID "id" ; schema:value {value} ] .'
    )
    before = set(graph)
    findings, result = _exit(graph, text)

    assert not _kinds(findings, FactsUnitFindingKind.UNGROUNDED_NUMBER)
    assert result.records == []
    assert set(graph) == before


def test_an_untyped_number_on_a_numeric_range_is_still_a_number() -> None:
    graph = _value('"4"', "several")
    assert _kinds(
        _findings(graph, "It lasted several rounds."),
        FactsUnitFindingKind.UNGROUNDED_NUMBER,
    )


def test_identifiers_in_a_text_without_quantities_lose_nothing() -> None:
    """Domain balance: an entity graph whose only digits are identifiers."""
    graph = _graph(
        "@prefix schema: <https://schema.org/> ."
        ' cd:a a ex:Item ; rdfs:label "item" ; ex:partOf cd:b ;'
        ' schema:identifier [ a schema:PropertyValue ; schema:value "12345" ] ,'
        ' [ a schema:PropertyValue ; schema:value "A0042" ] .'
        ' cd:b a ex:Item ; rdfs:label "group" .'
    )
    before = set(graph)
    findings, result = _exit(graph, "The item, registered as XY12345, is in a group.")

    value_kinds = {
        FactsUnitFindingKind.UNGROUNDED_NUMBER,
        FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL,
        FactsUnitFindingKind.UNKNOWN_TERM,
        FactsUnitFindingKind.DIMENSION_CONFLICT,
    }
    assert not [f for f in findings if f.kind in value_kinds]
    assert result.records == []
    assert set(graph) == before


# -- a cue counts only as a whole token ------------------------------------------


_UNITS = (
    CATALOG
    + """
unit:M a qudt:Unit ; rdfs:label "metre" ; qudt:symbol "m" .
unit:GM a qudt:Unit ; rdfs:label "gram" ; qudt:symbol "g" .
unit:DEG_C a qudt:Unit ; rdfs:label "degree Celsius" ; qudt:symbol "°C" .
unit:KiloM a qudt:Unit ; rdfs:label "kilometre" ; qudt:symbol "km" .
"""
)


def _unit_value(unit: str) -> RDFGraph:
    return _graph(
        'cd:a a ex:Item ; rdfs:label "item" ; ex:result [ a qudt:QuantityValue ;'
        f' rdfs:label "value" ; qudt:numericValue 5 ; qudt:unit unit:{unit} ] .'
    )


@pytest.mark.parametrize(
    ("unit", "text"),
    [
        ("M-PER-SEC", "It moves at 5 m/s."),
        ("GM-PER-L", "A solution of 5 g/L."),
        ("DEG_C-PER-MIN", "Heated at 5 °C/min."),
        ("M-PER-SEC2", "An acceleration of 5 m s-2."),
        ("GM-PER-MOL", "A mass of 5 g mol-1."),
        ("M2", "An area of 5 m2."),
    ],
)
def test_a_compound_unit_is_not_contradicted_by_its_first_factor(
    unit: str, text: str
) -> None:
    graph = _unit_value(unit)
    findings, _ = _exit(graph, text, _catalog(_UNITS))

    assert not _kinds(findings, FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL)
    assert (None, URIRef(QUDT + "unit"), URIRef(UNIT + unit)) in graph


def test_a_whole_token_cue_still_contradicts_a_unit_with_cues() -> None:
    graph = _unit_value("KiloM")
    findings, _ = _exit(graph, "It is 5 m long.", _catalog(_UNITS))

    (finding,) = _kinds(findings, FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL)
    assert finding.suggestions == [UNIT + "M"]
    assert not list(graph.triples((None, URIRef(QUDT + "unit"), None)))


# -- an individual a closed range takes is a listed member -----------------------


def test_a_minted_unit_on_a_value_is_unknown_and_dropped() -> None:
    graph = _graph(
        'cd:a a ex:Item ; ex:result [ a qudt:QuantityValue ; rdfs:label "3" ;'
        " qudt:numericValue 3 ; qudt:unit cd:made_up_unit ] ."
        ' cd:made_up_unit rdfs:label "made-up unit" .'
    )
    findings, result = _exit(graph, "3 made-up units")

    (finding,) = _kinds(findings, FactsUnitFindingKind.UNKNOWN_TERM)
    assert finding.predicate == CD + "made_up_unit"
    assert finding.facet == "closed_range_missing"
    assert not list(graph.triples((None, URIRef(QUDT + "unit"), None)))
    assert [record.kind for record in result.records] == [
        FactsGateRepairKind.UNKNOWN_TERM_DROPPED
    ]


def test_a_listed_currency_unit_is_a_member_of_the_unit_range() -> None:
    graph = _graph(
        'cd:a a ex:Item ; ex:result [ a qudt:QuantityValue ; rdfs:label "3" ;'
        " qudt:numericValue 3 ; qudt:unit <http://qudt.org/vocab/currency/EUR> ] ."
    )
    before = set(graph)
    findings, result = _exit(graph, "3 of them")

    assert not _kinds(findings, FactsUnitFindingKind.UNKNOWN_TERM)
    assert result.records == [] and set(graph) == before


# -- the exit removes a misused term only where it is misused --------------------


def test_a_fact_instance_used_as_a_class_loses_only_that_typing() -> None:
    graph = _graph(
        'cd:rex a cd:kind_one ; rdfs:label "Rex" .'
        ' cd:kind_one a ex:Item ; rdfs:label "kind one" ; ex:partOf cd:group .'
        " cd:group a ex:Item . cd:other a ex:Item ; ex:partOf cd:kind_one ."
    )
    findings, result = _exit(graph, "")

    (finding,) = _kinds(findings, FactsUnitFindingKind.UNKNOWN_TERM)
    assert finding.position == "type"
    assert (URIRef(CD + "rex"), RDFS.label, Literal("Rex")) in graph
    assert (URIRef(CD + "rex"), RDF.type, URIRef(CD + "kind_one")) not in graph
    # The instance's own description and the links to it stay.
    assert (
        URIRef(CD + "kind_one"),
        URIRef(EX + "partOf"),
        URIRef(CD + "group"),
    ) in graph
    assert (
        URIRef(CD + "other"),
        URIRef(EX + "partOf"),
        URIRef(CD + "kind_one"),
    ) in graph
    assert [record.triple_count for record in result.records] == [1]


# -- a namespace variant is the canonical term from the parse on -----------------


@pytest.mark.anyio
async def test_a_namespace_variant_is_canonical_before_any_finding(
    monkeypatch,
) -> None:
    async def fake_call_llm_with_retry(**kwargs):
        graph = RDFGraph()
        graph.parse(
            data=f"""
            @prefix cd: <{CD}> .
            @prefix ex: <{EX}> .
            cd:r a <https://www.w3.org/ns/sosa/Observation> ;
                <https://www.w3.org/ns/sosa/hasFeatureOfInterest> cd:x ;
                ex:count "5"^^<https://www.w3.org/2001/XMLSchema#integer> .
            cd:x a ex:Item .""",
            format="turtle",
        )
        return FactsRenderReport(semantic_graph=graph)

    monkeypatch.setattr(
        render_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )
    state = UnitFactsState(
        content_unit=ContentUnit(
            text="5 things", index=0, doc_iri=URIRef("https://ex.org/doc/1")
        ),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )

    result = await render_facts_module.render_facts_fresh(state, tools=_build_tools())

    graph = result.content_unit.graph
    assert (URIRef(CD + "r"), RDF.type, SOSA.Observation) in graph
    assert (URIRef(CD + "r"), SOSA.hasFeatureOfInterest, URIRef(CD + "x")) in graph
    assert Literal("5", datatype=XSD.integer) in set(graph.objects())
    assert not [
        term for triple in graph for term in triple if "https://www.w3" in str(term)
    ]


def test_a_patch_names_the_canonical_term() -> None:
    from ontocast.tool.facts_validation.critic_patch import _parse_fragment

    parsed = _parse_fragment(
        "cd:r <https://www.w3.org/ns/sosa/hasFeatureOfInterest> cd:x ;"
        ' ex:count "5"^^<https://www.w3.org/2001/XMLSchema#integer> .',
        _graph(""),
    )
    assert parsed is not None
    assert (URIRef(CD + "r"), SOSA.hasFeatureOfInterest, URIRef(CD + "x")) in parsed
    assert (
        URIRef(CD + "r"),
        URIRef(EX + "count"),
        Literal("5", datatype=XSD.integer),
    ) in (parsed)


# -- the mandate judges a pass's fixes together ----------------------------------


def test_a_link_and_the_type_of_its_new_end_may_come_as_two_fixes() -> None:
    graph = _graph(_GAPPED)
    mandate = _mandate(graph)
    link = _fix("cd:r1 sosa:hasFeatureOfInterest cd:thing .")
    typing = _fix('cd:thing a ex:Item ; rdfs:label "thing" .')

    kept, refused = mandate.admit([link, typing], graph)
    assert kept == [link, typing] and refused == {}
    # Alone, or before the link, the typing adds what nobody asked for.
    assert mandate.admit([typing, link], graph) == ([link], {"outside_mandate": 1})


# -- the calendar types are one family on a range --------------------------------


def test_any_valid_calendar_value_fits_a_temporal_range_and_stays_typed() -> None:
    graph = _graph(
        """
        cd:r a ex:Reading ;
            sosa:resultTime "2020-05-03"^^xsd:date , "2020"^^xsd:gYear ,
                "2020-05"^^xsd:gYearMonth , "2020-05-03T10:00:00"^^xsd:dateTime ,
                "sometime in spring" , "2020-13"^^xsd:gYearMonth ."""
    )
    found = range_mismatches(graph, _catalog(), CatalogRoles.of([_catalog()]))

    assert sorted(str(triple[2]) for triple, _ in found) == [
        "2020-13",
        "sometime in spring",
    ]


# -- small rules -----------------------------------------------------------------


def test_a_literal_is_already_stated_only_as_whole_tokens() -> None:
    from ontocast.tool.facts_validation.literal_repair import _states_text

    graph = _graph('cd:a rdfs:label "15 units" ; rdfs:comment "about 5 units" .')
    assert not _states_text(graph, URIRef(CD + "a"), "5 u")
    assert _states_text(graph, URIRef(CD + "a"), "about 5")
    assert not _states_text(
        _graph('cd:a rdfs:label "15 units" .'), URIRef(CD + "a"), "5"
    )


def test_the_cue_index_is_kept_per_graph_object_and_content() -> None:
    from ontocast.util.numeric_inventory import cue_index

    classes = {EX + "Qualifier"}
    first = _catalog()
    second = RDFGraph(identifier=first.identifier)
    second.parse(
        data=_PREFIXES
        + CATALOG.replace('skos:altLabel "about" , "~"', 'skos:altLabel "near" , "≈"'),
        format="turtle",
    )
    assert len(first) == len(second)
    assert "about" in cue_index(first, classes)
    assert "about" not in cue_index(second, classes)
    first.add((URIRef(EX + "precise"), SKOS.altLabel, Literal("sharp")))
    first.remove((URIRef(EX + "precise"), SKOS.altLabel, Literal("exactly")))
    assert "sharp" in cue_index(first, classes)


def test_a_superseded_vocabulary_is_accepted_but_never_offered():
    from ontocast.onto.well_known import (
        WELL_KNOWN_VOCABULARIES,
        is_well_known_term,
        well_known_prompt_list,
    )

    superseded = [v for v in WELL_KNOWN_VOCABULARIES if v.superseded]
    assert superseded
    for vocabulary in superseded:
        assert vocabulary.terms
        term = vocabulary.namespaces[0] + sorted(vocabulary.terms)[0]
        assert is_well_known_term(term)
        assert vocabulary.namespaces[0] not in well_known_prompt_list(measurement=True)
