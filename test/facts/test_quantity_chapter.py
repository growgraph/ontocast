"""The quantity chapter, attribution, required roles, dimensions and wider suggestions.

The quantity rules are one chapter, given to a content unit only when its
text states a number, in the render, critic and gap-pass prompts alike; its
terms are the catalog's first. A statement attributed to other work links to
that work. Results under an anchored property that disagree in dimension are
the unit's to resolve; any other such conflict is a document warning. A
non-catalog term is offered a same-named catalog term of its kind from any
namespace, and nothing applies it.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import FOAF, PROV, RDF, Literal, URIRef

from ontocast.config.settings import ChunkConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import LLMGraphFormat, OntologyAssemblyMode, Status
from ontocast.onto.model import (
    FactsCompletionReport,
    FactsCritiqueReport,
    FactsRenderReport,
    FactsUnitFinding,
    FactsUnitFindingKind,
    FactsValidationFindingKind,
)
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.prefixes import PrefixTable
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import FactScope, document_nodes
from ontocast.onto.unit_states import UnitFactsState
from ontocast.onto.well_known import well_known_value_classes
from ontocast.prompt.complete_facts import (
    build_term_sheet,
    completion_guidelines,
    output_instruction_for,
)
from ontocast.prompt.criticise_facts import evaluation_instruction_for
from ontocast.prompt.facts_guidelines import (
    DEFAULT_QUANTITY_FALLBACK_VOCABULARY,
    format_facts_operational_guidelines,
    format_quantity_chapter,
    format_quantity_fallback_clause,
    quantity_terms,
)
from ontocast.stategraph.unit_exit import ABSENCE_KINDS, EXIT_ACTIONS, ExitAction
from ontocast.tool.agg.statement_conflict import dimension_conflict_findings
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import collect_unit_findings
from ontocast.tool.facts_validation.literal_repair import repair_property_aliases
from ontocast.tool.facts_validation.mandate import GapMandate
from ontocast.tool.facts_validation.statements import StatementContract
from ontocast.tool.facts_validation.terms import CatalogRoles, ValidationPolicy
from ontocast.tool.facts_validation.unit_findings import unit_prompt_scope
from ontocast.util.measurement_lexicon import Mention
from ontocast.util.numeric_inventory import NumericInventory, states_quantity

pytestmark = pytest.mark.unit

CD = "https://growgraph.dev/facts/"
EX = "https://ex.org/v#"
OTHER = "https://other.org/w#"
DOC = URIRef("https://ex.org/doc/d1")

WITH_NUMBERS = "The archive holds 1,200 records and its main room is 40 m long."
WITHOUT_NUMBERS = (
    "The society is a members' association. It runs a library and a "
    "reading room, and its board appoints the librarian."
)

PREFIXES = f"""
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:   <http://www.w3.org/2002/07/owl#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix sosa:  <http://www.w3.org/ns/sosa/> .
@prefix qudt:  <http://qudt.org/schema/qudt/> .
@prefix unit:  <http://qudt.org/vocab/unit/> .
@prefix quantitykind: <http://qudt.org/vocab/quantitykind/> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
@prefix schema: <https://schema.org/> .
@prefix foaf:  <http://xmlns.com/foaf/0.1/> .
@prefix ex:    <{EX}> .
@prefix other: <{OTHER}> .
@prefix cd:    <{CD}> .
"""

CATALOG = """
ex:Reading a owl:Class ; rdfs:subClassOf sosa:Observation ; rdfs:label "reading" .
ex:hasAmount a owl:ObjectProperty ; rdfs:subPropertyOf sosa:hasResult .
ex:Amount a owl:Class ; rdfs:subClassOf qudt:QuantityValue .
ex:Item a owl:Class .
ex:hasLength a owl:ObjectProperty ; rdfs:label "has length" .
ex:length a sosa:ObservableProperty ; rdfs:label "length" ;
    qudt:applicableUnit unit:M .
ex:span a sosa:ObservableProperty ; rdfs:label "span" ;
    qudt:hasQuantityKind quantitykind:Length .
quantitykind:Length qudt:applicableUnit unit:M .
ex:usualUnit rdfs:subPropertyOf qudt:applicableUnit .
ex:extent a sosa:ObservableProperty ; ex:usualUnit unit:M .
ex:kindOnly a sosa:ObservableProperty ; qudt:hasQuantityKind ex:someKind .
"""


def _graph(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


def _catalog() -> RDFGraph:
    return _graph(CATALOG)


def _contract(catalog: RDFGraph | None = None) -> StatementContract:
    return CatalogRoles.of(
        [catalog if catalog is not None else _catalog()]
    ).statement_contract(frozenset(well_known_value_classes()))


def _citations() -> dict[str, str]:
    return dict(ChunkConfig().citation_vocabulary)


# -- the decision ---------------------------------------------------------------


def test_a_text_states_a_quantity_when_its_inventory_holds_a_number() -> None:
    assert states_quantity(WITH_NUMBERS)
    assert states_quantity("Membership rose by a third to 300.")
    assert not states_quantity(WITHOUT_NUMBERS)
    # A year is not a quantity.
    assert not states_quantity("The society was founded in 1998.")


PLAIN_CATALOG = """
ex:Item a owl:Class .
ex:hasLength a owl:ObjectProperty ; rdfs:label "has length" .
ex:partOf a owl:ObjectProperty .
"""


def _scope(text: str, catalog: RDFGraph | None, *, citation: bool = False):
    return unit_prompt_scope(
        extraction_text=text,
        ontology_graph=catalog,
        contract=_contract(catalog),
        is_citation_metadata=citation,
    )


def test_one_scope_decides_the_conditional_rules() -> None:
    plain = _graph(PLAIN_CATALOG)
    # Numbers bring the chapter and the statement rules.
    scope = _scope(WITH_NUMBERS, plain)
    assert (scope.quantities, scope.statements, scope.attribution) == (True, True, True)
    # A catalog statement class brings the statement rules, not the chapter.
    scope = _scope(WITHOUT_NUMBERS, _catalog())
    assert (scope.quantities, scope.statements) == (False, True)
    # Neither: no statement rule either.
    scope = _scope(WITHOUT_NUMBERS, plain)
    assert (scope.quantities, scope.statements) == (False, False)
    # A citation-metadata unit: no chapter, no attribution rule.
    scope = _scope(WITH_NUMBERS, plain, citation=True)
    assert (scope.quantities, scope.attribution) == (False, False)


# -- the render guidelines --------------------------------------------------------


def _chapter(catalog: RDFGraph | None = None) -> str:
    vocabulary = DEFAULT_QUANTITY_FALLBACK_VOCABULARY
    return format_quantity_chapter(
        vocabulary,
        quantity_terms(catalog, _contract(catalog), vocabulary),
    )


def _guidelines(
    quantity_chapter: str,
    citations: dict[str, str] | None = None,
    *,
    statement_rules: bool = False,
) -> str:
    return format_facts_operational_guidelines(
        facts_namespace=CD,
        domain_ontologies_clause="`ex:`",
        jsonld=False,
        quantity_chapter=quantity_chapter,
        statement_rules=statement_rules,
        citation_vocabulary=_citations() if citations is None else citations,
    )


#: Quantity terms and rules a unit without numbers must not see.
_QUANTITY_MARKERS = (
    "3a. QUANTITIES",
    "QUANTITATIVE COMPLETENESS",
    "EXACT QUANTITY",
    "STRUCTURED VALUE",
    "qudt:QuantityValue",
    "qudt:numericValue",
    "qudt:unit",
    "sosa:hasResult",
    "sosa:observedProperty",
    "`sosa:Observation`",
    "unit:M",
    "measurement unit",
)


def test_the_quantity_chapter_is_given_only_with_numbers() -> None:
    with_numbers = _guidelines(_chapter())
    without = _guidelines("")

    for marker in _QUANTITY_MARKERS:
        assert marker in with_numbers, marker
        assert marker not in without, marker
    # Every other rule is shared.
    for shared in ("3b. CONNECT", "8a. OBJECT PROPERTIES"):
        assert shared in with_numbers and shared in without
    # The currency line follows the chapter.
    assert "priceCurrency" in with_numbers and "priceCurrency" not in without


#: Statement rules outside the chapter.
_STATEMENT_MARKERS = (
    "3c. EVERY STATEMENT NAMES ITS SUBJECT",
    "sosa:madeBySensor",
    "carry every role its class requires",
    "`sosa:` <",
    "`qudt:` <",
    "`unit:` <",
)


def test_the_statement_rules_follow_the_scope() -> None:
    with_numbers = _guidelines(_chapter())
    catalog_only = _guidelines("", statement_rules=True)
    neither = _guidelines("")

    for marker in _STATEMENT_MARKERS:
        assert marker in with_numbers, marker
        assert marker in catalog_only, marker
        assert marker not in neither, marker
    # Rule 1e keeps its provenance teaching point.
    assert "`prov:wasDerivedFrom`" in neither
    # The catalog alone brings no quantity rule.
    assert "3a. QUANTITIES" not in catalog_only


def test_the_exact_quantity_sentence_is_where_the_observed_property_is_introduced() -> (
    None
):
    text = " ".join(_chapter().split())
    introduced = text.index("whose observed property names what was measured")
    exact = text.index(
        "Reuse a catalog individual only when it names exactly the quantity "
        "stated, its kind and what it is relative to. A change, difference or "
        "ratio between two states is a quantity of its own. Otherwise mint a "
        "`cd:` instance of the observable-property class TERMS names."
    )
    assert introduced < exact


def test_the_catalog_terms_come_before_the_registry_fallback() -> None:
    catalog = _catalog()
    terms = quantity_terms(
        catalog, _contract(catalog), DEFAULT_QUANTITY_FALLBACK_VOCABULARY
    )

    for own, fallback in (
        ("`ex:Reading`", "`sosa:Observation`"),
        ("`ex:hasAmount`", "`sosa:hasResult`"),
        ("`ex:Amount`", "`qudt:QuantityValue`"),
    ):
        assert terms.index(own) < terms.index(fallback), own
    # Without a catalog, only the fallback is named.
    bare = quantity_terms(
        None, StatementContract(), DEFAULT_QUANTITY_FALLBACK_VOCABULARY
    )
    assert "ex:" not in bare and "`sosa:Observation`" in bare


def test_terms_name_the_observable_property_class_rule_3a_mints_in() -> None:
    """Rule 3a mints an observed property in the class TERMS names: the
    catalog's own first, the well-known one otherwise."""
    assert "mint a `cd:` instance of the observable-property class TERMS names" in (
        " ".join(_chapter().split())
    )
    catalog = _catalog()
    terms = quantity_terms(
        catalog, _contract(catalog), DEFAULT_QUANTITY_FALLBACK_VOCABULARY
    )
    assert "observable-property class `sosa:ObservableProperty`" in terms
    own = _graph(
        CATALOG + "ex:Trait a owl:Class ; rdfs:subClassOf sosa:ObservableProperty ."
    )
    terms = quantity_terms(own, _contract(own), DEFAULT_QUANTITY_FALLBACK_VOCABULARY)
    assert terms.index("`ex:Trait`") < terms.index("`sosa:ObservableProperty`")


def test_without_statement_rules_no_rule_names_observations_or_measuring() -> None:
    plain = " ".join(_guidelines("", citations={}).split())
    rule_1e = plain.split("1e. DOMAIN FIT")[1][:300]
    assert "observation" not in rule_1e
    rule_3b = plain.split("3b. CONNECT WHAT THE TEXT CONNECTS")[1][:200]
    assert "measures" not in rule_3b
    with_statements = " ".join(_guidelines("", statement_rules=True).split())
    assert "provenance or observation property" in with_statements


def test_the_critic_and_the_currency_line_carry_no_domain_example() -> None:
    critic = evaluation_instruction_for(states_quantity=True)
    for word in ("AppealCourt", "Rouen", "fca:"):
        assert word not in critic
    assert "does not fit its property's declared range" in " ".join(critic.split())
    assert "USD" not in _guidelines(_chapter())


def test_no_text_falls_back_to_a_comment() -> None:
    for vocabulary in (DEFAULT_QUANTITY_FALLBACK_VOCABULARY, {}):
        assert "comment" not in format_quantity_fallback_clause(vocabulary)
    assert "rdfs:comment" not in _chapter()
    for mandate in (GapMandate(role_gaps=(_gap(),)), _measurement_mandate()):
        assert "rdfs:comment" not in completion_guidelines(mandate)
    # The one mention left names an identity annotation of opaque IRIs.
    text = _guidelines(_chapter())
    assert text.count("rdfs:comment") == 1
    assert "identity is determined EXCLUSIVELY by `rdfs:label`, `rdfs:comment`" in text
    assert "When no measurement unit fits, add no unit link" in " ".join(text.split())


def test_the_attribution_rule_names_the_configured_term() -> None:
    text = _guidelines("")

    assert "3d. ATTRIBUTED STATEMENTS" in text
    assert "links to that work with `dcterms:source`" in text
    assert "typed `schema:ScholarlyArticle`" in text
    # Rule 4: the marker is not an entity, the cited work is.
    assert (
        "The work a citation marker refers to is an entity (rule 3d); "
        "the marker is not." in text
    )
    critic = evaluation_instruction_for(
        states_quantity=False, citation_vocabulary=_citations()
    )
    assert "linked with `dcterms:source`" in critic

    plain = _guidelines("", citations={})
    assert "ATTRIBUTED STATEMENTS" not in plain
    assert "citation marker refers to" not in plain


def test_the_checklist_checks_every_required_role() -> None:
    for chapter, statements in ((_chapter(), False), ("", True)):
        checklist = _guidelines(chapter, statement_rules=statements).split(
            "FINAL STRUCTURAL VALIDATION CHECKLIST"
        )[1]
        assert "carry every role its class requires" in checklist
        assert "the instance it is about (rule 3c)" in checklist
        assert "the property it reports" in checklist


# -- the critic and the gap pass ------------------------------------------------


def test_the_critic_judges_quantities_only_with_numbers() -> None:
    assert "NUMERIC COVERAGE" in evaluation_instruction_for(states_quantity=True)
    assert "NUMERIC COVERAGE" not in evaluation_instruction_for(states_quantity=False)


def _gap() -> FactsUnitFinding:
    return FactsUnitFinding(
        kind=FactsUnitFindingKind.MISSING_ROLE,
        message="lacks its property",
        subject=CD + "r1",
        facet="property",
    )


def _measurement_mandate() -> GapMandate:
    return GapMandate(
        inventory=NumericInventory(
            measurements=[Mention(value="5", unit="m", start=0, end=0, context="")]
        )
    )


def _subject_gap() -> FactsUnitFinding:
    return _gap().model_copy(update={"facet": "subject"})


def test_the_gap_pass_asks_for_values_only_when_its_mandate_lists_measurements() -> (
    None
):
    """The prompt asks for what the mandate admits, from the same mandate: a
    call for a role gap is never shown how to write a value, whatever the
    text states."""
    catalog = _catalog()
    unit_property = {"http://qudt.org/schema/qudt/unit"}
    values = _measurement_mandate()
    property_gap = GapMandate(role_gaps=(_gap(),))
    subject_gap = GapMandate(role_gaps=(_subject_gap(),))

    with_values = completion_guidelines(values)
    assert "unit-valued property" in with_values
    assert "names exactly the quantity stated" in with_values
    # A statement lacking its property is told how to choose one, never how
    # to write a value.
    for_property = completion_guidelines(property_gap)
    assert "names exactly the quantity stated" in for_property
    assert "unit-valued property" not in for_property
    for_subject = completion_guidelines(subject_gap)
    assert "names exactly the quantity stated" not in for_subject
    assert "unit-valued property" not in for_subject

    sheet = build_term_sheet(
        catalog,
        unit_property,
        value_terms=DEFAULT_QUANTITY_FALLBACK_VOCABULARY,
        contract=_contract(catalog),
    )
    bare = build_term_sheet(
        catalog,
        unit_property,
        value_terms=DEFAULT_QUANTITY_FALLBACK_VOCABULARY,
        contract=_contract(catalog),
        quantities=False,
        values=False,
    )
    assert "Value node terms" in sheet and "Value node terms" not in bare
    assert "Quantity/observation/condition classes" not in bare
    # The subject links stay: a role gap is repaired with them.
    assert "sosa:hasFeatureOfInterest" in bare

    for fmt in LLMGraphFormat:
        assert "numericValue" in output_instruction_for(fmt, values)
        for mandate in (property_gap, subject_gap):
            assert "numericValue" not in output_instruction_for(fmt, mandate)


def test_the_term_sheet_names_the_observable_property_class() -> None:
    """The exact-quantity guideline mints in the class the term sheet names:
    the catalog's own first, the well-known one otherwise."""
    catalog = _catalog()
    plain = build_term_sheet(
        catalog, (), contract=_contract(catalog), property_class=True
    )
    assert "Observable-property class" in plain
    assert "sosa:ObservableProperty" in plain
    own = _graph(
        CATALOG + "ex:Trait a owl:Class ; rdfs:subClassOf sosa:ObservableProperty ."
    )
    sheet = build_term_sheet(own, (), contract=_contract(own), property_class=True)
    line = next(row for row in sheet.splitlines() if "Observable-property class" in row)
    assert "ex:Trait" in line and "sosa:ObservableProperty" not in line
    assert "Observable-property class" not in build_term_sheet(
        catalog, (), contract=_contract(catalog)
    )


# -- the three prompts, end to end ---------------------------------------------


def _tools(catalog: RDFGraph | None = None) -> AtomicToolBox:
    async def get_llm_tool(_budget_tracker):
        return object()

    catalog = catalog if catalog is not None else _catalog()
    contract = _contract(catalog)
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            prefix_table=lambda: PrefixTable({}, {}),
            get_llm_tool=get_llm_tool,
            web_grounding_enabled_for_node=lambda _node: False,
            object_property_literal_check=True,
            property_alias_min_ratio=0.85,
            catalog_terms=lambda: set(),
            catalog_roles=lambda: CatalogRoles.of([catalog]),
            statement_contract=lambda: contract,
            code_predicates=(),
            citation_vocabulary=_citations(),
            facts_completion_passes=0,
            facts_completion_measurements=False,
            quantity_fallback_vocabulary=dict(DEFAULT_QUANTITY_FALLBACK_VOCABULARY),
            facts_vocabulary="strict",
            additional_standard_namespaces=(),
            acceptance_policy=None,
            validation_policy=ValidationPolicy(),
        ),
    )


def _state(
    text: str,
    graph: RDFGraph | None = None,
    *,
    catalog: RDFGraph | None = None,
    citation: bool = False,
) -> UnitFactsState:
    unit = ContentUnit(text=text, index=0, doc_iri=DOC, is_citation_metadata=citation)
    if graph is not None:
        unit.graph = graph
    state = UnitFactsState(
        content_unit=unit,
        ontology_snapshot=OntologySnapshot.from_graph(
            catalog if catalog is not None else _catalog(),
            source_iris=["https://ex.org/v"],
            assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
        ),
    )
    state.status = Status.SUCCESS
    return state


async def _prompts(
    monkeypatch,
    text: str,
    *,
    catalog: RDFGraph | None = None,
    citation: bool = False,
    mandate: GapMandate | None = None,
) -> dict[str, str]:
    render_agent = importlib.import_module("ontocast.agent.render_facts")
    critic_agent = importlib.import_module("ontocast.agent.criticise_facts")
    gap_agent = importlib.import_module("ontocast.agent.complete_facts")
    prompts: dict[str, str] = {}

    def capturing(name: str, report):
        async def fake(**kwargs):
            prompts[name] = kwargs["prompt"].format(**kwargs["prompt_kwargs"])
            return report

        return fake

    monkeypatch.setattr(
        render_agent,
        "call_llm_with_retry",
        capturing("render", FactsRenderReport(semantic_graph=_graph(""))),
    )
    monkeypatch.setattr(
        critic_agent,
        "call_llm_with_retry",
        capturing(
            "critic",
            FactsCritiqueReport(
                success=True,
                score=90,
                actionable_triple_fixes=[],
                systemic_critique_summary="",
            ),
        ),
    )
    monkeypatch.setattr(
        gap_agent,
        "call_llm_with_retry",
        capturing("gap", FactsCompletionReport(proposed_fixes=[])),
    )
    tools = _tools(catalog)

    def state(graph: RDFGraph | None = None) -> UnitFactsState:
        return _state(text, graph, catalog=catalog, citation=citation)

    await render_agent.render_facts_fresh(state(), tools=tools)
    graph = _graph(
        'cd:r1 a ex:Reading ; rdfs:label "reading" ; ex:hasAmount [ a ex:Amount ] .'
    )
    await critic_agent.criticise_facts(state(graph), tools=tools)
    await gap_agent.complete_facts(
        state(graph), tools, mandate or GapMandate(role_gaps=(_gap(),))
    )
    return prompts


@pytest.mark.anyio
async def test_the_three_prompts_carry_the_quantity_rules_only_with_numbers(
    monkeypatch,
) -> None:
    with_numbers = await _prompts(monkeypatch, WITH_NUMBERS)
    without = await _prompts(monkeypatch, WITHOUT_NUMBERS)
    recovering = await _prompts(
        monkeypatch, WITH_NUMBERS, mandate=_measurement_mandate()
    )

    assert set(with_numbers) == set(without) == {"render", "critic", "gap"}
    assert "3a. QUANTITIES" in with_numbers["render"]
    assert "3a. QUANTITIES" not in without["render"]
    assert "NUMERIC COVERAGE" in with_numbers["critic"]
    assert "NUMERIC COVERAGE" not in without["critic"]
    assert "qudt:numericValue" in with_numbers["render"]
    assert "qudt:numericValue" not in without["render"]
    # The gap pass asks for a value only when its mandate lists measurements,
    # whatever the text states.
    assert "unit-valued property" in recovering["gap"]
    assert "qudt:numericValue" in recovering["gap"]
    for prompts in (with_numbers, without):
        assert "unit-valued property" not in prompts["gap"]
        assert "qudt:numericValue" not in prompts["gap"]
    # The catalog's terms come first in the render's chapter.
    render = with_numbers["render"]
    assert render.index("`ex:Reading`") < render.index("`sosa:Observation`")
    for name in ("render", "critic", "gap"):
        assert "rdfs:comment)" not in with_numbers[name]
        assert "as text on the value node" not in with_numbers[name]


# -- attribution ------------------------------------------------------------------


CITED = """
cd:item_1 a ex:Item ; rdfs:label "the item" .
cd:r1 a ex:Reading ; rdfs:label "reported length" ;
    sosa:hasFeatureOfInterest cd:item_1 ;
    sosa:observedProperty ex:length ;
    ex:hasAmount [ a ex:Amount ; rdfs:label "40 m" ;
        qudt:numericValue 40 ; qudt:unit unit:M ] ;
    dcterms:source cd:work_12 .
cd:work_12 a schema:ScholarlyArticle ; rdfs:label "[12]" .
"""


@pytest.mark.anyio
async def test_a_cited_statement_keeps_its_attribution_link(monkeypatch) -> None:
    render_agent = importlib.import_module("ontocast.agent.render_facts")

    async def fake(**kwargs):
        return FactsRenderReport(semantic_graph=_graph(CITED))

    monkeypatch.setattr(render_agent, "call_llm_with_retry", fake)
    text = "Earlier work reported that the item is 40 m long [12]."
    state = await render_agent.render_facts_fresh(_state(text), tools=_tools())

    graph = state.content_unit.graph
    work = URIRef(CD + "work_12")
    assert (URIRef(CD + "r1"), URIRef("http://purl.org/dc/terms/source"), work) in graph
    # PROV-O in the catalog does not make the link a domain violation.
    catalog = _catalog()
    catalog.parse(
        data=PREFIXES + "prov:wasQuotedFrom rdfs:domain prov:Entity .", format="turtle"
    )
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=catalog,
        quarantined=[],
        extraction_text=text,
        fact_namespaces=[CD, str(DOC)],
        contract=_contract(catalog),
        document=DOC,
    )
    assert not [
        f
        for f in findings
        if f.kind
        in (FactsUnitFindingKind.DOMAIN_VIOLATION, FactsUnitFindingKind.UNKNOWN_TERM)
    ]
    scope = FactScope.of(graph, [CD, str(DOC)], DOC)
    assert work in scope.instances
    assert work not in scope.scaffolding and work not in scope.documents


def test_a_cited_work_is_never_the_document_node() -> None:
    graph = _graph(CITED)
    for node in (DOC, URIRef(CD + "work_12")):
        # The worst case: the cited work typed as the pipeline types the
        # document node.
        graph.add((node, RDF.type, PROV.Entity))
        graph.add((node, RDF.type, FOAF.Document))
    graph.add((DOC, URIRef("http://purl.org/dc/terms/title"), Literal("The document")))

    assert document_nodes(graph, DOC) == {DOC}
    scope = FactScope.of(graph, [CD], DOC)
    assert scope.documents == {DOC}
    assert URIRef(CD + "work_12") in scope.instances


# -- the dimension finding ------------------------------------------------------


def _results(prop: str, *units: str) -> RDFGraph:
    body = 'cd:item_1 a ex:Item ; rdfs:label "the item" .\n'
    for index, unit in enumerate(units):
        unit_link = f"; qudt:unit {unit}" if unit else ""
        body += (
            f'cd:r{index} a ex:Reading ; rdfs:label "r{index}" ; '
            f"sosa:hasFeatureOfInterest cd:item_1 ; sosa:observedProperty {prop} ; "
            f'ex:hasAmount [ a ex:Amount ; rdfs:label "{index + 2}" ; '
            f"qudt:numericValue {index + 2} {unit_link} ] .\n"
        )
    return _graph(body)


def _unit_dimension_findings(graph: RDFGraph) -> list[FactsUnitFinding]:
    catalog = _catalog()
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=catalog,
        quarantined=[],
        extraction_text="readings of 2, 3 and 4",
        fact_namespaces=[CD, str(DOC)],
        contract=_contract(catalog),
        document=DOC,
    )
    return [f for f in findings if f.kind == FactsUnitFindingKind.DIMENSION_CONFLICT]


def _document_dimension_findings(graph: RDFGraph):
    return [
        f
        for f in dimension_conflict_findings(graph, _catalog(), [CD, str(DOC)], DOC)
        if f.kind == FactsValidationFindingKind.DIMENSION_CONFLICT
    ]


@pytest.mark.parametrize("prop", ["ex:length", "ex:span", "ex:extent"])
def test_a_dimension_conflict_under_an_anchored_property_is_mandatory(
    prop: str,
) -> None:
    # An applicable unit, a quantity kind whose unit the catalog names, and a
    # catalog sub-property of the anchor each anchor the property.
    graph = _results(prop, "unit:M", "unit:SEC")

    findings = _unit_dimension_findings(graph)
    assert len(findings) == 1
    assert findings[0].mandatory
    assert findings[0].suggestions == [CD + "r1"]
    assert [f.severity for f in _document_dimension_findings(graph)] == ["warning"]


def test_a_single_result_against_its_anchor_is_a_conflict() -> None:
    findings = _unit_dimension_findings(_results("ex:length", "unit:SEC"))
    assert [f.suggestions for f in findings] == [[CD + "r0"]]


def test_an_anchor_of_unknown_dimension_still_makes_a_disagreement_mandatory() -> None:
    findings = _unit_dimension_findings(_results("ex:kindOnly", "unit:M", "unit:SEC"))
    assert len(findings) == 1 and findings[0].mandatory
    assert findings[0].suggestions == []


def test_an_unanchored_dimension_conflict_is_a_document_warning() -> None:
    graph = _results("cd:size", "unit:M", "unit:SEC")

    assert _unit_dimension_findings(graph) == []
    assert [f.severity for f in _document_dimension_findings(graph)] == ["warning"]


def test_the_dimension_finding_is_inert_without_measurement_units() -> None:
    for prop in ("ex:length", "cd:size"):
        graph = _results(prop, "", "")
        assert _unit_dimension_findings(graph) == []
        assert _document_dimension_findings(graph) == []
    # Agreeing units are no conflict.
    assert (
        _document_dimension_findings(_results("cd:size", "unit:M", "unit:NanoM")) == []
    )


def test_dimensionless_quantities_never_disagree() -> None:
    graph = _results("cd:share", "unit:PERCENT", "unit:RAD", "unit:NUM")
    assert _document_dimension_findings(graph) == []


def test_results_of_one_statement_that_disagree_are_reported() -> None:
    graph = _graph(
        """
        cd:item_1 a ex:Item ; rdfs:label "the item" .
        cd:r1 a ex:Reading ; rdfs:label "r1" ; sosa:hasFeatureOfInterest cd:item_1 ;
            ex:hasAmount [ a ex:Amount ; rdfs:label "2" ; qudt:numericValue 2 ;
                           qudt:unit unit:M ] ,
                         [ a ex:Amount ; rdfs:label "3" ; qudt:numericValue 3 ;
                           qudt:unit unit:SEC ] .
        """
    )
    assert [f.subject for f in _document_dimension_findings(graph)] == [CD + "r1"]


def test_a_dimension_conflict_is_carried_to_the_document_report() -> None:
    assert EXIT_ACTIONS[FactsUnitFindingKind.DIMENSION_CONFLICT] is ExitAction.REPORT
    assert FactsUnitFindingKind.DIMENSION_CONFLICT in ABSENCE_KINDS


# -- wider suggestions ----------------------------------------------------------


def test_a_same_named_catalog_term_in_another_namespace_is_suggested_not_applied() -> (
    None
):
    catalog = _catalog()
    catalog.parse(
        data=PREFIXES
        + "other:Anchor a owl:Class .\nother:note a owl:DatatypeProperty .",
        format="turtle",
    )
    graph = _graph(
        """
        cd:item_1 a ex:Item ; rdfs:label "the item" ;
            other:has_length cd:item_2 .
        cd:item_2 a ex:Item ; rdfs:label "the other item" .
        """
    )
    before = set(graph)

    _, _, records = repair_property_aliases(graph, catalog)
    assert records == []
    assert set(graph) == before

    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=catalog,
        quarantined=[],
        extraction_text="The item is linked to the other item.",
        fact_namespaces=[CD, str(DOC)],
        contract=_contract(catalog),
        document=DOC,
    )
    unknown = [
        f
        for f in findings
        if f.kind == FactsUnitFindingKind.UNKNOWN_TERM
        and f.predicate == OTHER + "has_length"
    ]
    assert len(unknown) == 1
    assert EX + "hasLength" in unknown[0].suggestions
    # A class of the same name is not a property's suggestion.
    assert all("Anchor" not in s for s in unknown[0].suggestions)


@pytest.mark.anyio
async def test_without_numbers_or_a_catalog_statement_class_no_statement_rule_is_shown(
    monkeypatch,
) -> None:
    detached = GapMandate(detached=(URIRef(CD + "a"), URIRef(CD + "b")))
    plain = await _prompts(
        monkeypatch, WITHOUT_NUMBERS, catalog=_graph(PLAIN_CATALOG), mandate=detached
    )
    with_class = await _prompts(monkeypatch, WITHOUT_NUMBERS, mandate=detached)

    assert "3c. EVERY STATEMENT NAMES ITS SUBJECT" not in plain["render"]
    assert "`sosa:` <" not in plain["render"]
    assert "sosa:madeBySensor" not in plain["render"]
    assert "3c. EVERY STATEMENT NAMES ITS SUBJECT" in with_class["render"]
    assert "`sosa:` <" in with_class["render"]
    assert "3a. QUANTITIES" not in with_class["render"]
    # A gap pass for detached instances alone repairs no statement.
    for prompts in (plain, with_class):
        assert "Subject properties" not in prompts["gap"]
        assert "STATEMENT GAPS, add" not in prompts["gap"]
        assert "DETACHED INSTANCES, add" in prompts["gap"]


@pytest.mark.anyio
async def test_a_role_gap_keeps_the_statement_roles_in_the_gap_pass(
    monkeypatch,
) -> None:
    prompts = await _prompts(
        monkeypatch, WITHOUT_NUMBERS, catalog=_graph(PLAIN_CATALOG)
    )
    assert "Subject properties" in prompts["gap"]
    assert "STATEMENT GAPS, add" in prompts["gap"]


@pytest.mark.anyio
async def test_a_citation_unit_gets_neither_the_chapter_nor_attribution(
    monkeypatch,
) -> None:
    prompts = await _prompts(monkeypatch, WITH_NUMBERS, citation=True)

    assert "CITATION-METADATA UNIT" in prompts["render"]
    for absent in ("3a. QUANTITIES", "ATTRIBUTED STATEMENTS", "rule 3d"):
        assert absent not in prompts["render"], absent
    assert "NUMERIC COVERAGE" not in prompts["critic"]
    assert "linked with `dcterms:source`" not in prompts["critic"]
    assert "unit-valued property" not in prompts["gap"]
