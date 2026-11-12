"""The link pass: what it shows the model, what it accepts back, what it changes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from rdflib import URIRef

from ontocast.agent.link_facts import render_link_entities, render_link_properties
from ontocast.config.settings import FactsValidationConfig
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.enum import FactsLinkRejection, RetrievalMetric
from ontocast.onto.model import FactsLinkProposal
from ontocast.onto.prefixes import PrefixTable
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.well_known import well_known_value_classes
from ontocast.tool.agg.rewriter import add_reified_provenance
from ontocast.tool.facts_validation.linking import (
    LinkCandidates,
    select_link_candidates,
    statement_sources,
    validate_link_proposals,
)

pytestmark = pytest.mark.unit

VALUE_CLASSES = well_known_value_classes()

DOC = "https://x.org/doc/1"
D = DOC + "/"
Q = "https://x.org/schema#"

ONTOLOGY = f"""
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix q:    <{Q}> .

q:Edition a owl:Class ; rdfs:label "edition" .
q:Tournament a owl:Class ; rdfs:label "tournament" .
q:Venue a owl:Class ; rdfs:label "venue" .
q:editionOf a owl:ObjectProperty ; rdfs:label "edition of" ;
    rdfs:domain q:Edition ; rdfs:range q:Tournament .
q:heldAt a owl:ObjectProperty ; rdfs:label "held at" ;
    rdfs:domain q:Tournament ; rdfs:range q:Venue .
q:year a owl:DatatypeProperty ; rdfs:range xsd:gYear .
"""

TEXT = (
    "The Spring Cup is held at Harbour Hall. "
    "The 1971 edition was the first edition of the Spring Cup. "
    "Tickets sold out."
)

FACTS = f"""
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix q: <{Q}> .
@prefix d: <{D}> .

d:springCup a q:Tournament ; rdfs:label "Spring Cup" ; q:heldAt d:harbourHall .
d:harbourHall a q:Venue ; rdfs:label "Harbour Hall" .
d:edition1971 a q:Edition ; rdfs:label "1971 edition" ; q:year "1971" .
"""


def _ontology() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=ONTOLOGY, format="turtle")
    return graph


def _unit(text: str = TEXT) -> ContentUnit:
    return ContentUnit(text=text, index=0, doc_iri=URIRef(DOC), type=OutputType.FACTS)


def _served(unit: ContentUnit, facts: str = FACTS) -> RDFGraph:
    """The facts with every statement derived from ``unit``, as aggregation leaves them."""
    graph = RDFGraph()
    graph.parse(data=facts, format="turtle")
    for s, p, o in list(graph):
        add_reified_provenance(graph, s, p, o, URIRef(unit.iri_absolute))
    return graph


def _candidates(graph: RDFGraph, unit: ContentUnit) -> LinkCandidates:
    candidates = select_link_candidates(
        graph, [D + "edition1971"], [unit], _ontology(), [D], max_isolated=20
    )
    assert candidates is not None
    return candidates


def test_statement_sources_read_the_units_from_the_reifiers() -> None:
    unit = _unit()
    sources = statement_sources(_served(unit))

    assert sources[URIRef(D + "edition1971")] == {URIRef(unit.iri_absolute)}


def test_candidates_list_the_named_target_first_and_only_object_properties() -> None:
    unit = _unit()
    candidates = _candidates(_served(unit), unit)

    assert candidates.isolated == [D + "edition1971"]
    assert candidates.targets[0] == D + "springCup"
    assert "q:editionOf" in candidates.properties
    assert Q + "year" not in candidates.properties.values()
    # Tournament -> Venue fits no pair with the isolated edition at either end.
    assert Q + "heldAt" not in candidates.properties.values()
    assert "first edition of the Spring Cup" in candidates.passages[D + "edition1971"]


def test_an_instance_no_unit_names_is_not_shown() -> None:
    unit = _unit("Nothing here mentions it.")

    assert (
        select_link_candidates(
            _served(unit), [D + "edition1971"], [unit], _ontology(), [D], max_isolated=5
        )
        is None
    )


def test_the_prompt_shows_no_iri_and_does_not_depend_on_triple_order() -> None:
    unit = _unit()
    forward = _served(unit)
    backward = RDFGraph()
    for triple in sorted(forward, key=str, reverse=True):
        backward.add(triple)
    for reifier, triple in forward.reifications:
        backward.reifications.add(reifier, triple)

    def render(graph: RDFGraph) -> str:
        candidates = _candidates(graph, unit)
        return render_link_entities(candidates, graph) + render_link_properties(
            candidates, _ontology()
        )

    text = render(forward)
    assert text == render(backward)
    assert D not in text and "_:" not in text
    assert "Entity 1 (ISOLATED)" in text
    assert "q:editionOf" in text


def _proposal(subject: int, prop: str, obj: int, evidence: str) -> FactsLinkProposal:
    return FactsLinkProposal(
        subject=subject, property=prop, object=obj, evidence=evidence
    )


def test_a_stated_link_is_accepted_with_its_unit() -> None:
    unit = _unit()
    graph = _served(unit)
    candidates = _candidates(graph, unit)
    cup = candidates.entities().index(D + "springCup") + 1

    accepted, rejected = validate_link_proposals(
        # Case and spacing differ from the text: evidence is compared normalised.
        [_proposal(1, "q:editionOf", cup, "first  edition of the SPRING cup")],
        candidates,
        graph,
        _ontology(),
        [unit],
        value_classes=VALUE_CLASSES,
    )

    assert [(a.subject, a.predicate, a.object) for a in accepted] == [
        (D + "edition1971", Q + "editionOf", D + "springCup")
    ]
    assert accepted[0].unit == unit.iri_absolute
    assert not rejected


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda e: _proposal(1, "q:editionOf", 99, "first edition"), "unknown_entity"),
        (
            lambda e: _proposal(
                e["springCup"], "q:heldAt", e["harbourHall"], "held at"
            ),
            "no_isolated_endpoint",
        ),
        (lambda e: _proposal(1, "q:editionOf", 1, "first edition"), "self_link"),
        (
            lambda e: _proposal(1, "q:invented", e["springCup"], "first edition"),
            "unknown_property",
        ),
        (
            lambda e: _proposal(1, "q:year", e["springCup"], "first edition"),
            "not_object_property",
        ),
        (
            lambda e: _proposal(1, "q:heldAt", e["harbourHall"], "first edition"),
            "domain_mismatch",
        ),
        (
            lambda e: _proposal(1, "q:editionOf", e["harbourHall"], "first edition"),
            "range_mismatch",
        ),
        (
            lambda e: _proposal(
                1, "q:editionOf", e["springCup"], "the final was in May"
            ),
            "evidence_missing",
        ),
    ],
)
def test_each_check_rejects_with_its_reason(make: Any, reason: str) -> None:
    unit = _unit()
    graph = _served(unit)
    candidates = _candidates(graph, unit)
    # Listed here only to reach the checks after the listing: no shown pair
    # fits q:heldAt, and q:year is a datatype property.
    candidates.properties["q:year"] = Q + "year"
    candidates.properties["q:heldAt"] = Q + "heldAt"
    numbers = {
        iri.removeprefix(D): number
        for number, iri in enumerate(candidates.entities(), start=1)
    }

    accepted, rejected = validate_link_proposals(
        [make(numbers)],
        candidates,
        graph,
        _ontology(),
        [unit],
        value_classes=VALUE_CLASSES,
    )

    assert accepted == []
    assert rejected == {FactsLinkRejection(reason): 1}


def test_a_repeated_link_is_a_duplicate_and_a_taken_functional_property_is_refused() -> (
    None
):
    unit = _unit()
    graph = _served(unit)
    candidates = _candidates(graph, unit)
    cup = candidates.entities().index(D + "springCup") + 1
    link = _proposal(1, "q:editionOf", cup, "first edition of the Spring Cup")

    _, rejected = validate_link_proposals(
        [link, link],
        candidates,
        graph,
        _ontology(),
        [unit],
        value_classes=VALUE_CLASSES,
    )
    assert rejected == {FactsLinkRejection.DUPLICATE: 1}

    functional = _ontology()
    functional.parse(
        data=f"<{Q}editionOf> a <http://www.w3.org/2002/07/owl#FunctionalProperty> .",
        format="turtle",
    )
    _, rejected = validate_link_proposals(
        [link, link], candidates, graph, functional, [unit], value_classes=VALUE_CLASSES
    )
    assert rejected == {FactsLinkRejection.FUNCTIONAL_TAKEN: 1}


def _state(graph: RDFGraph, isolated: list[str]) -> AgentState:
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.aggregated_facts = graph
    state.facts_isolated_instances = isolated
    state.retrieval_metrics[RetrievalMetric.FACTS_ISOLATED] = len(isolated)
    state.retrieval_metrics[RetrievalMetric.FACTS_COMPONENTS] = 2
    return state


def _tools(**settings: Any) -> Any:
    facts_validation = FactsValidationConfig(**settings)
    return SimpleNamespace(
        prefix_table=lambda: PrefixTable({}, {}),
        get_atomic_tools=lambda: SimpleNamespace(value_classes=lambda: VALUE_CLASSES),
        config=SimpleNamespace(
            get_tool_config=lambda: SimpleNamespace(facts_validation=facts_validation)
        ),
    )


@pytest.mark.anyio
async def test_the_pass_adds_a_valid_link_with_provenance(monkeypatch) -> None:
    from ontocast.agent import link_facts
    from ontocast.stategraph.facts_link import run_link_pass

    unit = _unit()
    graph = _served(unit)

    async def propose(state, tools, candidates, graph, ontology_graph):
        cup = candidates.entities().index(D + "springCup") + 1
        return [
            _proposal(1, "q:editionOf", cup, "first edition of the Spring Cup"),
            _proposal(1, "q:editionOf", cup, "not in the text"),
        ]

    monkeypatch.setattr(link_facts, "propose_links", propose)
    state = _state(graph, [D + "edition1971"])

    await run_link_pass(state, _tools(link_pass=True), _ontology(), [unit])

    link = (URIRef(D + "edition1971"), URIRef(Q + "editionOf"), URIRef(D + "springCup"))
    assert link in state.aggregated_facts
    reifiers = state.aggregated_facts.reifications.reifiers_of(link)
    assert reifiers and URIRef(unit.iri_absolute) in set(
        state.aggregated_facts.objects(next(iter(reifiers)), None)
    )
    metrics = state.retrieval_metrics
    assert metrics[RetrievalMetric.FACTS_LINK_ISOLATED] == 1
    assert metrics[RetrievalMetric.FACTS_LINK_ADDED] == 1
    assert metrics[RetrievalMetric.FACTS_LINK_REJECTIONS_BY_REASON] == {
        "evidence_missing": 1
    }
    assert metrics[RetrievalMetric.FACTS_ISOLATED] == 0
    assert metrics[RetrievalMetric.FACTS_COMPONENTS] == 1
    assert state.facts_isolated_instances == []
    assert [r.unit for r in state.facts_links_added] == [unit.iri_absolute]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("isolated", "settings"),
    [([], {"link_pass": True}), ([D + "edition1971"], {"link_pass": False})],
)
async def test_no_call_without_isolated_instances_or_with_the_pass_off(
    monkeypatch, isolated: list[str], settings: dict[str, Any]
) -> None:
    from ontocast.agent import link_facts
    from ontocast.stategraph.facts_link import run_link_pass

    async def never(*_args: Any, **_kwargs: Any) -> list[FactsLinkProposal]:
        raise AssertionError("the model must not be called")

    monkeypatch.setattr(link_facts, "propose_links", never)
    unit = _unit()
    state = _state(_served(unit), isolated)
    before = len(state.aggregated_facts)

    await run_link_pass(state, _tools(**settings), _ontology(), [unit])

    assert len(state.aggregated_facts) == before
    assert RetrievalMetric.FACTS_LINK_ADDED not in state.retrieval_metrics


@pytest.mark.anyio
async def test_a_failed_call_changes_nothing(monkeypatch) -> None:
    from ontocast.agent import link_facts
    from ontocast.stategraph.facts_link import run_link_pass

    calls: list[int] = []

    async def failing_call(*_args: Any, **_kwargs: Any) -> Any:
        calls.append(1)
        raise ValueError("unparseable")

    monkeypatch.setattr(link_facts, "call_llm_with_retry", failing_call)

    async def llm_tool(_tracker: Any) -> Any:
        return None

    tools = _tools(link_pass=True)
    tools.get_llm_tool = llm_tool
    unit = _unit()
    state = _state(_served(unit), [D + "edition1971"])
    before = len(state.aggregated_facts)

    await run_link_pass(state, tools, _ontology(), [unit])

    assert calls == [1]
    assert len(state.aggregated_facts) == before
    assert state.retrieval_metrics[RetrievalMetric.FACTS_LINK_ADDED] == 0


@pytest.mark.anyio
async def test_the_single_unit_path_runs_the_gate_then_the_pass(monkeypatch) -> None:
    from ontocast.api import process_helpers

    order: list[str] = []
    monkeypatch.setattr(
        process_helpers,
        "validate_unit_pipeline_facts",
        lambda *_args: order.append("gate"),
    )

    async def link(state, tools, ontology_graph, units):
        order.append(f"link:{len(units)}")

    monkeypatch.setattr(process_helpers, "run_link_pass", link)

    await process_helpers.finish_unit_pipeline_facts(
        AgentState(), RDFGraph(), _tools(), [_unit()]
    )

    assert order == ["gate", "link:1"]


def test_a_structured_value_is_reached_only_through_a_fitting_range() -> None:
    """A quantity is no place: an undecidable range does not admit a value."""
    unit = _unit(TEXT + " The hall covers ten acres.")
    facts = (
        FACTS
        + """
d:hallArea a <http://qudt.org/schema/qudt/QuantityValue> ; rdfs:label "ten acres" .
"""
    )
    graph = _served(unit, facts)
    candidates = select_link_candidates(
        graph, [D + "hallArea"], [unit], _ontology(), [D], max_isolated=5
    )
    assert candidates is not None
    candidates.properties["schema:location"] = "https://schema.org/location"
    hall = candidates.entities().index(D + "harbourHall") + 1

    accepted, rejected = validate_link_proposals(
        [_proposal(hall, "schema:location", 1, "The hall covers ten acres")],
        candidates,
        graph,
        _ontology(),
        [unit],
        value_classes=VALUE_CLASSES,
    )

    assert accepted == []
    assert rejected == {FactsLinkRejection.VALUE_ENDPOINT: 1}


def test_a_bare_local_name_matching_one_listed_property_is_accepted() -> None:
    unit = _unit()
    graph = _served(unit)
    candidates = _candidates(graph, unit)
    cup = candidates.entities().index(D + "springCup") + 1

    accepted, rejected = validate_link_proposals(
        [_proposal(1, "editionOf", cup, "first edition of the Spring Cup")],
        candidates,
        graph,
        _ontology(),
        [unit],
        value_classes=VALUE_CLASSES,
    )

    assert [a.predicate for a in accepted] == [Q + "editionOf"]
    assert not rejected


def test_evidence_must_come_from_where_both_ends_are_named() -> None:
    """A real quote from a sentence that names neither end proves no link."""
    unit = _unit(
        "The 1971 edition was the first edition of the Spring Cup. "
        "The Spring Cup is held at Harbour Hall. "
        "Prices rose that year. Parking was scarce. "
        "Tickets sold out."
    )
    graph = _served(unit)
    candidates = _candidates(graph, unit)
    cup = candidates.entities().index(D + "springCup") + 1

    accepted, rejected = validate_link_proposals(
        [_proposal(1, "q:editionOf", cup, "Tickets sold out")],
        candidates,
        graph,
        _ontology(),
        [unit],
        value_classes=VALUE_CLASSES,
    )

    assert accepted == []
    assert rejected == {FactsLinkRejection.EVIDENCE_UNRELATED: 1}


COUNT_ONTOLOGY = f"""
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix q:    <{Q}> .
q:Count a owl:Class .
q:countAt a owl:ObjectProperty ; rdfs:domain q:Count ; rdfs:range q:Tournament .
"""

COUNT_FACTS = f"""
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix q: <{Q}> .
@prefix d: <{D}> .

d:springCup a q:Tournament ; rdfs:label "Spring Cup" .
d:crowd a q:Count ; rdfs:label "attendance figure of the event" ;
    q:countValue d:crowd.countValue .
d:crowd.countValue rdfs:label "12,000 spectators" ; q:number "12000"^^xsd:integer .
"""


def test_a_statement_is_found_by_its_value_as_written_when_its_label_is_not() -> None:
    """The value's label is the mention as written; the statement's is a paraphrase."""
    unit = _unit(
        "A crowd of 12,000 spectators watched the Spring Cup. Tickets sold out."
    )
    graph = _served(unit, COUNT_FACTS)
    ontology = _ontology()
    ontology.parse(
        data="@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> ." + COUNT_ONTOLOGY,
        format="turtle",
    )

    candidates = select_link_candidates(
        graph, [D + "crowd"], [unit], ontology, [D], max_isolated=5
    )

    assert candidates is not None
    assert "12,000 spectators watched" in candidates.passages[D + "crowd"]
    cup = candidates.entities().index(D + "springCup") + 1
    accepted, rejected = validate_link_proposals(
        [_proposal(1, "q:countAt", cup, "12,000 spectators watched the Spring Cup")],
        candidates,
        graph,
        ontology,
        [unit],
        value_classes=VALUE_CLASSES,
    )
    assert [(link.subject, link.object) for link in accepted] == [
        (D + "crowd", D + "springCup")
    ]
    assert rejected == {}


def test_a_bare_number_does_not_locate_a_statement() -> None:
    """A number alone occurs inside other numbers, so it names nothing."""
    unit = _unit("In 120005 the Spring Cup moved. Tickets sold out.")
    graph = _served(
        unit,
        COUNT_FACTS.replace("12,000 spectators", "12000"),
    )
    ontology = _ontology()
    ontology.parse(
        data="@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> ." + COUNT_ONTOLOGY,
        format="turtle",
    )

    assert (
        select_link_candidates(
            graph, [D + "crowd"], [unit], ontology, [D], max_isolated=5
        )
        is None
    )
