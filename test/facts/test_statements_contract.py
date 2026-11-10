"""The served facts are statements.

An instance that states nothing (no link, no value of its own) is dropped at
the gate, and so is a blank-node value no statement carries. Values are blank
nodes until the gate names them under their carrier.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDFS, BNode, Literal, URIRef

from ontocast.config import FactsValidationConfig
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import FactsGateRepairKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.facts_gate import drop_unlinked_instances, run_facts_gate
from ontocast.tool.agg.value_names import carrierless_values, name_values
from ontocast.tool.facts_validation import isolated_instances, silent_instances
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

CD = f"{DEFAULT_IRI}/"
EX = "https://x.org/schema#"
DOC = URIRef("https://x.org/doc/1")

PREFIXES = f"""
@prefix cd:   <{CD}> .
@prefix ex:   <{EX}> .
@prefix doc:  <https://x.org/doc/> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
"""


def _facts(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


def _reified(body: str) -> RDFGraph:
    """A facts graph with every triple reified, as the aggregator does."""
    graph = _facts(body)
    for triple in list(graph):
        graph.reifications.add(BNode(), triple)
    return graph


def _cd(name: str) -> URIRef:
    return URIRef(f"{CD}{name}")


def _quoted(graph: RDFGraph) -> set[tuple]:
    return {triple for _, triple in graph.reifications}


# ---------------------------------------------------------------------------
# Which instances state nothing
# ---------------------------------------------------------------------------


def test_silent_instances_are_isolated_ones_with_only_type_and_names() -> None:
    graph = _facts(
        """
        cd:obs a ex:Observation ; rdfs:label "obs" ; ex:about cd:place .
        cd:place a ex:Place ; rdfs:label "place" .
        cd:mention a ex:Place ; rdfs:label "mention" ; rdfs:comment "a place" .
        cd:coded a ex:Sample ; rdfs:label "coded" ; ex:hasCode "S1" .
        cd:measured a ex:Sample ; rdfs:label "measured" ;
            ex:hasValue [ ex:numericValue "3"^^xsd:decimal ] .
        """
    )

    assert isolated_instances(graph, [CD]) == {
        _cd("mention"),
        _cd("coded"),
        _cd("measured"),
    }
    assert silent_instances(graph, [CD]) == {_cd("mention")}


def test_a_link_to_the_document_is_a_statement() -> None:
    graph = _facts(
        """
        doc:1 ex:cites cd:work .
        cd:work a ex:Work ; rdfs:label "work" .
        """
    )

    assert silent_instances(graph, [CD], document=DOC) == set()
    assert silent_instances(graph, [CD]) == {_cd("work")}


def test_carrierless_values_are_blank_nodes_nothing_points_to() -> None:
    graph = _facts(
        """
        cd:s a ex:Sample ; rdfs:label "s" ; ex:hasValue [ ex:numericValue "1"^^xsd:decimal ] .
        [] a ex:Value ; rdfs:label "loose" ; ex:numericValue "2"^^xsd:decimal .
        """
    )

    loose = carrierless_values(graph)

    assert len(loose) == 1
    assert (next(iter(loose)), RDFS.label, Literal("loose")) in graph


# ---------------------------------------------------------------------------
# The drop
# ---------------------------------------------------------------------------


def test_drop_removes_silent_instances_and_loose_values_with_their_reifiers() -> None:
    graph = _reified(
        """
        cd:obs a ex:Observation ; rdfs:label "obs" ; ex:about cd:place ;
            ex:hasResult [ a ex:Value ; ex:numericValue "1"^^xsd:decimal ] .
        cd:place a ex:Place ; rdfs:label "place" .
        cd:mention a ex:Place ; rdfs:label "mention" .
        [] a ex:Value ; rdfs:label "loose" .
        """
    )
    quoted_before = len(_quoted(graph))

    _, dropped, values = drop_unlinked_instances(graph, [CD], DOC)

    assert dropped == [str(_cd("mention"))]
    assert values == 1
    assert (_cd("mention"), None, None) not in graph
    assert (None, RDFS.label, Literal("loose")) not in graph
    # The statement's value stays.
    assert (_cd("obs"), URIRef(f"{EX}hasResult"), None) in graph
    # Two triples of the mention, two of the loose value: their reifiers go.
    assert len(_quoted(graph)) == quoted_before - 4


def test_gate_drops_what_states_nothing_and_counts_it() -> None:
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.aggregated_facts = _facts(
        """
        cd:obs a ex:Observation ; rdfs:label "obs" ; ex:about cd:place .
        cd:place a ex:Place ; rdfs:label "place" .
        cd:mention a ex:Place ; rdfs:label "mention" .
        cd:coded a ex:Sample ; rdfs:label "coded" ; ex:hasCode "S1" .
        """
    )

    run_facts_gate(state, RDFGraph(), _fake_tools(), merge_repair=False)

    facts = state.aggregated_facts
    assert (_cd("mention"), None, None) not in facts
    assert (_cd("coded"), RDFS.label, Literal("coded")) in facts
    assert state.retrieval_metrics[RetrievalMetric.FACTS_UNLINKED_DROPPED] == 1
    assert [
        record.source
        for record in state.facts_gate_repairs
        if record.kind == FactsGateRepairKind.UNLINKED_DROPPED
    ] == [str(_cd("mention"))]
    # What is left isolated holds a value of its own.
    assert state.retrieval_metrics[RetrievalMetric.FACTS_ISOLATED] == 1


def _fake_tools() -> ToolBox:
    facts_validation = FactsValidationConfig()
    return cast(
        ToolBox,
        SimpleNamespace(
            shapes_catalog=SimpleNamespace(graph=lambda: None),
            shapes_prompt_contract=lambda: ("", (), False),
            shapes_chapter_terms=lambda context_terms: frozenset(),
            ontology_manager=SimpleNamespace(catalog_terms=lambda: set()),
            config=SimpleNamespace(
                get_tool_config=lambda: SimpleNamespace(
                    facts_validation=facts_validation
                )
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Values named under their carrier
# ---------------------------------------------------------------------------


def test_values_are_named_under_their_carrier_and_property() -> None:
    graph = _facts(
        """
        cd:obs a ex:Observation ;
            ex:hasResult [ a ex:Value ; rdfs:label "about 2"
                           ; ex:uncertainty [ ex:numericValue "0.4"^^xsd:decimal ] ] .
        """
    )

    assert name_values(graph) == 2

    result = URIRef(f"{CD}obs.hasResult")
    assert (_cd("obs"), URIRef(f"{EX}hasResult"), result) in graph
    assert (result, RDFS.label, Literal("about 2")) in graph
    assert (
        result,
        URIRef(f"{EX}uncertainty"),
        URIRef(f"{result}.uncertainty"),
    ) in graph
    assert not any(isinstance(node, BNode) for node in graph.all_nodes())


def test_several_values_on_one_property_are_numbered_by_content() -> None:
    body = """
        cd:s ex:hasValue [ rdfs:label "b" ] , [ rdfs:label "a" ] .
    """
    first, second = _facts(body), _facts(body)

    name_values(first)
    name_values(second)

    assert set(first) == set(second)
    assert (URIRef(f"{CD}s.hasValue"), RDFS.label, Literal("a")) in first
    assert (URIRef(f"{CD}s.hasValue2"), RDFS.label, Literal("b")) in first


def test_naming_repoints_the_reifiers_of_renamed_statements() -> None:
    graph = _reified(
        """
        cd:obs a ex:Observation ; ex:hasResult [ ex:numericValue "1"^^xsd:decimal ] .
        """
    )
    quoted_before = len(_quoted(graph))

    name_values(graph)

    named = URIRef(f"{CD}obs.hasResult")
    quoted = _quoted(graph)
    assert len(quoted) == quoted_before
    assert any(
        len(t) == 3 and str(t[0]) == str(_cd("obs")) and str(t[2]) == str(named)
        for t in quoted
    )


# ---------------------------------------------------------------------------
# Twin labels
# ---------------------------------------------------------------------------


def test_twin_labels_are_labels_several_instances_share() -> None:
    from ontocast.tool.facts_validation import twin_labels

    graph = _facts(
        """
        cd:a a ex:Place ; rdfs:label "Portland" .
        cd:b a ex:Place ; rdfs:label "portland " .
        cd:c a ex:Place ; rdfs:label "Seattle" .
        doc:1 rdfs:label "Portland" .
        """
    )

    assert twin_labels(graph, [CD], document=DOC) == {"portland": 2}


def test_gate_counts_twin_labels() -> None:
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.aggregated_facts = _facts(
        """
        cd:a a ex:Place ; rdfs:label "Portland" ; ex:hasCode "P1" .
        cd:b a ex:Place ; rdfs:label "Portland" ; ex:hasCode "P2" .
        """
    )

    run_facts_gate(state, RDFGraph(), _fake_tools(), merge_repair=False)

    assert state.retrieval_metrics[RetrievalMetric.FACTS_TWIN_LABELS] == 1
