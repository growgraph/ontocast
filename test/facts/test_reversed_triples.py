"""A triple written backwards is turned around, on two-sided evidence only.

``x p y`` is reversed when ``x`` contradicts ``p``'s domain and ``y`` its
range as written, and both fit once swapped. One-sided or undecidable
evidence leaves the triple to the domain finding.
"""

from __future__ import annotations

import pytest
from rdflib import BNode, URIRef

from ontocast.onto.model import FactsUnitFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation import repair_reversed_triples

pytestmark = pytest.mark.unit

EX = "https://ex.org/onto#"
CD = "https://ex.org/facts/"
LED_BY = URIRef(EX + "ledBy")
ORG, TASK = URIRef(CD + "org"), URIRef(CD + "task")

_PREFIXES = f"""
@prefix ex: <{EX}> . @prefix cd: <{CD}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
"""


def _ontology() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(
        data=_PREFIXES
        + """
        ex:Agent a owl:Class . ex:Task a owl:Class . ex:Place a owl:Class .
        ex:Org a owl:Class ; rdfs:subClassOf ex:Agent .
        ex:ledBy a owl:ObjectProperty ; rdfs:domain ex:Task ; rdfs:range ex:Agent .
        ex:pairedWith a owl:ObjectProperty , owl:SymmetricProperty ;
            rdfs:domain ex:Task ; rdfs:range ex:Agent .
        ex:noted a owl:ObjectProperty ; rdfs:domain ex:Task .
        """,
        format="turtle",
    )
    return graph


def _facts(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=_PREFIXES + body, format="turtle")
    return graph


_TYPED = "cd:org a ex:Org . cd:task a ex:Task . "


def test_a_triple_wrong_on_both_ends_is_turned_around_with_a_record() -> None:
    graph = _facts(_TYPED + "cd:org ex:ledBy cd:task .")
    reifier = BNode()
    graph.reifications.add(reifier, (ORG, LED_BY, TASK))

    count, records = repair_reversed_triples(graph, _ontology())

    assert count == 1
    assert (TASK, LED_BY, ORG) in graph and (ORG, LED_BY, TASK) not in graph
    assert graph.reifications.reifiers_of((TASK, LED_BY, ORG)) == {reifier}
    assert [(r.kind, r.source, r.target) for r in records] == [
        (FactsUnitFindingKind.TRIPLE_REVERSED, f"{ORG} {LED_BY}", str(TASK))
    ]


def test_a_reversal_already_stated_only_loses_the_wrong_triple() -> None:
    graph = _facts(_TYPED + "cd:org ex:ledBy cd:task . cd:task ex:ledBy cd:org .")

    assert repair_reversed_triples(graph, _ontology())[0] == 1
    assert list(graph.triples((None, LED_BY, None))) == [(TASK, LED_BY, ORG)]


@pytest.mark.parametrize(
    "body",
    [
        # the object fits the range: the subject is mistyped, not the direction
        "cd:org a ex:Org . cd:a a ex:Agent . cd:org ex:ledBy cd:a .",
        # swapped, the new object would not fit the range
        "cd:org a ex:Org . cd:p a ex:Place . cd:org ex:ledBy cd:p .",
        # an end with no type, or a type the context does not describe
        "cd:org a ex:Org . cd:org ex:ledBy cd:task .",
        "cd:org a ex:Org . cd:task a ex:Unknown . cd:org ex:ledBy cd:task .",
        # a symmetric property has no direction to get wrong
        _TYPED + "cd:org ex:pairedWith cd:task .",
        # no declared range: nothing to hold the swap against
        _TYPED + "cd:org ex:noted cd:task .",
        # a literal cannot be a subject
        _TYPED + 'cd:org ex:ledBy "the task" .',
        # written the right way round
        _TYPED + "cd:task ex:ledBy cd:org .",
    ],
)
def test_anything_short_of_two_sided_evidence_is_left_alone(body: str) -> None:
    graph = _facts(body)
    before = set(graph)

    assert repair_reversed_triples(graph, _ontology()) == (0, [])
    assert set(graph) == before


def test_no_context_is_a_no_op() -> None:
    graph = _facts(_TYPED + "cd:org ex:ledBy cd:task .")
    assert repair_reversed_triples(graph, None) == (0, [])


def test_axioms_the_snapshot_lacks_are_read_from_the_whole_catalog() -> None:
    """Retrieval may keep a property's domain and drop its range or a class's place."""
    from ontocast.tool.facts_validation.terms import CatalogRoles

    snapshot = RDFGraph()
    snapshot.parse(
        data=_PREFIXES
        + "ex:Task a owl:Class . ex:ledBy a owl:ObjectProperty ; rdfs:domain ex:Task .",
        format="turtle",
    )
    graph = _facts(_TYPED + "cd:org ex:ledBy cd:task .")
    assert repair_reversed_triples(graph, snapshot) == (0, [])

    context = CatalogRoles.of([_ontology()]).with_axioms(snapshot)

    assert repair_reversed_triples(graph, context)[0] == 1
    assert (TASK, LED_BY, ORG) in graph
    assert len(snapshot) == 3, "the snapshot itself is left as it was"
