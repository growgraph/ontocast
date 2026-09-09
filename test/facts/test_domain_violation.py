"""Tests for the ``rdfs:domain`` contradiction check on rendered facts.

Asserting a triple entails the subject belongs to the predicate's declared
domain, so an untyped subject is never a violation. The check fires only when
the subject carries an asserted type that is unrelated to the domain, since
that is where inference produces a contradiction the renderer can still fix.
"""

import pytest

from ontocast.onto.model import FactsUnitFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation import (
    collect_unit_findings,
    domain_violation_findings,
)

pytestmark = pytest.mark.unit

ONTOLOGY = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <https://x.org/schema#> .
@prefix ext:  <https://outside.org/schema#> .

ex:Process a owl:Class .
ex:Observation a owl:Class .
ex:TimedObservation a owl:Class ; rdfs:subClassOf ex:Observation .
ex:Effect a owl:Class .

# Defined only as an intersection: no asserted subClassOf edge to its genus.
ex:TimedEffect a owl:Class ;
    owl:equivalentClass [ a owl:Class ; owl:intersectionOf ( ex:Effect ) ] .

ex:hasTimeResult a owl:ObjectProperty ;
    rdfs:domain ex:TimedObservation .
ex:hasEffect a owl:ObjectProperty ;
    rdfs:domain ex:Effect .

# Domain in a vocabulary this context does not describe.
ex:hasExternal a owl:ObjectProperty ;
    rdfs:domain ext:Thing .
"""

EX = "https://x.org/schema#"
CD = "https://growgraph.dev/facts#"

PREFIXES = f"""
@prefix ex: <{EX}> .
@prefix ext: <https://outside.org/schema#> .
@prefix cd: <{CD}> .
"""


def _ontology() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=ONTOLOGY, format="turtle")
    return graph


def _facts(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


def test_reports_unrelated_asserted_type():
    """A type in a different branch than the declared domain is a violation."""
    graph = _facts("cd:p a ex:Process ; ex:hasTimeResult cd:v .")
    findings = domain_violation_findings(graph, _ontology())

    assert len(findings) == 1
    finding = findings[0]
    assert finding.kind == FactsUnitFindingKind.DOMAIN_VIOLATION
    assert finding.mandatory is True
    assert finding.subject == f"{CD}p"
    assert finding.predicate == f"{EX}hasTimeResult"
    assert finding.suggestions == [f"{EX}TimedObservation"]


@pytest.mark.parametrize(
    "body,reason",
    [
        (
            "cd:o a ex:TimedObservation ; ex:hasTimeResult cd:v .",
            "exact domain",
        ),
        (
            "cd:o ex:hasTimeResult cd:v .",
            "untyped subject: the domain is entailed, not contradicted",
        ),
        (
            "cd:o a ex:Observation ; ex:hasTimeResult cd:v .",
            "supertype of the domain: inference specializes it",
        ),
        (
            "cd:o a ex:Process, ex:TimedObservation ; ex:hasTimeResult cd:v .",
            "one compatible type among several is enough",
        ),
        (
            "cd:e a ex:TimedEffect ; ex:hasEffect cd:v .",
            "genus reached only through an owl:equivalentClass intersection",
        ),
        (
            "cd:x a ex:Process ; ex:hasExternal cd:v .",
            "domain belongs to a vocabulary the context does not describe",
        ),
        (
            "cd:x a ext:Thing ; ex:hasTimeResult cd:v .",
            "asserted type belongs to a vocabulary the context does not describe",
        ),
    ],
)
def test_no_false_positive(body, reason):
    assert domain_violation_findings(_facts(body), _ontology()) == [], reason


def test_no_ontology_context_is_not_a_violation():
    graph = _facts("cd:p a ex:Process ; ex:hasTimeResult cd:v .")
    assert domain_violation_findings(graph, None) == []
    assert domain_violation_findings(graph, RDFGraph()) == []


def test_one_finding_per_subject_predicate_pair():
    """Repeating the predicate on one subject reports once, not per object."""
    graph = _facts("cd:p a ex:Process ; ex:hasTimeResult cd:v1, cd:v2 .")
    assert len(domain_violation_findings(graph, _ontology())) == 1


def test_surfaces_through_collect_unit_findings():
    """The check reaches the renderer through the unit findings assembly."""
    graph = _facts("cd:p a ex:Process ; ex:hasTimeResult cd:v .")
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_ontology(),
        quarantined=[],
        extraction_text="text with no numbers",
        fact_namespaces=[CD],
    )
    kinds = [finding.kind for finding in findings]
    assert FactsUnitFindingKind.DOMAIN_VIOLATION in kinds
