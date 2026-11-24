"""Side-based unit checks read the whole catalog's axioms, not what retrieval kept.

A unit's snapshot may list a property without its domain. The domain and
role-link checks read the snapshot together with the catalog's axioms, so a
violation does not depend on what retrieval kept; the other checks still read
the snapshot the unit was given.
"""

from __future__ import annotations

import pytest
from rdflib import URIRef
from rdflib.namespace import SOSA

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.model import FactsUnitFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation.terms import CatalogRoles
from ontocast.tool.facts_validation.unit_findings import collect_unit_findings

pytestmark = pytest.mark.unit

CD = DEFAULT_IRI
EX = "https://ex.org/lab#"
PREFIXES = f"""@prefix cd: <{CD}> . @prefix ex: <{EX}> . @prefix sosa: <{SOSA}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <https://schema.org/> .
"""


def _graph(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


#: What retrieval kept: the classes and the property, without its domain.
SNAPSHOT = _graph(
    "ex:Place a owl:Class . ex:Record a owl:Class . "
    "ex:derivedFrom a owl:ObjectProperty ."
)
#: The catalog states the domain.
CATALOG = _graph(
    "ex:Place a owl:Class . ex:Record a owl:Class . "
    "ex:derivedFrom a owl:ObjectProperty ; rdfs:domain ex:Record . "
    "sosa:Observation a owl:Class . "
    "sosa:hasResult a owl:ObjectProperty ; schema:domainIncludes sosa:Observation ."
)
ROLES = CatalogRoles.of([CATALOG])
FACTS = _graph(
    'cd:station a ex:Place ; rdfs:label "station" ; ex:derivedFrom cd:old . '
    'cd:old a ex:Place ; rdfs:label "old station" .'
)


def _kinds(graph: RDFGraph, roles: CatalogRoles | None) -> list[FactsUnitFindingKind]:
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=SNAPSHOT,
        quarantined=[],
        extraction_text="The station was derived from the old station.",
        fact_namespaces=[CD],
        contract=ROLES.statement_contract(frozenset()),
        catalog_roles=roles,
    )
    return [finding.kind for finding in findings]


def test_a_domain_only_the_catalog_states_is_a_violation() -> None:
    assert FactsUnitFindingKind.DOMAIN_VIOLATION in _kinds(FACTS, ROLES)


def test_without_the_catalog_the_snapshot_alone_decides() -> None:
    assert FactsUnitFindingKind.DOMAIN_VIOLATION not in _kinds(FACTS, None)


def test_a_role_link_side_only_the_catalog_states_is_read() -> None:
    graph = _graph(
        'cd:station a ex:Place ; rdfs:label "station" ; '
        'sosa:hasResult [ rdfs:label "5 m" ] .'
    )
    assert FactsUnitFindingKind.ROLE_OUTSIDE_STATEMENT in _kinds(graph, ROLES)


def test_the_snapshot_given_to_the_unit_is_not_changed() -> None:
    before = len(SNAPSHOT)
    _kinds(FACTS, ROLES)
    assert len(SNAPSHOT) == before
    assert (URIRef(EX + "derivedFrom"), None, URIRef(EX + "Record")) not in SNAPSHOT
