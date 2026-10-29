"""How the served facts hang together: components, isolated instances, links lost."""

import pytest
from rdflib import URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation import (
    facts_connectivity,
    instance_links,
    links_lost,
)

pytestmark = pytest.mark.unit

CD = "https://facts.example/cd/"
DOC = URIRef("https://facts.example/doc/1")

PREFIXES = f"""
@prefix cd:   <{CD}> .
@prefix ex:   <https://x.org/schema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
"""


def _facts(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


def test_links_through_blank_nodes_join_instances() -> None:
    graph = _facts(
        """
        cd:a a ex:Sample ; ex:has [ ex:of cd:b ] .
        cd:b a ex:Device .
        cd:c a ex:Device ; ex:value [ ex:number 3 ] .
        """
    )
    result = facts_connectivity(graph, [CD])
    assert (result.instances, result.components, result.isolated) == (3, 2, 1)


def test_typing_aliases_and_provenance_join_nothing() -> None:
    graph = _facts(
        """
        cd:a a ex:Sample ; owl:sameAs cd:b ; prov:wasDerivedFrom cd:b .
        cd:b a ex:Sample .
        """
    )
    result = facts_connectivity(graph, [CD])
    assert (result.components, result.isolated) == (2, 2)


def test_the_document_node_joins_what_it_links_and_is_not_counted() -> None:
    graph = _facts(
        f"""
        <{DOC}> a foaf:Document ; ex:cites cd:paper_1, cd:paper_2 .
        cd:paper_1 a ex:Paper .
        cd:paper_2 a ex:Paper .
        """
    )
    joined = facts_connectivity(graph, [CD, str(DOC)], document=DOC)
    assert (joined.instances, joined.components, joined.isolated) == (2, 1, 0)
    apart = facts_connectivity(graph, [CD])
    assert (apart.components, apart.isolated) == (2, 2)


def test_links_lost_counts_removed_links_only() -> None:
    graph = _facts(
        """
        cd:a a ex:Sample ; rdfs:label "a" ; ex:on cd:b ; ex:with cd:c .
        cd:b a ex:Device .
        cd:c a ex:Device .
        """
    )
    before = instance_links(graph, [CD])
    assert len(before) == 2
    graph.remove(
        (URIRef(f"{CD}a"), URIRef("https://x.org/schema#with"), URIRef(f"{CD}c"))
    )
    graph.remove(
        (URIRef(f"{CD}a"), URIRef("http://www.w3.org/2000/01/rdf-schema#label"), None)
    )
    assert links_lost(before, graph) == 1


def test_an_empty_graph_has_no_components() -> None:
    assert facts_connectivity(RDFGraph(), [CD]).components == 0
