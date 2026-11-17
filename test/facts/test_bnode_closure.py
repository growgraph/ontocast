"""A critic fix that touches a blank-node structure removes all of it."""

import pytest
from rdflib import OWL, RDF, RDFS, BNode, URIRef
from rdflib.collection import Collection

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation.critic_patch import _bnode_closure

pytestmark = pytest.mark.unit

EX = "https://example.com/onto#"
CLASS = URIRef(f"{EX}Sample")


def _graph_with_nested_union() -> tuple[RDFGraph, tuple]:
    """``Sample subClassOf [ owl:unionOf (A B) ]``: a list nested in a restriction."""
    graph = RDFGraph()
    restriction, members = BNode(), BNode()
    edge = (CLASS, RDFS.subClassOf, restriction)
    graph.add(edge)
    graph.add((restriction, RDF.type, OWL.Class))
    graph.add((restriction, OWL.unionOf, members))
    Collection(graph, members, [URIRef(f"{EX}A"), URIRef(f"{EX}B")])
    return graph, edge


def test_deleting_the_edge_to_a_nested_structure_removes_every_level() -> None:
    graph, edge = _graph_with_nested_union()

    closure = _bnode_closure([edge], graph)

    assert set(closure) == set(graph)
    assert closure[0] == edge


def test_deleting_one_list_cell_removes_the_structure_and_its_edge() -> None:
    graph, edge = _graph_with_nested_union()
    cell = next(t for t in graph if t[1] == RDF.first)

    closure = set(_bnode_closure([cell], graph))

    assert closure == set(graph)


def test_a_structure_still_pointed_at_by_another_subject_is_kept() -> None:
    graph, edge = _graph_with_nested_union()
    other = (URIRef(f"{EX}Other"), RDFS.subClassOf, edge[2])
    graph.add(other)

    closure = _bnode_closure([edge], graph)

    assert closure == [edge]
