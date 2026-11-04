"""Scaffolding is the nodes the pipeline mints, never a namespace."""

import pytest
from rdflib import URIRef

from ontocast.onto.embedding_policy import strip_provenance_triples_for_embedding
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import (
    chunk_nodes,
    document_nodes,
    is_scaffolding,
    reifier_nodes,
    scaffolding_nodes,
)
from ontocast.tool.triple_manager.core import TripleStoreManager

pytestmark = pytest.mark.unit

CD = "https://facts.example/cd/"

_GRAPH = f"""
@prefix cd:     <{CD}> .
@prefix ex:     <https://x.org/schema#> .
@prefix rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:    <http://www.w3.org/2002/07/owl#> .
@prefix prov:   <http://www.w3.org/ns/prov#> .
@prefix schema: <https://schema.org/> .
@prefix foaf:   <http://xmlns.com/foaf/0.1/> .
@prefix dcterms: <http://purl.org/dc/terms/> .

cd:doc a prov:Entity, foaf:Document ; dcterms:title "A report" .
cd:chunk_0 a prov:Entity, schema:Text ; schema:position 0 .
cd:study a prov:Activity ; prov:used cd:dataset ; rdfs:label "study" .
cd:dataset a prov:Entity ; prov:wasDerivedFrom cd:survey .
cd:survey a ex:Survey .
[ a rdf:Statement ; rdf:subject cd:study ; rdf:predicate prov:used ;
  rdf:object cd:dataset ; prov:wasDerivedFrom cd:chunk_0 ] .
"""


def _graph() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=_GRAPH, format="turtle")
    return graph


def _cd(local: str) -> URIRef:
    return URIRef(CD + local)


def test_minted_nodes_are_recognised_by_shape() -> None:
    graph = _graph()

    assert chunk_nodes(graph) == {_cd("chunk_0")}
    assert len(reifier_nodes(graph)) == 1
    assert document_nodes(graph) == {_cd("doc")}
    # A domain prov:Entity is neither a chunk nor the document.
    assert _cd("dataset") not in scaffolding_nodes(graph) | document_nodes(graph)


def test_domain_prov_triples_are_not_scaffolding() -> None:
    graph = _graph()
    nodes = scaffolding_nodes(graph)

    used = (_cd("study"), URIRef("http://www.w3.org/ns/prov#used"), _cd("dataset"))
    assert not is_scaffolding(used, nodes)
    assert any(
        is_scaffolding(t, nodes) for t in graph.triples((None, None, _cd("chunk_0")))
    )


def test_strip_provenance_keeps_domain_prov_and_drops_bookkeeping() -> None:
    clean = TripleStoreManager.strip_provenance(_graph())

    prov = "http://www.w3.org/ns/prov#"
    assert (_cd("study"), URIRef(prov + "used"), _cd("dataset")) in clean
    assert (_cd("dataset"), URIRef(prov + "wasDerivedFrom"), _cd("survey")) in clean
    assert not list(clean.triples((_cd("chunk_0"), None, None)))
    assert not list(clean.triples((None, None, _cd("chunk_0"))))


def test_strip_provenance_drops_an_ontology_headers_lineage() -> None:
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix prov: <http://www.w3.org/ns/prov#> .
        <https://o.example/onto> a owl:Ontology ;
            prov:wasDerivedFrom <https://o.example/doc/1> .
        """,
        format="turtle",
    )
    clean = TripleStoreManager.strip_provenance(graph)

    assert len(clean) == 1


def test_embedding_keeps_domain_prov_and_drops_the_document_hub() -> None:
    embedded = strip_provenance_triples_for_embedding(_graph())

    prov = "http://www.w3.org/ns/prov#"
    assert (_cd("study"), URIRef(prov + "used"), _cd("dataset")) in embedded
    assert not list(embedded.triples((_cd("doc"), None, None)))
    assert not list(embedded.triples((_cd("chunk_0"), None, None)))
