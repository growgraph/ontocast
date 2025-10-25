"""Test Fuseki triple store functionality using mock implementation."""

from ontocast.tool.triple_manager.mock import MockFusekiTripleStoreManager
from test.conftest import (
    triple_store_roundtrip,
    triple_store_serialize_empty_facts,
    triple_store_serialize_facts,
)


def test_fuseki_triple_store_roundtrip(fuseki_triple_store_manager, test_ontology):
    """Test roundtrip operations with mock Fuseki triple store."""
    triple_store_roundtrip(fuseki_triple_store_manager, test_ontology)


def test_fuseki_serialize_facts(fuseki_triple_store_manager):
    """Test serializing facts with mock Fuseki triple store."""
    triple_store_serialize_facts(fuseki_triple_store_manager)


def test_fuseki_serialize_empty_facts(fuseki_triple_store_manager):
    """Test serializing empty facts with mock Fuseki triple store."""
    triple_store_serialize_empty_facts(fuseki_triple_store_manager)


def test_mock_fuseki_initialization():
    """Test that mock Fuseki triple store initializes correctly."""
    manager = MockFusekiTripleStoreManager(
        uri="http://localhost:3030/test",
        auth=("admin", "password"),
        dataset="test",
        clean=True,
    )

    assert manager.uri == "http://localhost:3030/test"
    assert manager.auth == ("admin", "password")
    assert manager.dataset == "test"
    assert manager.ontologies_dataset == "ontologies"
    assert len(manager.fetch_ontologies()) == 0


def test_mock_fuseki_clean_operation():
    """Test that mock Fuseki triple store clean operation works."""
    manager = MockFusekiTripleStoreManager(clean=True)

    # Add some data with owl:Ontology declaration
    from rdflib import Literal, URIRef
    from rdflib.namespace import OWL, RDF, RDFS

    from ontocast.onto.rdfgraph import RDFGraph

    graph = RDFGraph()
    # Add ontology declaration
    graph.add((URIRef("http://example.org/test"), RDF.type, OWL.Ontology))
    graph.add((URIRef("http://example.org/Person"), RDF.type, RDFS.Class))
    graph.add((URIRef("http://example.org/Person"), RDFS.label, Literal("Person")))

    result = manager.serialize(graph, graph_uri="http://example.org/test")
    assert result is True
    assert len(manager.fetch_ontologies()) == 1

    # Clean and verify
    manager.clear()
    assert len(manager.fetch_ontologies()) == 0
    assert len(manager.graphs) == 0
