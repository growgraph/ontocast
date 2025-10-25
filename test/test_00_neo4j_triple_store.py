"""Test Neo4j triple store functionality using mock implementation."""

from ontocast.tool.triple_manager.mock import MockNeo4jTripleStoreManager
from test.conftest import (
    triple_store_roundtrip,
    triple_store_serialize_empty_facts,
    triple_store_serialize_facts,
)


def test_neo4j_triple_store_roundtrip(neo4j_triple_store_manager, test_ontology):
    """Test roundtrip operations with mock Neo4j triple store."""
    triple_store_roundtrip(neo4j_triple_store_manager, test_ontology)


def test_neo4j_serialize_facts(neo4j_triple_store_manager):
    """Test serializing facts with mock Neo4j triple store."""
    triple_store_serialize_facts(neo4j_triple_store_manager)


def test_neo4j_serialize_empty_facts(neo4j_triple_store_manager):
    """Test serializing empty facts with mock Neo4j triple store."""
    triple_store_serialize_empty_facts(neo4j_triple_store_manager)


def test_mock_neo4j_initialization():
    """Test that mock Neo4j triple store initializes correctly."""
    manager = MockNeo4jTripleStoreManager(
        uri="bolt://localhost:7687", auth=("neo4j", "password"), clean=True
    )

    assert manager.uri == "bolt://localhost:7687"
    assert manager.auth == ("neo4j", "password")
    assert len(manager.fetch_ontologies()) == 0


def test_mock_neo4j_clean_operation():
    """Test that mock Neo4j triple store clean operation works."""
    manager = MockNeo4jTripleStoreManager(clean=True)

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
    assert result is not None
    assert "nodes_created" in result
    assert len(manager.fetch_ontologies()) == 1

    # Clean and verify
    manager.clear()
    assert len(manager.fetch_ontologies()) == 0
    assert len(manager.graphs) == 0


def test_mock_neo4j_return_format():
    """Test that mock Neo4j returns proper summary format."""
    manager = MockNeo4jTripleStoreManager(clean=True)

    from rdflib import Literal, URIRef
    from rdflib.namespace import RDF, RDFS

    from ontocast.onto.rdfgraph import RDFGraph

    graph = RDFGraph()
    graph.add((URIRef("http://example.org/Person"), RDF.type, RDFS.Class))
    graph.add((URIRef("http://example.org/Person"), RDFS.label, Literal("Person")))

    result = manager.serialize(graph)

    assert isinstance(result, dict)
    assert "nodes_created" in result
    assert "relationships_created" in result
    assert "properties_set" in result
    assert "labels_added" in result
    assert result["nodes_created"] == 2  # Two triples
    assert result["relationships_created"] == 0
    assert result["properties_set"] == 2
    assert result["labels_added"] == 1
