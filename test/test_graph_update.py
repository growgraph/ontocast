"""Test for GraphUpdate SPARQL query generation and execution.

This test verifies that GraphUpdate.generate_sparql_queries() generates valid SPARQL
queries that can be executed on RDFGraph instances using rdflib's update() method.
"""

from rdflib import Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import (
    GraphUpdate,
    TripleOp,
)


def test_rdfgraph_recovers_dangling_semicolon_at_eof() -> None:
    """RDFGraph should recover from common LLM-truncated Turtle at EOF."""
    ttl = """
    @prefix ex: <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

    ex:Case85_968 a ex:Appeal ;
        ex:appealsTo ex:Cassation ;
    """

    graph = RDFGraph._from_turtle_str(ttl)

    assert len(graph) == 2
    assert (
        URIRef("http://example.org/Case85_968"),
        URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        URIRef("http://example.org/Appeal"),
    ) in graph
    assert (
        URIRef("http://example.org/Case85_968"),
        URIRef("http://example.org/appealsTo"),
        URIRef("http://example.org/Cassation"),
    ) in graph


def test_graph_update_with_language_tags():
    """Test GraphUpdate with language-tagged literals."""
    # Create initial RDFGraph
    graph = RDFGraph._from_turtle_str(
        """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .
        
        ex:Test a rdfs:Class .
        """
    )

    initial_triple_count = len(graph)

    # Create Turtle with language-tagged literals
    triples = """
    @prefix ex: <http://example.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    
    ex:Test rdfs:label "Test Label"@en ;
        rdfs:comment "Un commentaire"@fr .
    """

    graph_update = GraphUpdate(
        triple_operations=[
            TripleOp(
                type="insert",
                graph=RDFGraph._from_turtle_str(triples),
                prefixes={"ex": "http://example.org/"},
            )
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate one query
    assert len(queries) == 1

    # Execute the query on the graph
    graph.update(queries[0])

    # Verify new triples were added
    assert len(graph) == initial_triple_count + 2


def test_graph_update_insert_operation():
    """Test GraphUpdate with TripleOp insert operations using Turtle format."""
    # Create initial RDFGraph
    graph = RDFGraph._from_turtle_str(
        """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        
        ex:Person a rdfs:Class ;
            rdfs:label "Person" .
        """
    )

    initial_triple_count = len(graph)

    # Create triples in Turtle format
    triples = """
    @prefix ex: <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    
    ex:John a ex:Person ;
        rdfs:label "John Doe" .
    """

    graph_update = GraphUpdate(
        triple_operations=[
            TripleOp(
                type="insert",
                graph=RDFGraph._from_turtle_str(triples),
                prefixes={"ex": "http://example.org/"},
            )
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate one query
    assert len(queries) == 1

    # Execute the query on the graph
    graph.update(queries[0])

    # Verify new triples were added
    assert len(graph) == initial_triple_count + 2
    assert (
        URIRef("http://example.org/John"),
        URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        URIRef("http://example.org/Person"),
    ) in graph
    assert (
        URIRef("http://example.org/John"),
        URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
        Literal("John Doe"),
    ) in graph


def test_graph_update_extract_insert_graph() -> None:
    """Test GraphUpdate.extract_insert_graph returns only insert triples."""
    insert_ttl = """
    @prefix ex: <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ex:Person a rdfs:Class .
    ex:Person rdfs:label "Person" .
    """
    delete_ttl = """
    @prefix ex: <http://example.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ex:Obsolete a rdfs:Class .
    """
    gu = GraphUpdate(
        triple_operations=[
            TripleOp(type="insert", graph=RDFGraph._from_turtle_str(insert_ttl)),
            TripleOp(type="delete", graph=RDFGraph._from_turtle_str(delete_ttl)),
        ]
    )
    insert_graph = gu.extract_insert_graph()
    assert len(insert_graph) == 2
    person_uri = URIRef("http://example.org/Person")
    rdf_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    rdfs_class = URIRef("http://www.w3.org/2000/01/rdf-schema#Class")
    rdfs_label = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
    assert (person_uri, rdf_type, rdfs_class) in insert_graph
    assert (person_uri, rdfs_label, Literal("Person")) in insert_graph
    obsolete_uri = URIRef("http://example.org/Obsolete")
    assert (obsolete_uri, rdf_type, rdfs_class) not in insert_graph


def test_graph_update_delete_operation():
    """Test GraphUpdate with TripleOp delete operations."""
    # Create RDFGraph with existing triples
    graph = RDFGraph._from_turtle_str(
        """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        
        ex:Person a rdfs:Class ;
            rdfs:label "Person" .
        
        ex:John a ex:Person ;
            rdfs:label "John Doe" .
        
        ex:Jane a ex:Person ;
            rdfs:label "Jane Smith" .
        """
    )

    initial_triple_count = len(graph)

    # Create GraphUpdate with TripleOp using Turtle format
    triples = """
    @prefix ex: <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    
    ex:John a ex:Person ;
        rdfs:label "John Doe" .
    """

    graph_update = GraphUpdate(
        triple_operations=[
            TripleOp(
                type="delete",
                graph=RDFGraph._from_turtle_str(triples),
                prefixes={"ex": "http://example.org/"},
            )
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate one query
    assert len(queries) == 1

    # Execute the query on the graph
    graph.update(queries[0])

    # Verify triples were removed
    assert len(graph) == initial_triple_count - 2
    assert (
        URIRef("http://example.org/John"),
        URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        URIRef("http://example.org/Person"),
    ) not in graph
    assert (
        URIRef("http://example.org/John"),
        URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
        Literal("John Doe"),
    ) not in graph
    # Jane should still be there
    assert (
        URIRef("http://example.org/Jane"),
        URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        URIRef("http://example.org/Person"),
    ) in graph


def test_graph_update_with_prefixes():
    """Test GraphUpdate with TripleOp operations that declare custom prefixes."""
    # Create initial RDFGraph
    graph = RDFGraph._from_turtle_str(
        """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        
        ex:Person a rdf:Class .
        """
    )

    initial_triple_count = len(graph)

    # Create GraphUpdate with custom prefixes using Turtle format
    triples = """
    @prefix ex: <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix schema: <https://schema.org/> .
    
    ex:John a ex:Person ;
        schema:name "John Doe" .
    """

    graph_update = GraphUpdate(
        triple_operations=[
            TripleOp(
                type="insert",
                graph=RDFGraph._from_turtle_str(triples),
                prefixes={
                    "ex": "http://example.org/",
                    "schema": "https://schema.org/",
                },
            ),
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate one query
    assert len(queries) == 1

    # Verify the query includes PREFIX declarations
    assert "PREFIX schema: <https://schema.org/>" in queries[0]

    # Execute the query on the graph
    graph.update(queries[0])

    # Verify new triples were added
    assert len(graph) == initial_triple_count + 2
    assert (
        URIRef("http://example.org/John"),
        URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        URIRef("http://example.org/Person"),
    ) in graph
    assert (
        URIRef("http://example.org/John"),
        URIRef("https://schema.org/name"),
        Literal("John Doe"),
    ) in graph


def test_graph_update_mixed_operations_ordered():
    """Test GraphUpdate with mixed operations in specific order."""
    # Create initial RDFGraph
    graph = RDFGraph._from_turtle_str(
        """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        
        ex:Person a rdfs:Class ;
            rdfs:label "Person" .
        
        ex:John a ex:Person ;
            rdfs:label "John Doe" .
        """
    )

    initial_triple_count = len(graph)

    # Create GraphUpdate with mixed operations using Turtle format
    insert_jane = """
    @prefix ex: <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix schema: <https://schema.org/> .
    
    ex:Jane a ex:Person ;
        schema:name "Jane Smith" .
    """
    delete_john_label = """
    @prefix ex: <http://example.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    
    ex:John rdfs:label "John Doe" .
    """
    insert_john_label = """
    @prefix ex: <http://example.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    
    ex:John rdfs:label "John Updated" .
    """

    graph_update = GraphUpdate(
        triple_operations=[
            # First: Insert new person with custom schema prefix
            TripleOp(
                type="insert",
                graph=RDFGraph._from_turtle_str(insert_jane),
                prefixes={
                    "ex": "http://example.org/",
                    "schema": "https://schema.org/",
                },
            ),
            # Second: Delete John's label
            TripleOp(
                type="delete",
                graph=RDFGraph._from_turtle_str(delete_john_label),
                prefixes={"ex": "http://example.org/"},
            ),
            # Third: Insert new label for John
            TripleOp(
                type="insert",
                graph=RDFGraph._from_turtle_str(insert_john_label),
                prefixes={"ex": "http://example.org/"},
            ),
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate 3 queries (one for each TripleOp)
    assert len(queries) == 3

    # Execute queries in order
    for query in queries:
        graph.update(query)

    # Verify final state
    # Should have: 4 initial + 2 added (Jane) - 1 deleted (John's old label) + 1 added (John's new label) = 6 triples
    assert (
        len(graph) == initial_triple_count + 2
    )  # +2 net change: +2 for Jane, -1 for John's old label, +1 for John's new label

    # Verify John's label was updated
    assert (
        URIRef("http://example.org/John"),
        URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
        Literal("John Updated"),
    ) in graph
    assert (
        URIRef("http://example.org/John"),
        URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
        Literal("John Doe"),
    ) not in graph

    # Verify Jane was added
    assert (
        URIRef("http://example.org/Jane"),
        URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        URIRef("http://example.org/Person"),
    ) in graph
    assert (
        URIRef("http://example.org/Jane"),
        URIRef("https://schema.org/name"),
        Literal("Jane Smith"),
    ) in graph


def test_graph_update_empty_operations():
    """Test GraphUpdate with empty operations list."""
    graph = RDFGraph._from_turtle_str(
        """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        
        ex:Person a rdf:Class .
        """
    )

    initial_triple_count = len(graph)

    # Create GraphUpdate with empty operations
    graph_update = GraphUpdate(triple_operations=[])

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate no queries
    assert len(queries) == 0

    # Graph should remain unchanged
    assert len(graph) == initial_triple_count


def test_graph_update_operations_with_empty_triples():
    """Test GraphUpdate with operations that have empty triples lists."""
    graph = RDFGraph._from_turtle_str(
        """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        
        ex:Person a rdf:Class .
        """
    )

    initial_triple_count = len(graph)

    # Create GraphUpdate with operations that have empty triples
    graph_update = GraphUpdate(
        triple_operations=[
            TripleOp(type="insert", graph=RDFGraph()),
            TripleOp(type="delete", graph=RDFGraph()),
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate no queries (empty triples are skipped)
    assert len(queries) == 0

    # Graph should remain unchanged
    assert len(graph) == initial_triple_count
