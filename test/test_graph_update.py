"""Test for GraphUpdate SPARQL query generation and execution.

This test verifies that GraphUpdate.generate_sparql_queries() generates valid SPARQL
queries that can be executed on RDFGraph instances using rdflib's update() method.
"""

from rdflib import Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import (
    AddPrefixOp,
    DeleteOp,
    GenericSparqlQuery,
    GraphUpdate,
    InsertOp,
    Triple,
)


def test_graph_update_insert_operation():
    """Test GraphUpdate with InsertOp operations."""
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

    # Create GraphUpdate with InsertOp
    graph_update = GraphUpdate(
        operations=[
            InsertOp(
                triples=[
                    Triple(subject="ex:John", predicate="rdf:type", object="ex:Person"),
                    Triple(
                        subject="ex:John", predicate="rdfs:label", object='"John Doe"'
                    ),
                ]
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


def test_graph_update_delete_operation():
    """Test GraphUpdate with DeleteOp operations."""
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

    # Create GraphUpdate with DeleteOp
    graph_update = GraphUpdate(
        operations=[
            DeleteOp(
                triples=[
                    Triple(subject="ex:John", predicate="rdf:type", object="ex:Person"),
                    Triple(
                        subject="ex:John", predicate="rdfs:label", object='"John Doe"'
                    ),
                ]
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
    """Test GraphUpdate with AddPrefixOp operations."""
    # Create initial RDFGraph
    graph = RDFGraph._from_turtle_str(
        """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        
        ex:Person a rdf:Class .
        """
    )

    initial_triple_count = len(graph)

    # Create GraphUpdate with AddPrefixOp and InsertOp using the prefix
    graph_update = GraphUpdate(
        operations=[
            AddPrefixOp(prefix="schema", namespace_uri="https://schema.org/"),
            InsertOp(
                triples=[
                    Triple(subject="ex:John", predicate="rdf:type", object="ex:Person"),
                    Triple(
                        subject="ex:John", predicate="schema:name", object='"John Doe"'
                    ),
                ]
            ),
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate one query (AddPrefixOp doesn't generate a separate query)
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

    # Create GraphUpdate with mixed operations in specific order
    graph_update = GraphUpdate(
        operations=[
            # First: Add prefix
            AddPrefixOp(prefix="schema", namespace_uri="https://schema.org/"),
            # Second: Insert new person
            InsertOp(
                triples=[
                    Triple(subject="ex:Jane", predicate="rdf:type", object="ex:Person"),
                    Triple(
                        subject="ex:Jane",
                        predicate="schema:name",
                        object='"Jane Smith"',
                    ),
                ]
            ),
            # Third: Delete John's label
            DeleteOp(
                triples=[
                    Triple(
                        subject="ex:John", predicate="rdfs:label", object='"John Doe"'
                    )
                ]
            ),
            # Fourth: Insert new label for John
            InsertOp(
                triples=[
                    Triple(
                        subject="ex:John",
                        predicate="rdfs:label",
                        object='"John Updated"',
                    )
                ]
            ),
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate 3 queries (AddPrefixOp doesn't generate a separate query)
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


def test_graph_update_generic_sparql_query():
    """Test GraphUpdate with GenericSparqlQuery operation."""
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

    # Create GraphUpdate with GenericSparqlQuery
    graph_update = GraphUpdate(
        operations=[
            AddPrefixOp(prefix="schema", namespace_uri="https://schema.org/"),
            GenericSparqlQuery(
                query="INSERT { ex:John schema:age 30 } WHERE { ex:John rdf:type ex:Person }"
            ),
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate one query
    assert len(queries) == 1

    # Verify the query includes the custom SPARQL
    assert (
        "INSERT { ex:John schema:age 30 } WHERE { ex:John rdf:type ex:Person }"
        in queries[0]
    )
    assert "PREFIX schema: <https://schema.org/>" in queries[0]

    # Execute the query on the graph
    graph.update(queries[0])

    # Verify the custom query was executed
    assert len(graph) == initial_triple_count + 1
    assert (
        URIRef("http://example.org/John"),
        URIRef("https://schema.org/age"),
        Literal(30),
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
    graph_update = GraphUpdate(operations=[])

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
        operations=[
            InsertOp(triples=[]),
            DeleteOp(triples=[]),
            AddPrefixOp(prefix="schema", namespace_uri="https://schema.org/"),
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate no queries (empty triples are skipped)
    assert len(queries) == 0

    # Graph should remain unchanged
    assert len(graph) == initial_triple_count
