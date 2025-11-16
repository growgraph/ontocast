"""Test for GraphUpdate SPARQL query generation and execution.

This test verifies that GraphUpdate.generate_sparql_queries() generates valid SPARQL
queries that can be executed on RDFGraph instances using rdflib's update() method.
"""

import json
from typing import Any

import pytest
from rdflib import Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import (
    GenericSparqlQuery,
    GraphUpdate,
    TripleOp,
)


def create_jsonld_triples(
    subject_id: str,
    properties: dict[str, Any],
    context: dict[str, str] | None = None,
) -> str:
    """Create a JSON-LD string for triples.

    Args:
        subject_id: The subject ID (e.g., "ex:John")
        properties: Dictionary of properties to add (e.g., {"rdf:type": "ex:Person", "rdfs:label": "John"})
        context: Optional @context dictionary. If None, will infer from properties.

    Returns:
        JSON-LD string representation
    """
    default_context = {
        "ex": "http://example.org/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    }

    if context is None:
        # Infer context from property keys and values
        context = default_context.copy()
        for key in properties.keys():
            if ":" in key:
                prefix = key.split(":")[0]
                if prefix not in context and prefix not in [
                    "rdf",
                    "rdfs",
                    "owl",
                    "xsd",
                ]:
                    # Add a default namespace for unknown prefixes
                    context[prefix] = f"http://example.org/{prefix}#"

        # Also check property values for prefixes
        for value in properties.values():
            if (
                isinstance(value, str)
                and ":" in value
                and not value.startswith("http")
                and not value.startswith('"')
            ):
                prefix = value.split(":")[0]
                if prefix not in context and prefix not in [
                    "rdf",
                    "rdfs",
                    "owl",
                    "xsd",
                ]:
                    context[prefix] = f"http://example.org/{prefix}#"

    # Convert property values that look like URIs to use @id
    processed_properties = {}
    for key, value in properties.items():
        # If value is a string that looks like a prefixed URI (contains ":" but not a full URI or quoted)
        if (
            isinstance(value, str)
            and ":" in value
            and not value.startswith("http")
            and not value.startswith('"')
        ):
            # Treat as URI reference
            processed_properties[key] = {"@id": value}
        else:
            processed_properties[key] = value

    jsonld_data = {"@context": context, "@id": subject_id, **processed_properties}
    return json.dumps(jsonld_data, indent=2)


def create_jsonld_with_language_tag(
    subject_id: str,
    property_key: str,
    value: str,
    language: str,
    context: dict[str, str] | None = None,
) -> str:
    """Create JSON-LD with a language-tagged literal.

    Args:
        subject_id: The subject ID
        property_key: The property key (e.g., "rdfs:label")
        value: The literal value
        language: The language tag (e.g., "en")
        context: Optional @context dictionary

    Returns:
        JSON-LD string representation
    """
    default_context = {
        "ex": "http://example.org/",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    }
    if context is None:
        context = default_context

    jsonld_data = {
        "@context": context,
        "@id": subject_id,
        property_key: {"@value": value, "@language": language},
    }
    return json.dumps(jsonld_data, indent=2)


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

    # Create JSON-LD with language tags using helper function
    # jsonld_label = create_jsonld_with_language_tag(
    #     "ex:Test", "rdfs:label", "Test Label", "en"
    # )
    # jsonld_comment = create_jsonld_with_language_tag(
    #     "ex:Test", "rdfs:comment", "Un commentaire", "fr"
    # )

    # Create a combined JSON-LD with both properties
    combined_jsonld = json.dumps(
        {
            "@context": {
                "ex": "http://example.org/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            },
            "@id": "ex:Test",
            "rdfs:label": {"@value": "Test Label", "@language": "en"},
            "rdfs:comment": {"@value": "Un commentaire", "@language": "fr"},
        },
        indent=2,
    )

    graph_update = GraphUpdate(
        operations=[
            TripleOp(
                type="insert",
                triples=combined_jsonld,  # type: ignore[arg-type]
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


@pytest.mark.parametrize(
    "format_type",
    ["jsonld", "turtle"],
)
def test_graph_update_insert_operation(format_type: str):
    """Test GraphUpdate with TripleOp insert operations using different formats."""
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

    # Create triples in the specified format
    if format_type == "jsonld":
        triples = create_jsonld_triples(
            "ex:John",
            {"rdf:type": "ex:Person", "rdfs:label": "John Doe"},
        )
    else:  # turtle
        triples = """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        
        ex:John a ex:Person ;
            rdfs:label "John Doe" .
        """

    graph_update = GraphUpdate(
        operations=[
            TripleOp(
                type="insert",
                triples=triples,  # type: ignore[arg-type]
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

    # Create GraphUpdate with TripleOp using helper function
    triples = create_jsonld_triples(
        "ex:John",
        {"rdf:type": "ex:Person", "rdfs:label": "John Doe"},
    )

    graph_update = GraphUpdate(
        operations=[
            TripleOp(
                type="delete",
                triples=triples,  # type: ignore[arg-type]
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

    # Create GraphUpdate with custom prefixes using helper function
    triples = create_jsonld_triples(
        "ex:John",
        {"rdf:type": "ex:Person", "schema:name": "John Doe"},
        context={
            "ex": "http://example.org/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "schema": "https://schema.org/",
        },
    )

    graph_update = GraphUpdate(
        operations=[
            TripleOp(
                type="insert",
                triples=triples,  # type: ignore[arg-type]
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

    # Create GraphUpdate with mixed operations using helper functions
    insert_jane = create_jsonld_triples(
        "ex:Jane",
        {"rdf:type": "ex:Person", "schema:name": "Jane Smith"},
        context={
            "ex": "http://example.org/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "schema": "https://schema.org/",
        },
    )
    delete_john_label = create_jsonld_triples(
        "ex:John",
        {"rdfs:label": "John Doe"},
    )
    insert_john_label = create_jsonld_triples(
        "ex:John",
        {"rdfs:label": "John Updated"},
    )

    graph_update = GraphUpdate(
        operations=[
            # First: Insert new person with custom schema prefix
            TripleOp(
                type="insert",
                triples=insert_jane,  # type: ignore[arg-type]
                prefixes={
                    "ex": "http://example.org/",
                    "schema": "https://schema.org/",
                },
            ),
            # Second: Delete John's label
            TripleOp(
                type="delete",
                triples=delete_john_label,  # type: ignore[arg-type]
                prefixes={"ex": "http://example.org/"},
            ),
            # Third: Insert new label for John
            TripleOp(
                type="insert",
                triples=insert_john_label,  # type: ignore[arg-type]
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
    # Note: GenericSparqlQuery handles its own prefix declarations
    graph_update = GraphUpdate(
        operations=[
            GenericSparqlQuery(
                query="PREFIX ex: <http://example.org/>\nPREFIX schema: <https://schema.org/>\nPREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nINSERT { ex:John schema:age 30 } WHERE { ex:John rdf:type ex:Person }"
            ),
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate one query
    assert len(queries) == 1

    # Verify the query includes the custom SPARQL with prefixes
    assert "INSERT { ex:John schema:age 30 }" in queries[0]
    assert "WHERE { ex:John rdf:type ex:Person }" in queries[0]

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
            TripleOp(type="insert", triples=RDFGraph()),
            TripleOp(type="delete", triples=RDFGraph()),
        ]
    )

    # Generate SPARQL queries
    queries = graph_update.generate_sparql_queries()

    # Should generate no queries (empty triples are skipped)
    assert len(queries) == 0

    # Graph should remain unchanged
    assert len(graph) == initial_triple_count
