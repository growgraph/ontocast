"""Test validation and aggregation functionality."""

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import FOAF, RDF, RDFS, SKOS

from ontocast.onto.chunk import Chunk
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.aggregate import ChunkRDFGraphAggregator
from ontocast.tool.validate import (
    RDFGraphConnectivityValidator,
    validate_and_connect_chunk,
)


def create_sample_chunk_graph(current_domain, chunk_id: str) -> Chunk:
    """Create a sample chunk graph for testing.

    Args:
        current_domain: The domain to use for URIs.
        chunk_id: The chunk identifier.

    Returns:
        Chunk: A sample chunk with a graph.
    """
    g = RDFGraph()

    ttl = f"""
        @prefix ns1: <https://example.com/doc/123/chunk/{chunk_id}/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ns1:person1 rdfs:label "John Doe" ;
            ns1:knows ns1:person2 .
        ns1:person3 rdfs:label "Alexander Bell" .
        ns1:person2 rdfs:label "Jane Smith" .
    """
    g.parse(data=ttl, format="turtle")

    doc_iri = f"{current_domain}/doc/123"

    c = Chunk(graph=g, doc_iri=doc_iri, text="", hid=chunk_id)
    return c


@pytest.fixture
def doc_id():
    """Fixture for document ID."""
    return "123"


@pytest.fixture
def sample_chunks(current_domain):
    """Fixture for sample chunks."""
    ids = ["abc123", "def456"]
    sample_chunks = [
        create_sample_chunk_graph(chunk_id=i, current_domain=current_domain)
        for i in ids
    ]
    return sample_chunks


@pytest.fixture
def connected_chunks(sample_chunks):
    """Fixture for connected chunks."""
    connected_chunks = []
    for chunk in sample_chunks:
        new_chunk = validate_and_connect_chunk(chunk, auto_connect=True)
        connected_chunks.append(new_chunk)
    return connected_chunks


def test_validation(sample_chunks):
    """Test basic validation functionality."""
    gs = []
    for chunk in sample_chunks:
        chunk.sanitize()
        new_chunk = validate_and_connect_chunk(chunk, auto_connect=True)
        gs += [new_chunk]

    assert [len(x.graph) for x in gs] == [10, 10]


def test_aggregation(doc_id, connected_chunks, current_domain):
    """Test graph aggregation functionality."""
    # Aggregate graphs (now using connected versions)
    aggregator = ChunkRDFGraphAggregator()
    for chunk in connected_chunks:
        chunk.sanitize()
    aggregated_graph = aggregator.aggregate_graphs(
        chunks=connected_chunks, doc_namespace=f"{current_domain}/{doc_id}/"
    )

    # Validate aggregated graph connectivity
    connectivity_result = RDFGraphConnectivityValidator(
        aggregated_graph
    ).validate_connectivity()
    assert len(aggregated_graph) == 25
    assert connectivity_result.num_components == 1


def create_test_chunks_basic_similarity(current_domain):
    """Test basic entity and predicate similarity detection.

    Args:
        current_domain: The domain to use for URIs.

    Returns:
        tuple: List of chunks and document IRI.
    """
    chunks = []

    # Chunk 1: Person entities with similar names
    g1 = RDFGraph()
    doc_iri = f"{current_domain}/doc/test1"
    c1 = Chunk(
        graph=g1, doc_iri=doc_iri, text="First chunk about John", hid="chunk_001"
    )

    person1 = URIRef(c1.namespace + "john_doe")
    person2 = URIRef(c1.namespace + "jane_smith")
    company1 = URIRef(c1.namespace + "acme_corp")

    g1.add((person1, RDFS.label, Literal("John Doe")))
    g1.add((person1, RDF.type, FOAF.Person))
    g1.add((person1, URIRef(c1.namespace + "worksAt"), company1))
    g1.add((person2, RDFS.label, Literal("Jane Smith")))
    g1.add((person2, RDF.type, FOAF.Person))
    g1.add((company1, RDFS.label, Literal("ACME Corporation")))
    g1.add((company1, RDF.type, URIRef(c1.namespace + "Company")))

    chunks.append(c1)

    # Chunk 2: Similar entities with slight variations
    g2 = RDFGraph()
    c2 = Chunk(
        graph=g2, doc_iri=doc_iri, text="Second chunk about John", hid="chunk_002"
    )

    person1_var = URIRef(c2.namespace + "john_d")
    person3 = URIRef(c2.namespace + "bob_johnson")
    company1_var = URIRef(c2.namespace + "acme_company")

    g2.add((person1_var, RDFS.label, Literal("John D.")))
    g2.add((person1_var, RDF.type, FOAF.Person))
    g2.add((person1_var, URIRef(c2.namespace + "employedBy"), company1_var))
    g2.add((person3, RDFS.label, Literal("Robert Johnson")))
    g2.add((person3, RDF.type, FOAF.Person))
    g2.add((company1_var, RDFS.label, Literal("ACME Corp")))
    g2.add((company1_var, RDF.type, URIRef(c2.namespace + "Organization")))

    chunks.append(c2)

    # Chunk 3: More variations and edge cases
    g3 = RDFGraph()
    c3 = Chunk(
        graph=g3, doc_iri=doc_iri, text="Third chunk with variations", hid="chunk_003"
    )

    person1_var2 = URIRef(c3.namespace + "j_doe")
    person4 = URIRef(c3.namespace + "jane_s")
    skill1 = URIRef(c3.namespace + "programming")

    g3.add((person1_var2, RDFS.label, Literal("J. Doe")))
    g3.add((person1_var2, RDFS.comment, Literal("Software developer")))
    g3.add((person1_var2, RDF.type, FOAF.Person))
    g3.add((person1_var2, URIRef(c3.namespace + "hasSkill"), skill1))
    g3.add((person4, RDFS.label, Literal("Jane S.")))
    g3.add((person4, RDF.type, FOAF.Person))
    g3.add((skill1, RDFS.label, Literal("Programming")))
    g3.add((skill1, RDF.type, SKOS.Concept))

    chunks.append(c3)

    return chunks, doc_iri


def create_test_chunks_predicate_disambiguation(current_domain):
    """Test predicate disambiguation with similar properties.

    Args:
        current_domain: The domain to use for URIs.

    Returns:
        tuple: List of chunks and document IRI.
    """
    chunks = []

    # Chunk 1: Various relationship predicates
    g1 = RDFGraph()
    doc_iri = f"{current_domain}/doc/test2"
    c1 = Chunk(graph=g1, doc_iri=doc_iri, text="Relationships chunk 1", hid="chunk_101")

    person1 = URIRef(c1.namespace + "alice")
    person2 = URIRef(c1.namespace + "bob")
    knows_pred = URIRef(c1.namespace + "knows")
    friend_pred = URIRef(c1.namespace + "friendOf")

    g1.add((person1, RDFS.label, Literal("Alice Johnson")))
    g1.add((person2, RDFS.label, Literal("Bob Wilson")))
    g1.add((person1, knows_pred, person2))
    g1.add((person1, friend_pred, person2))

    # Add metadata for predicates
    g1.add((knows_pred, RDFS.label, Literal("knows")))
    g1.add((knows_pred, RDFS.domain, FOAF.Person))
    g1.add((knows_pred, RDFS.range, FOAF.Person))
    g1.add((friend_pred, RDFS.label, Literal("friend of")))
    g1.add((friend_pred, RDF.type, RDF.Property))

    chunks.append(c1)

    # Chunk 2: Similar predicates with variations
    g2 = RDFGraph()
    c2 = Chunk(graph=g2, doc_iri=doc_iri, text="Relationships chunk 2", hid="chunk_102")

    person3 = URIRef(c2.namespace + "charlie")
    person4 = URIRef(c2.namespace + "diana")
    knows_var = URIRef(c2.namespace + "knowsAbout")
    acquainted = URIRef(c2.namespace + "acquaintedWith")

    g2.add((person3, RDFS.label, Literal("Charlie Brown")))
    g2.add((person4, RDFS.label, Literal("Diana Prince")))
    g2.add((person3, knows_var, person4))
    g2.add((person3, acquainted, person4))

    # Add metadata for predicates
    g2.add((knows_var, RDFS.label, Literal("knows about")))
    g2.add((knows_var, RDFS.comment, Literal("Indicates knowledge or familiarity")))
    g2.add((acquainted, RDFS.label, Literal("acquainted with")))
    g2.add((acquainted, RDFS.domain, FOAF.Person))

    chunks.append(c2)

    # Chunk 3: Different domain predicates (should not be merged)
    g3 = RDFGraph()
    c3 = Chunk(
        graph=g3, doc_iri=doc_iri, text="Different domain chunk", hid="chunk_103"
    )

    book1 = URIRef(c3.namespace + "book1")
    author1 = URIRef(c3.namespace + "author1")
    knows_book = URIRef(c3.namespace + "knows")  # Same name, different domain

    g3.add((book1, RDFS.label, Literal("Programming Guide")))
    g3.add((author1, RDFS.label, Literal("Expert Author")))
    g3.add((author1, knows_book, book1))  # Author knows (about) book

    # Different domain - should not merge with person knows
    g3.add((knows_book, RDFS.label, Literal("knows")))
    g3.add((knows_book, RDFS.domain, URIRef(c3.namespace + "Author")))
    g3.add((knows_book, RDFS.range, URIRef(c3.namespace + "Book")))

    chunks.append(c3)

    return chunks, doc_iri


def create_test_chunks_edge_cases(current_domain):
    """Test edge cases: exact matches, no labels, special characters.

    Args:
        current_domain: The domain to use for URIs.

    Returns:
        tuple: List of chunks and document IRI.
    """
    chunks = []

    # Chunk 1: Entities without labels (using local names)
    g1 = RDFGraph()
    doc_iri = f"{current_domain}/doc/test3"
    c1 = Chunk(graph=g1, doc_iri=doc_iri, text="Edge case chunk 1", hid="chunk_201")

    entity1 = URIRef(c1.namespace + "mysterious_entity")
    entity2 = URIRef(c1.namespace + "unknown_thing")
    relation1 = URIRef(c1.namespace + "weird-relation")

    # No explicit labels - should use local names
    g1.add((entity1, relation1, entity2))
    g1.add((entity1, RDF.type, URIRef(c1.namespace + "Thing")))

    chunks.append(c1)

    # Chunk 2: Exact URI matches (should be trivially merged)
    g2 = RDFGraph()
    c2 = Chunk(graph=g2, doc_iri=doc_iri, text="Edge case chunk 2", hid="chunk_202")

    # Intentionally same URIs as chunk 1 (simulating perfect matches)
    entity1_exact = URIRef(c1.namespace + "mysterious_entity")  # Exact match
    entity3 = URIRef(c2.namespace + "another_entity")
    relation1_exact = URIRef(c1.namespace + "weird-relation")  # Exact match

    g2.add((entity1_exact, RDFS.label, Literal("Mysterious Entity")))  # Now has label
    g2.add((entity1_exact, relation1_exact, entity3))
    g2.add((entity3, RDFS.label, Literal("Another Entity")))

    chunks.append(c2)

    # Chunk 3: Special characters and Unicode
    g3 = RDFGraph()
    c3 = Chunk(
        graph=g3, doc_iri=doc_iri, text="Special characters chunk", hid="chunk_203"
    )

    entity_unicode = URIRef(c3.namespace + "café_owner")
    entity_special = URIRef(c3.namespace + "company@location")
    relation_special = URIRef(c3.namespace + "works@")

    g3.add((entity_unicode, RDFS.label, Literal("Café Owner")))
    g3.add((entity_special, RDFS.label, Literal("Company @ Location")))
    g3.add((entity_unicode, relation_special, entity_special))

    # Similar entity with cleaned name
    entity_unicode_var = URIRef(c3.namespace + "cafe_owner")
    g3.add((entity_unicode_var, RDFS.label, Literal("Cafe Owner")))

    chunks.append(c3)

    return chunks, doc_iri


def create_test_chunks_type_disambiguation(current_domain):
    """Test type-based disambiguation.

    Args:
        current_domain: The domain to use for URIs.

    Returns:
        tuple: List of chunks and document IRI.
    """
    chunks = []

    # Chunk 1: Person named "Apple"
    g1 = RDFGraph()
    doc_iri = f"{current_domain}/doc/test4"
    c1 = Chunk(graph=g1, doc_iri=doc_iri, text="Person Apple chunk", hid="chunk_301")

    apple_person = URIRef(c1.namespace + "apple")
    john = URIRef(c1.namespace + "john")

    g1.add((apple_person, RDFS.label, Literal("Apple Johnson")))
    g1.add((apple_person, RDF.type, FOAF.Person))
    g1.add((john, RDFS.label, Literal("John Smith")))
    g1.add((john, RDF.type, FOAF.Person))
    g1.add((john, URIRef(c1.namespace + "knows"), apple_person))

    chunks.append(c1)

    # Chunk 2: Company named "Apple"
    g2 = RDFGraph()
    c2 = Chunk(graph=g2, doc_iri=doc_iri, text="Company Apple chunk", hid="chunk_302")

    apple_company = URIRef(c2.namespace + "apple")  # Same local name!
    employee = URIRef(c2.namespace + "employee1")

    g2.add((apple_company, RDFS.label, Literal("Apple Inc.")))
    g2.add((apple_company, RDF.type, URIRef(c2.namespace + "Company")))
    g2.add((employee, RDFS.label, Literal("Jane Doe")))
    g2.add((employee, RDF.type, FOAF.Person))
    g2.add((employee, URIRef(c2.namespace + "worksFor"), apple_company))

    chunks.append(c2)

    # Chunk 3: Fruit "apple" (no specific type)
    g3 = RDFGraph()
    c3 = Chunk(graph=g3, doc_iri=doc_iri, text="Fruit apple chunk", hid="chunk_303")

    apple_fruit = URIRef(c3.namespace + "apple")  # Same local name again!
    color_red = URIRef(c3.namespace + "red")

    g3.add((apple_fruit, RDFS.label, Literal("Apple")))
    g3.add((apple_fruit, RDF.type, URIRef(c3.namespace + "Fruit")))
    g3.add((apple_fruit, URIRef(c3.namespace + "hasColor"), color_red))
    g3.add((color_red, RDFS.label, Literal("Red")))

    chunks.append(c3)

    return chunks, doc_iri


def create_test_chunks_large_scale(current_domain):
    """Test performance with larger number of entities and relationships.

    Args:
        current_domain: The domain to use for URIs.

    Returns:
        tuple: List of chunks and document IRI.
    """
    chunks = []
    doc_iri = f"{current_domain}/doc/test5"

    # Create 5 chunks with overlapping entities
    for chunk_num in range(5):
        g = RDFGraph()
        chunk_id = f"chunk_{400 + chunk_num:03d}"
        c = Chunk(
            graph=g,
            doc_iri=doc_iri,
            text=f"Large scale chunk {chunk_num}",
            hid=chunk_id,
        )

        # Create 20 entities per chunk with some overlap
        for i in range(20):
            entity_id = (chunk_num * 15 + i) % 50  # Creates overlap
            entity = URIRef(c.namespace + f"entity_{entity_id:03d}")

            # Add variations of the same logical entity across chunks
            if entity_id % 10 == 0:  # Every 10th entity gets variations
                labels = [
                    f"Entity {entity_id}",
                    f"Entity-{entity_id}",
                    f"Entity_{entity_id}",
                    f"Entity#{entity_id}",
                    f"Ent {entity_id}",
                ]
                label = labels[chunk_num % len(labels)]
            else:
                label = f"Entity {entity_id}"

            g.add((entity, RDFS.label, Literal(label)))
            g.add((entity, RDF.type, URIRef(c.namespace + f"Type{entity_id % 5}")))

            # Add relationships
            if i > 0:
                prev_entity = URIRef(
                    c.namespace + f"entity_{((chunk_num * 15 + i - 1) % 50):03d}"
                )
                relation_name = "relatedTo" if i % 2 == 0 else "connectedTo"
                relation = URIRef(c.namespace + relation_name)
                g.add((entity, relation, prev_entity))

                # Add predicate metadata occasionally
                if i == 1:
                    g.add(
                        (
                            relation,
                            RDFS.label,
                            Literal(relation_name.replace("To", " to")),
                        )
                    )

        chunks.append(c)

    return chunks, doc_iri


def create_test_chunks_complex_predicates(current_domain):
    """Test complex predicate scenarios with inheritance and similar meanings.

    Args:
        current_domain: The domain to use for URIs.

    Returns:
        tuple: List of chunks and document IRI.
    """
    chunks = []

    # Chunk 1: Family relationships
    g1 = RDFGraph()
    doc_iri = f"{current_domain}/doc/test6"
    c1 = Chunk(graph=g1, doc_iri=doc_iri, text="Family relationships", hid="chunk_501")

    father = URIRef(c1.namespace + "john_senior")
    son = URIRef(c1.namespace + "john_junior")
    daughter = URIRef(c1.namespace + "mary")

    parent_of = URIRef(c1.namespace + "parentOf")
    father_of = URIRef(c1.namespace + "fatherOf")
    child_of = URIRef(c1.namespace + "childOf")

    g1.add((father, RDFS.label, Literal("John Senior")))
    g1.add((son, RDFS.label, Literal("John Junior")))
    g1.add((daughter, RDFS.label, Literal("Mary Johnson")))

    g1.add((father, parent_of, son))
    g1.add((father, father_of, son))
    g1.add((father, parent_of, daughter))
    g1.add((son, child_of, father))

    # Predicate metadata
    g1.add((parent_of, RDFS.label, Literal("parent of")))
    g1.add((parent_of, RDFS.domain, FOAF.Person))
    g1.add((parent_of, RDFS.range, FOAF.Person))
    g1.add((father_of, RDFS.label, Literal("father of")))
    g1.add((father_of, RDFS.domain, FOAF.Person))
    g1.add((child_of, RDFS.label, Literal("child of")))

    chunks.append(c1)

    # Chunk 2: Similar family relationships with different naming
    g2 = RDFGraph()
    c2 = Chunk(
        graph=g2, doc_iri=doc_iri, text="More family relationships", hid="chunk_502"
    )

    mother = URIRef(c2.namespace + "susan")
    daughter2 = URIRef(c2.namespace + "alice")

    parent_rel = URIRef(c2.namespace + "isParentOf")  # Similar to parentOf
    mother_of = URIRef(c2.namespace + "motherOf")
    offspring = URIRef(c2.namespace + "hasOffspring")  # Similar meaning to parentOf

    g2.add((mother, RDFS.label, Literal("Susan Wilson")))
    g2.add((daughter2, RDFS.label, Literal("Alice Wilson")))

    g2.add((mother, parent_rel, daughter2))
    g2.add((mother, mother_of, daughter2))
    g2.add((mother, offspring, daughter2))

    # Predicate metadata
    g2.add((parent_rel, RDFS.label, Literal("is parent of")))
    g2.add((parent_rel, RDFS.domain, FOAF.Person))
    g2.add((parent_rel, RDFS.range, FOAF.Person))
    g2.add((mother_of, RDFS.label, Literal("mother of")))
    g2.add((offspring, RDFS.label, Literal("has offspring")))
    g2.add((offspring, RDFS.comment, Literal("Indicates parental relationship")))

    chunks.append(c2)

    return chunks, doc_iri


@pytest.fixture
def basic_similarity_chunks(current_domain):
    """Fixture for basic similarity test chunks."""
    chunks, doc_iri = create_test_chunks_basic_similarity(current_domain)
    return chunks, doc_iri


@pytest.fixture
def predicate_disambiguation_chunks(current_domain):
    """Fixture for predicate disambiguation test chunks."""
    chunks, doc_iri = create_test_chunks_predicate_disambiguation(current_domain)
    return chunks, doc_iri


@pytest.fixture
def edge_cases_chunks(current_domain):
    """Fixture for edge cases test chunks."""
    chunks, doc_iri = create_test_chunks_edge_cases(current_domain)
    return chunks, doc_iri


@pytest.fixture
def type_disambiguation_chunks(current_domain):
    """Fixture for type-based disambiguation test chunks."""
    chunks, doc_iri = create_test_chunks_type_disambiguation(current_domain)
    return chunks, doc_iri


@pytest.fixture
def large_scale_chunks(current_domain):
    """Fixture for large scale test chunks."""
    chunks, doc_iri = create_test_chunks_large_scale(current_domain)
    return chunks, doc_iri


@pytest.fixture
def complex_predicates_chunks(current_domain):
    """Fixture for complex predicates test chunks."""
    chunks, doc_iri = create_test_chunks_complex_predicates(current_domain)
    return chunks, doc_iri


def test_basic_similarity_aggregation(basic_similarity_chunks):
    """Test aggregation with basic similarity scenarios."""
    chunks, doc_iri = basic_similarity_chunks
    aggregator = ChunkRDFGraphAggregator()
    aggregated_graph = aggregator.aggregate_graphs(chunks=chunks, doc_namespace=doc_iri)

    # Validate aggregated graph connectivity
    connectivity_result = RDFGraphConnectivityValidator(
        aggregated_graph
    ).validate_connectivity()

    # Check that similar entities are merged
    assert len(aggregated_graph) > 0
    assert connectivity_result.num_components > 0


def test_predicate_disambiguation_aggregation(predicate_disambiguation_chunks):
    """Test aggregation with predicate disambiguation scenarios."""
    chunks, doc_iri = predicate_disambiguation_chunks
    aggregator = ChunkRDFGraphAggregator()
    aggregated_graph = aggregator.aggregate_graphs(chunks=chunks, doc_namespace=doc_iri)

    # Validate aggregated graph connectivity
    connectivity_result = RDFGraphConnectivityValidator(
        aggregated_graph
    ).validate_connectivity()

    # Check that similar predicates are merged
    assert len(aggregated_graph) > 0
    assert connectivity_result.num_components > 0


def test_edge_cases_aggregation(edge_cases_chunks):
    """Test aggregation with edge cases."""
    chunks, doc_iri = edge_cases_chunks
    aggregator = ChunkRDFGraphAggregator()
    aggregated_graph = aggregator.aggregate_graphs(chunks=chunks, doc_namespace=doc_iri)

    # Validate aggregated graph connectivity
    connectivity_result = RDFGraphConnectivityValidator(
        aggregated_graph
    ).validate_connectivity()

    # Check that edge cases are handled properly
    assert len(aggregated_graph) > 0
    assert connectivity_result.num_components > 0


def test_type_disambiguation_aggregation(type_disambiguation_chunks):
    """Test aggregation with type-based disambiguation."""
    chunks, doc_iri = type_disambiguation_chunks
    aggregator = ChunkRDFGraphAggregator()
    aggregated_graph = aggregator.aggregate_graphs(chunks=chunks, doc_namespace=doc_iri)

    # Validate aggregated graph connectivity
    connectivity_result = RDFGraphConnectivityValidator(
        aggregated_graph
    ).validate_connectivity()

    # Check that entities with same name but different types are not merged
    assert len(aggregated_graph) > 0
    assert connectivity_result.num_components > 0


def test_large_scale_aggregation(large_scale_chunks):
    """Test aggregation with large scale scenarios."""
    chunks, doc_iri = large_scale_chunks
    aggregator = ChunkRDFGraphAggregator()
    aggregated_graph = aggregator.aggregate_graphs(chunks=chunks, doc_namespace=doc_iri)

    # Validate aggregated graph connectivity
    connectivity_result = RDFGraphConnectivityValidator(
        aggregated_graph
    ).validate_connectivity()

    # Check that large scale aggregation works
    assert len(aggregated_graph) > 0
    assert connectivity_result.num_components > 0


def test_complex_predicates_aggregation(complex_predicates_chunks):
    """Test aggregation with complex predicate scenarios."""
    chunks, doc_iri = complex_predicates_chunks
    aggregator = ChunkRDFGraphAggregator()
    aggregated_graph = aggregator.aggregate_graphs(chunks=chunks, doc_namespace=doc_iri)

    # Validate aggregated graph connectivity
    connectivity_result = RDFGraphConnectivityValidator(
        aggregated_graph
    ).validate_connectivity()

    # Check that complex predicates are handled properly
    assert len(aggregated_graph) > 0
    assert connectivity_result.num_components > 0
