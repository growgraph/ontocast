"""Tests for ontology merging functionality."""

import logging
from datetime import datetime, timezone

import pytest
from rdflib import DCTERMS, OWL, PROV, RDF, RDFS, Literal, URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.ontology_manager import OntologyManager

logger = logging.getLogger(__name__)


@pytest.fixture
def ontology_manager():
    """Create an ontology manager for testing."""
    return OntologyManager()


@pytest.fixture
def base_ontology():
    """Create a base ontology for testing."""
    graph = RDFGraph()
    iri = URIRef("http://example.org/test")
    graph.add((iri, RDF.type, OWL.Ontology))
    graph.add((iri, RDFS.label, Literal("Test Ontology")))

    # Add some classes
    class1 = URIRef("http://example.org/test#Class1")
    graph.add((class1, RDF.type, OWL.Class))
    graph.add((class1, RDFS.label, Literal("Class 1")))

    ontology = Ontology(
        graph=graph,
        iri=str(iri),
        ontology_id="test",
        title="Test Ontology",
        version="1.0.0",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    # Ensure hash is computed
    if not ontology.hash:
        ontology._compute_and_set_hash()
    return ontology


@pytest.fixture
def branch1_ontology(base_ontology):
    """Create a branch 1 ontology (child of base)."""
    # Create a copy of the base graph
    graph = base_ontology.graph.copy()

    # Add new class
    class2 = URIRef("http://example.org/test#Class2")
    graph.add((class2, RDF.type, OWL.Class))
    graph.add((class2, RDFS.label, Literal("Class 2")))

    ontology = Ontology(
        graph=graph,
        iri=base_ontology.iri,
        ontology_id=base_ontology.ontology_id,
        title=base_ontology.title,
        version="1.1.0",
        parent_hashes=[base_ontology.hash] if base_ontology.hash else [],
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    return ontology


@pytest.fixture
def branch2_ontology(base_ontology):
    """Create a branch 2 ontology (child of base)."""
    # Create a copy of the base graph
    graph = base_ontology.graph.copy()

    # Add different new class
    class3 = URIRef("http://example.org/test#Class3")
    graph.add((class3, RDF.type, OWL.Class))
    graph.add((class3, RDFS.label, Literal("Class 3")))

    ontology = Ontology(
        graph=graph,
        iri=base_ontology.iri,
        ontology_id=base_ontology.ontology_id,
        title=base_ontology.title,
        version="1.2.0",
        parent_hashes=[base_ontology.hash] if base_ontology.hash else [],
        created_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
    )
    return ontology


def test_merge_ontologies_basic(ontology_manager, branch1_ontology, branch2_ontology):
    """Test basic ontology merging."""
    from ontocast.onto.ontology_operations import merge_ontologies

    # Ensure hashes are computed
    if not branch1_ontology.hash:
        branch1_ontology._compute_and_set_hash()
    if not branch2_ontology.hash:
        branch2_ontology._compute_and_set_hash()

    # Merge
    merged = merge_ontologies(branch1_ontology, branch2_ontology)

    # Check that merged ontology has both parents
    assert merged.parent_hashes == [branch1_ontology.hash, branch2_ontology.hash]
    assert merged.iri == branch1_ontology.iri
    assert merged.created_at is not None
    assert merged.hash is not None

    # Check that merged graph contains content triples from both (excluding metadata)
    # Metadata (version, title, description, created_at, hash, parent_hash) is not compared
    # as it may differ in the merged ontology
    def get_content_triples(graph, onto_iri):
        """Get content triples (excluding metadata) from a graph."""
        content_triples = set()
        onto_iri_ref = URIRef(onto_iri)
        for s, p, o in graph:
            # Skip metadata triples for the ontology IRI
            if s == onto_iri_ref:
                if (
                    p == DCTERMS.identifier
                    and isinstance(o, Literal)
                    and str(o).startswith("hash:")
                ):
                    continue
                if p == PROV.wasDerivedFrom:
                    continue
                if p == DCTERMS.created:
                    continue
                if p == OWL.versionInfo:
                    continue
                if p == RDFS.label:
                    continue
                if p == DCTERMS.title:
                    continue
                if p == DCTERMS.description:
                    continue
                if p == RDFS.comment:
                    continue
            content_triples.add((s, p, o))
        return content_triples

    branch1_content = get_content_triples(branch1_ontology.graph, branch1_ontology.iri)
    branch2_content = get_content_triples(branch2_ontology.graph, branch2_ontology.iri)
    merged_content = get_content_triples(merged.graph, merged.iri)

    # All content triples from both branches should be in merged
    assert branch1_content.issubset(merged_content), (
        f"Missing triples from branch1: {branch1_content - merged_content}"
    )
    assert branch2_content.issubset(merged_content), (
        f"Missing triples from branch2: {branch2_content - merged_content}"
    )


def test_merge_ontologies_with_contradictions(ontology_manager):
    """Test merging ontologies with contradictions."""
    from ontocast.onto.ontology_operations import merge_ontologies

    # Create two ontologies with conflicting property values
    graph1 = RDFGraph()
    iri = URIRef("http://example.org/test")
    graph1.add((iri, RDF.type, OWL.Ontology))
    class1 = URIRef("http://example.org/test#Class1")
    graph1.add((class1, RDF.type, OWL.Class))
    graph1.add((class1, RDFS.label, Literal("Class One")))  # Different label

    graph2 = RDFGraph()
    graph2.add((iri, RDF.type, OWL.Ontology))
    graph2.add((class1, RDF.type, OWL.Class))
    graph2.add((class1, RDFS.label, Literal("Class 1")))  # Different label

    onto1 = Ontology(
        graph=graph1,
        iri=str(iri),
        ontology_id="test",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    onto2 = Ontology(
        graph=graph2,
        iri=str(iri),
        ontology_id="test",
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    # Merge should succeed (both values kept in RDF)
    merged = merge_ontologies(onto1, onto2)

    # Both label values should be in merged graph
    labels = [o for _, _, o in merged.graph.triples((class1, RDFS.label, None))]
    assert len(labels) == 2
    label_strings = {str(label) for label in labels}
    assert "Class One" in label_strings or '"Class One"' in label_strings
    assert "Class 1" in label_strings or '"Class 1"' in label_strings


def test_merge_terminal_ontologies_pairwise(
    ontology_manager, base_ontology, branch1_ontology, branch2_ontology
):
    """Test merging terminal ontologies pair-wise."""
    from ontocast.onto.ontology_operations import merge_ontologies

    # Add all ontologies to manager
    ontology_manager.add_ontology(base_ontology)
    ontology_manager.add_ontology(branch1_ontology)
    ontology_manager.add_ontology(branch2_ontology)

    # Get terminal ontologies (should be branch1 and branch2)
    terminals = ontology_manager.get_terminal_ontologies_by_iri(base_ontology.iri)
    assert len(terminals) == 2

    # Sort by created_at
    terminals.sort(
        key=lambda x: x.created_at or datetime.min.replace(tzinfo=timezone.utc)
    )

    # Merge the two terminals
    merged = merge_ontologies(terminals[0], terminals[1])

    # Add merged to manager
    ontology_manager.add_ontology(merged)

    # Check that we now have one terminal
    new_terminals = ontology_manager.get_terminal_ontologies_by_iri(base_ontology.iri)
    assert len(new_terminals) == 1
    assert new_terminals[0].hash == merged.hash


def test_merge_ontologies_preserves_namespaces(ontology_manager):
    """Test that merging preserves namespace bindings."""
    from ontocast.onto.ontology_operations import merge_ontologies

    graph1 = RDFGraph()
    graph1.bind("ex", "http://example.org/")
    iri = URIRef("http://example.org/test")
    graph1.add((iri, RDF.type, OWL.Ontology))

    graph2 = RDFGraph()
    graph2.bind("test", "http://test.org/")
    graph2.add((iri, RDF.type, OWL.Ontology))

    onto1 = Ontology(graph=graph1, iri=str(iri), created_at=datetime.now(timezone.utc))
    onto2 = Ontology(graph=graph2, iri=str(iri), created_at=datetime.now(timezone.utc))

    merged = merge_ontologies(onto1, onto2)

    # Check that both namespaces are present
    namespaces = dict(merged.graph.namespaces())
    assert "ex" in namespaces
    assert "test" in namespaces


def test_merge_ontologies_created_at_set(ontology_manager):
    """Test that merged ontology has created_at set to merge time."""
    from ontocast.onto.ontology_operations import merge_ontologies

    graph1 = RDFGraph()
    iri = URIRef("http://example.org/test")
    graph1.add((iri, RDF.type, OWL.Ontology))

    graph2 = RDFGraph()
    graph2.add((iri, RDF.type, OWL.Ontology))

    onto1 = Ontology(
        graph=graph1,
        iri=str(iri),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    onto2 = Ontology(
        graph=graph2,
        iri=str(iri),
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    before_merge = datetime.now(timezone.utc)
    merged = merge_ontologies(onto1, onto2)
    after_merge = datetime.now(timezone.utc)

    # Created_at should be set to merge time (between before and after)
    assert merged.created_at is not None
    assert before_merge <= merged.created_at <= after_merge
