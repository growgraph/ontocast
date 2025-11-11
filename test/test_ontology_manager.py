"""Test suite for OntologyManager.

This test suite ensures that:
1. Every ontology in the manager has a created_at field set (not None)
2. Version tracking works correctly
3. Terminal ontology detection works
4. Freshest ontology selection works
5. Lineage graphs are built correctly
"""

from datetime import datetime, timezone

import pytest

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import OntologyManager


@pytest.fixture
def ontology_manager():
    """Create a fresh OntologyManager for each test."""
    return OntologyManager()


@pytest.fixture
def sample_ontology():
    """Create a sample ontology with minimal required fields."""
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        
        <https://example.org/test> a owl:Ontology ;
            rdfs:label "Test Ontology" .
        """,
        format="turtle",
    )
    ontology = Ontology(
        graph=graph,
        ontology_id="test",
        iri="https://example.org/test",
        title="Test Ontology",
        version="1.0.0",
    )
    # Compute hash if not set
    if not ontology.hash:
        ontology._compute_and_set_hash()
    return ontology


@pytest.fixture
def ontology_with_parent(sample_ontology):
    """Create an ontology that has sample_ontology as parent."""
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        
        <https://example.org/test> a owl:Ontology ;
            rdfs:label "Test Ontology v2" .
        """,
        format="turtle",
    )
    ontology = Ontology(
        graph=graph,
        ontology_id="test",
        iri="https://example.org/test",
        title="Test Ontology v2",
        version="2.0.0",
        parent_hashes=[sample_ontology.hash] if sample_ontology.hash else [],
    )
    if not ontology.hash:
        ontology._compute_and_set_hash()
    return ontology


class TestOntologyManagerCreatedAt:
    """Test that created_at is always set when adding ontologies."""

    def test_add_ontology_sets_created_at_if_missing(
        self, ontology_manager, sample_ontology
    ):
        """Test that add_ontology sets created_at if it's None."""
        assert sample_ontology.created_at is None
        ontology_manager.add_ontology(sample_ontology)

        # Check that created_at was set
        assert sample_ontology.created_at is not None
        assert isinstance(sample_ontology.created_at, datetime)

        # Check that it's in the manager
        versions = ontology_manager.get_ontology_versions("test")
        assert len(versions) == 1
        assert versions[0].created_at is not None

    def test_add_ontology_preserves_existing_created_at(
        self, ontology_manager, sample_ontology
    ):
        """Test that add_ontology preserves existing created_at."""
        original_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        sample_ontology.created_at = original_time

        ontology_manager.add_ontology(sample_ontology)

        # Check that created_at was preserved
        assert sample_ontology.created_at == original_time

        versions = ontology_manager.get_ontology_versions("test")
        assert versions[0].created_at == original_time

    def test_all_ontologies_have_created_at(self, ontology_manager, sample_ontology):
        """Test that all ontologies in manager have created_at set."""
        ontology_manager.add_ontology(sample_ontology)

        # Check all ontologies property
        ontologies = ontology_manager.ontologies
        assert len(ontologies) > 0
        for ontology in ontologies:
            assert ontology.created_at is not None, (
                f"Ontology {ontology.ontology_id} (hash: {ontology.hash}) "
                "should have created_at set"
            )

    def test_get_ontology_versions_all_have_created_at(
        self, ontology_manager, sample_ontology, ontology_with_parent
    ):
        """Test that all versions returned have created_at set."""
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        versions = ontology_manager.get_ontology_versions("test")
        assert len(versions) == 2
        for version in versions:
            assert version.created_at is not None, (
                f"Version with hash {version.hash} should have created_at set"
            )


class TestOntologyManagerVersionTracking:
    """Test version tracking functionality."""

    def test_add_ontology_creates_version_tree(self, ontology_manager, sample_ontology):
        """Test that adding an ontology creates a version tree."""
        ontology_manager.add_ontology(sample_ontology)

        assert "test" in ontology_manager.ontology_versions
        assert len(ontology_manager.ontology_versions["test"]) == 1

    def test_add_duplicate_hash_not_added(self, ontology_manager, sample_ontology):
        """Test that adding the same ontology twice doesn't create duplicates."""
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(sample_ontology)

        versions = ontology_manager.get_ontology_versions("test")
        assert len(versions) == 1

    def test_add_ontology_without_hash_rejected(self, ontology_manager):
        """Test that adding ontology without hash is rejected."""
        ontology = Ontology(
            ontology_id="test",
            iri="https://example.org/test",
        )
        assert ontology.hash is None

        ontology_manager.add_ontology(ontology)

        # Should not be added
        assert "test" not in ontology_manager.ontology_versions

    def test_add_ontology_without_ontology_id_rejected(
        self, ontology_manager, sample_ontology
    ):
        """Test that adding ontology without ontology_id is rejected."""
        sample_ontology.ontology_id = None

        ontology_manager.add_ontology(sample_ontology)

        # Should not be added
        assert len(ontology_manager.ontology_versions) == 0


class TestTerminalOntologies:
    """Test terminal ontology detection."""

    def test_single_ontology_is_terminal(self, ontology_manager, sample_ontology):
        """Test that a single ontology is terminal."""
        ontology_manager.add_ontology(sample_ontology)

        terminals = ontology_manager.get_terminal_ontologies("test")
        assert len(terminals) == 1
        assert terminals[0].hash == sample_ontology.hash

    def test_parent_is_not_terminal_when_child_exists(
        self, ontology_manager, sample_ontology, ontology_with_parent
    ):
        """Test that parent is not terminal when child exists."""
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        terminals = ontology_manager.get_terminal_ontologies("test")
        assert len(terminals) == 1
        assert terminals[0].hash == ontology_with_parent.hash
        assert sample_ontology.hash not in [t.hash for t in terminals]

    def test_multiple_terminals_for_different_ontology_ids(
        self, ontology_manager, sample_ontology
    ):
        """Test that we can have terminals for different ontology_ids."""
        # Create second ontology with different ID
        graph2 = RDFGraph()
        graph2.parse(
            data="""
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            
            <https://example.org/test2> a owl:Ontology ;
                rdfs:label "Test Ontology 2" .
            """,
            format="turtle",
        )
        ontology2 = Ontology(
            graph=graph2,
            ontology_id="test2",
            iri="https://example.org/test2",
        )
        if not ontology2.hash:
            ontology2._compute_and_set_hash()

        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology2)

        terminals = ontology_manager.get_terminal_ontologies()
        assert len(terminals) == 2
        assert {t.ontology_id for t in terminals} == {"test", "test2"}


class TestFreshestTerminalOntology:
    """Test freshest terminal ontology selection."""

    def test_freshest_single_ontology(self, ontology_manager, sample_ontology):
        """Test that freshest returns the only ontology when there's one."""
        ontology_manager.add_ontology(sample_ontology)

        freshest = ontology_manager.get_freshest_terminal_ontology("test")
        assert freshest is not None
        assert freshest.hash == sample_ontology.hash

    def test_freshest_selects_most_recent(
        self, ontology_manager, sample_ontology, ontology_with_parent
    ):
        """Test that freshest selects the most recently created ontology."""
        # Set explicit timestamps
        sample_ontology.created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ontology_with_parent.created_at = datetime(
            2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc
        )

        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        freshest = ontology_manager.get_freshest_terminal_ontology("test")
        assert freshest is not None
        assert freshest.hash == ontology_with_parent.hash

    def test_freshest_handles_no_timestamps(self, ontology_manager, sample_ontology):
        """Test that freshest falls back when no timestamps."""
        # This shouldn't happen in practice since add_ontology sets created_at,
        # but test the fallback logic
        sample_ontology.created_at = None
        # Manually add to bypass add_ontology's created_at setting
        if "test" not in ontology_manager.ontology_versions:
            ontology_manager.ontology_versions["test"] = []
        ontology_manager.ontology_versions["test"].append(sample_ontology)

        freshest = ontology_manager.get_freshest_terminal_ontology("test")
        # Should still return something (fallback to first)
        assert freshest is not None

    def test_freshest_returns_none_when_no_ontologies(self, ontology_manager):
        """Test that freshest returns None when no ontologies exist."""
        freshest = ontology_manager.get_freshest_terminal_ontology("nonexistent")
        assert freshest is None


class TestOntologiesProperty:
    """Test the ontologies property."""

    def test_ontologies_returns_freshest_per_ontology_id(
        self, ontology_manager, sample_ontology, ontology_with_parent
    ):
        """Test that ontologies property returns one per ontology_id."""
        sample_ontology.created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ontology_with_parent.created_at = datetime(
            2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc
        )

        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        ontologies = ontology_manager.ontologies
        assert len(ontologies) == 1  # One per ontology_id
        assert ontologies[0].hash == ontology_with_parent.hash

    def test_ontologies_all_have_created_at(self, ontology_manager, sample_ontology):
        """Test that all ontologies returned have created_at."""
        ontology_manager.add_ontology(sample_ontology)

        ontologies = ontology_manager.ontologies
        for ontology in ontologies:
            assert ontology.created_at is not None


class TestLineageGraph:
    """Test lineage graph building."""

    def test_get_lineage_graph_creates_graph(
        self, ontology_manager, sample_ontology, ontology_with_parent
    ):
        """Test that lineage graph is created correctly."""
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        lineage = ontology_manager.get_lineage_graph("test")
        assert lineage is not None
        import networkx as nx

        assert isinstance(lineage, nx.DiGraph)

        # Check nodes exist
        assert sample_ontology.hash in lineage.nodes()
        assert ontology_with_parent.hash in lineage.nodes()

        # Check edge from child to parent
        assert lineage.has_edge(ontology_with_parent.hash, sample_ontology.hash)

    def test_get_lineage_graph_returns_none_for_missing_id(self, ontology_manager):
        """Test that lineage graph returns None for missing ontology_id."""
        lineage = ontology_manager.get_lineage_graph("nonexistent")
        assert lineage is None


class TestGetOntology:
    """Test get_ontology method."""

    def test_get_ontology_by_hash(self, ontology_manager, sample_ontology):
        """Test getting ontology by hash."""
        ontology_manager.add_ontology(sample_ontology)

        retrieved = ontology_manager.get_ontology(hash=sample_ontology.hash)
        assert retrieved.hash == sample_ontology.hash
        assert retrieved.created_at is not None

    def test_get_ontology_by_ontology_id_returns_terminal(
        self, ontology_manager, sample_ontology, ontology_with_parent
    ):
        """Test that getting by ontology_id returns terminal."""
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        retrieved = ontology_manager.get_ontology(ontology_id="test")
        assert retrieved.hash == ontology_with_parent.hash
        assert retrieved.created_at is not None

    def test_get_ontology_by_iri(self, ontology_manager, sample_ontology):
        """Test getting ontology by IRI."""
        ontology_manager.add_ontology(sample_ontology)

        retrieved = ontology_manager.get_ontology(ontology_iri=sample_ontology.iri)
        assert retrieved.iri == sample_ontology.iri
        assert retrieved.created_at is not None


class TestGetOntologyNames:
    """Test get_ontology_names method."""

    def test_get_ontology_names_returns_all_ids(
        self, ontology_manager, sample_ontology
    ):
        """Test that get_ontology_names returns all ontology IDs."""
        ontology_manager.add_ontology(sample_ontology)

        # Create second ontology
        graph2 = RDFGraph()
        graph2.parse(
            data="""
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            
            <https://example.org/test2> a owl:Ontology .
            """,
            format="turtle",
        )
        ontology2 = Ontology(
            graph=graph2,
            ontology_id="test2",
            iri="https://example.org/test2",
        )
        if not ontology2.hash:
            ontology2._compute_and_set_hash()
        ontology_manager.add_ontology(ontology2)

        names = ontology_manager.get_ontology_names()
        assert "test" in names
        assert "test2" in names
        assert len(names) == 2


class TestContains:
    """Test __contains__ method."""

    def test_contains_by_ontology_id(self, ontology_manager, sample_ontology):
        """Test checking containment by ontology_id."""
        ontology_manager.add_ontology(sample_ontology)

        assert "test" in ontology_manager
        assert "nonexistent" not in ontology_manager

    def test_contains_by_iri(self, ontology_manager, sample_ontology):
        """Test checking containment by IRI."""
        ontology_manager.add_ontology(sample_ontology)

        assert sample_ontology.iri in ontology_manager
        assert "https://example.org/nonexistent" not in ontology_manager
