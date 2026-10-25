"""The catalog held in memory: stamped versions keyed by version IRI.

Covers registration (stamped versions only), heads per IRI, the one head the
``ontologies`` property serves, the lineage DAG, lookups by id / IRI /
version IRI, and the header round trip through a version graph.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import networkx as nx
import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_header import (
    ontology_from_version_graph,
    stamp_version,
)
from ontocast.onto.ontology_operations import merge_ontologies
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import OntologyManager
from test.ontology_helpers import next_version, stamped_ontology

pytestmark = pytest.mark.unit

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2024, 1, 3, 12, 0, 0, tzinfo=timezone.utc)

IRI = "https://example.org/test"
IRI2 = "https://example.org/test2"

ROOT_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<https://example.org/test> a owl:Ontology ;
    rdfs:label "Test Ontology" .

<https://example.org/test#Thing> a owl:Class ;
    rdfs:label "Thing" .
"""

SECOND_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<https://example.org/test2> a owl:Ontology ;
    rdfs:label "Test Ontology 2" .

<https://example.org/test2#Other> a owl:Class .
"""


def _with_class(base: Ontology, local: str) -> RDFGraph:
    graph = base.graph.copy()
    term = URIRef(f"{base.iri}#{local}")
    graph.add((term, RDF.type, OWL.Class))
    graph.add((term, RDFS.label, Literal(local)))
    return graph


@pytest.fixture
def ontology_manager() -> OntologyManager:
    return OntologyManager()


@pytest.fixture
def sample_ontology() -> Ontology:
    """A stamped root version of ``https://example.org/test``."""
    return stamped_ontology(ROOT_TTL, now=T0)


@pytest.fixture
def ontology_with_parent(sample_ontology: Ontology) -> Ontology:
    """A stamped child of ``sample_ontology`` with one more class."""
    return next_version(
        sample_ontology, _with_class(sample_ontology, "NewClass"), now=T1
    )


@pytest.fixture
def second_ontology() -> Ontology:
    """A stamped root of a different ontology, ``https://example.org/test2``."""
    return stamped_ontology(SECOND_TTL, now=T0)


class TestVersionTimestamps:
    """``created_at`` is the start of a lineage; ``modified_at`` the version's."""

    def test_root_version_is_created_and_modified_at_its_stamp_time(
        self, sample_ontology: Ontology
    ) -> None:
        assert sample_ontology.created_at == T0
        assert sample_ontology.modified_at == T0

    def test_child_version_keeps_the_root_created_at(
        self, sample_ontology: Ontology, ontology_with_parent: Ontology
    ) -> None:
        assert ontology_with_parent.created_at == sample_ontology.created_at
        assert ontology_with_parent.modified_at == T1

    def test_every_registered_version_has_timestamps(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        ontology_with_parent: Ontology,
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        versions = ontology_manager.get_ontology_versions("test")
        assert len(versions) == 2
        for version in versions:
            assert version.created_at is not None
            assert version.modified_at is not None


class TestVersionRegistration:
    """Only stamped versions enter the catalog, once each."""

    def test_add_ontology_creates_version_tree(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)

        assert sample_ontology.iri in ontology_manager.ontology_versions
        assert len(ontology_manager.ontology_versions[sample_ontology.iri]) == 1

    def test_same_version_iri_is_not_added_twice(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(sample_ontology)

        assert len(ontology_manager.get_ontology_versions("test")) == 1

    def test_unstamped_artifact_is_refused(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        derived = sample_ontology.derive_updated_version(
            _with_class(sample_ontology, "Unstamped")
        )
        assert derived.version_iri is None

        with pytest.raises(ValueError, match="not stamped"):
            ontology_manager.add_ontology(derived)
        assert ontology_manager.ontology_versions == {}

    def test_null_ontology_is_refused(self, ontology_manager: OntologyManager) -> None:
        with pytest.raises(ValueError, match="null ontology"):
            ontology_manager.add_ontology(NULL_ONTOLOGY)
        assert ontology_manager.ontology_versions == {}

    def test_registered_version_becomes_its_own_base(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)

        held = ontology_manager.get_ontology_versions("test")[0]
        assert [base.version_iri for base in held.bases] == [held.version_iri]


class TestHeads:
    """A head is a version no held version names as prior."""

    def test_single_version_is_the_head(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)

        heads = ontology_manager.get_heads("test")
        assert [head.version_iri for head in heads] == [sample_ontology.version_iri]

    def test_prior_is_not_a_head_once_its_child_is_held(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        ontology_with_parent: Ontology,
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        heads = ontology_manager.get_heads("test")
        assert [head.version_iri for head in heads] == [
            ontology_with_parent.version_iri
        ]

    def test_one_head_per_ontology(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        second_ontology: Ontology,
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(second_ontology)

        heads = ontology_manager.get_heads()
        assert {head.ontology_id for head in heads} == {"test", "test2"}


class TestHead:
    """``get_head`` is the one head of an ontology, whatever its timestamps."""

    def test_head_of_a_single_version(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)

        head = ontology_manager.get_head("test")
        assert head is not None
        assert head.version_iri == sample_ontology.version_iri

    def test_head_is_the_child_whatever_the_registration_order(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        ontology_with_parent: Ontology,
    ) -> None:
        ontology_manager.add_ontology(ontology_with_parent)
        ontology_manager.add_ontology(sample_ontology)

        head = ontology_manager.get_head("test")
        assert head is not None
        assert head.version_iri == ontology_with_parent.version_iri

    def test_head_is_none_when_nothing_is_held(
        self, ontology_manager: OntologyManager
    ) -> None:
        assert ontology_manager.get_head("nonexistent") is None


class TestOntologiesProperty:
    """``ontologies`` serves one head per IRI and tracks it by version IRI."""

    def test_ontologies_returns_the_head_per_iri(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        ontology_with_parent: Ontology,
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        ontologies = ontology_manager.ontologies
        assert [o.version_iri for o in ontologies] == [ontology_with_parent.version_iri]

    def test_ontologies_cache_is_updated_incrementally(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        second_ontology: Ontology,
    ) -> None:
        assert not ontology_manager.has_ontologies
        assert len(ontology_manager.ontologies) == 0

        ontology_manager.add_ontology(sample_ontology)
        assert ontology_manager.has_ontologies
        assert len(ontology_manager.ontologies) == 1
        assert (
            ontology_manager._cached_ontologies[sample_ontology.iri]
            == sample_ontology.version_iri
        )

        ontology_manager.add_ontology(second_ontology)
        assert len(ontology_manager.ontologies) == 2
        assert (
            ontology_manager._cached_ontologies[second_ontology.iri]
            == second_ontology.version_iri
        )

    def test_ontologies_cache_moves_to_the_new_head(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        ontology_with_parent: Ontology,
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        assert len(ontology_manager.ontologies) == 1
        assert (
            ontology_manager._cached_ontologies[sample_ontology.iri]
            == sample_ontology.version_iri
        )

        ontology_manager.add_ontology(ontology_with_parent)
        assert sample_ontology.iri not in ontology_manager._cached_ontologies
        assert len(ontology_manager.ontologies) == 1
        assert (
            ontology_manager._cached_ontologies[sample_ontology.iri]
            == ontology_with_parent.version_iri
        )


class TestHasOntologies:
    def test_has_ontologies_works_with_cache(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        assert not ontology_manager.has_ontologies

        ontology_manager.add_ontology(sample_ontology)
        assert ontology_manager.has_ontologies

        _ = ontology_manager.ontologies
        assert ontology_manager.has_ontologies


class TestLineageGraph:
    """The version DAG: nodes are version IRIs, edges go child -> prior."""

    def test_get_lineage_graph_links_child_to_prior(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        ontology_with_parent: Ontology,
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        lineage = ontology_manager.get_lineage_graph("test")
        assert isinstance(lineage, nx.DiGraph)
        assert set(lineage.nodes()) == {
            sample_ontology.version_iri,
            ontology_with_parent.version_iri,
        }
        assert lineage.has_edge(
            ontology_with_parent.version_iri, sample_ontology.version_iri
        )

    def test_get_lineage_graph_returns_none_for_missing_id(
        self, ontology_manager: OntologyManager
    ) -> None:
        assert ontology_manager.get_lineage_graph("nonexistent") is None


class TestGetOntology:
    def test_get_ontology_by_version_iri(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        ontology_with_parent: Ontology,
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        retrieved = ontology_manager.get_ontology(
            version_iri=sample_ontology.version_iri
        )
        assert retrieved.version_iri == sample_ontology.version_iri

    def test_get_ontology_by_ontology_id_returns_the_head(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        ontology_with_parent: Ontology,
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(ontology_with_parent)

        retrieved = ontology_manager.get_ontology(ontology_id="test")
        assert retrieved.version_iri == ontology_with_parent.version_iri

    def test_get_ontology_by_iri(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)

        retrieved = ontology_manager.get_ontology(ontology_iri=sample_ontology.iri)
        assert retrieved.iri == sample_ontology.iri

    def test_get_ontology_unknown_is_the_null_ontology(
        self, ontology_manager: OntologyManager
    ) -> None:
        assert ontology_manager.get_ontology(ontology_id="nonexistent").is_null()


class TestGetOntologyNames:
    def test_get_ontology_names_returns_all_ids(
        self,
        ontology_manager: OntologyManager,
        sample_ontology: Ontology,
        second_ontology: Ontology,
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.add_ontology(second_ontology)

        assert ontology_manager.get_ontology_names() == ["test", "test2"]


class TestContains:
    def test_contains_by_ontology_id(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)

        assert "test" in ontology_manager
        assert "nonexistent" not in ontology_manager

    def test_contains_by_iri(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)

        assert sample_ontology.iri in ontology_manager
        assert "https://example.org/nonexistent" not in ontology_manager


class TestVersionGraphRoundTrip:
    """A version graph carries its whole header: lineage survives the store."""

    @staticmethod
    def _versions() -> dict[str, Ontology]:
        root = stamped_ontology(ROOT_TTL, now=T0)
        child = next_version(root, _with_class(root, "Child"), now=T1)
        sibling = next_version(root, _with_class(root, "Sibling"), now=T2)
        merged = stamp_version(merge_ontologies(child, sibling), now=T2)
        assert merged is not None
        return {"root": root, "child": child, "merge": merged}

    @pytest.mark.parametrize("kind", ["root", "child", "merge"])
    def test_lineage_and_timestamps_survive_a_version_graph_round_trip(
        self, kind: str
    ) -> None:
        version = self._versions()[kind]
        assert version.version_iri is not None

        recreated = ontology_from_version_graph(
            version.version_iri, version.graph.copy()
        )

        assert recreated.version_iri == version.version_iri
        assert recreated.prior_version_iris == sorted(version.prior_version_iris)
        assert recreated.created_at == version.created_at
        assert recreated.modified_at == version.modified_at
        assert recreated.version == version.version
        assert recreated.ontology_id == version.ontology_id
        assert recreated.title == version.title
        assert recreated.hash == version.hash
        assert [base.version_iri for base in recreated.bases] == [version.version_iri]

    def test_merge_round_trip_names_both_priors(self) -> None:
        merged = self._versions()["merge"]
        assert merged.version_iri is not None
        assert len(merged.prior_version_iris) == 2

        recreated = ontology_from_version_graph(merged.version_iri, merged.graph.copy())

        assert set(recreated.prior_version_iris) == set(merged.prior_version_iris)


class TestOntologyManagerVectorAndRemoval:
    """Vector indexing flags and IRI removal."""

    def test_add_ontology_skip_vector_index_skips_reindex(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        patch_retriever = MagicMock()
        patch_retriever.vector_store = MagicMock()
        ontology_manager.register_vector_store(patch_retriever)
        ontology_manager.add_ontology(sample_ontology, skip_vector_index=True)
        patch_retriever.vector_store.reindex_ontology.assert_not_called()

    def test_add_ontology_reindexes_by_default(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        patch_retriever = MagicMock()
        patch_retriever.vector_store = MagicMock()
        ontology_manager.register_vector_store(patch_retriever)
        ontology_manager.add_ontology(sample_ontology)
        patch_retriever.vector_store.reindex_ontology.assert_called_once()

    @pytest.mark.anyio
    async def test_aadd_ontology_reindexes_off_loop(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        patch_retriever = MagicMock()
        patch_retriever.vector_store = MagicMock()
        ontology_manager.register_vector_store(patch_retriever)
        await ontology_manager.aadd_ontology(sample_ontology)
        patch_retriever.vector_store.reindex_ontology.assert_called_once_with(
            sample_ontology
        )

    @pytest.mark.anyio
    async def test_add_ontology_reindex_refuses_running_loop(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        patch_retriever = MagicMock()
        patch_retriever.vector_store = MagicMock()
        ontology_manager.register_vector_store(patch_retriever)
        with pytest.raises(RuntimeError, match="aadd_ontology"):
            ontology_manager.add_ontology(sample_ontology)
        patch_retriever.vector_store.reindex_ontology.assert_not_called()
        assert sample_ontology.iri not in ontology_manager.ontology_versions

    @pytest.mark.anyio
    async def test_get_patch_contexts_refuses_running_loop(
        self, ontology_manager: OntologyManager
    ) -> None:
        with pytest.raises(RuntimeError, match="aget_patch_contexts_with_sources"):
            ontology_manager.get_patch_contexts_with_sources(queries=["q"])

    @pytest.mark.anyio
    async def test_aget_patch_contexts_fallback_copies_graph(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        ontology_manager.add_ontology(sample_ontology, skip_vector_index=True)
        results = await ontology_manager.aget_patch_contexts_with_sources(
            queries=["a", "b"]
        )
        assert len(results) == 2
        g0, sources0 = results[0]
        g1, sources1 = results[1]
        assert g0 is not None and g1 is not None
        assert g0 is not g1
        assert sources0 == sources1 == [sample_ontology.iri]

    def test_remove_ontology_by_iri_clears_versions(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.remove_ontology_by_iri(sample_ontology.iri)
        assert sample_ontology.iri not in ontology_manager.ontology_versions
        assert sample_ontology.iri not in ontology_manager._cached_ontologies

    def test_remove_ontology_by_iri_evicts_the_graph_cache(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        """Eviction uses the key entries were inserted under (the graph URI)."""
        graph_uri = "urn:tenant:acme:onto:test-v1"
        ontology_manager._cache_graph(sample_ontology, graph_uri)
        assert graph_uri in ontology_manager._graph_cache

        ontology_manager.remove_ontology_by_iri(sample_ontology.iri)

        assert graph_uri not in ontology_manager._graph_cache
        assert sample_ontology.iri not in ontology_manager._graph_uris_by_iri

    def test_graph_cache_is_bounded(
        self, ontology_manager: OntologyManager, sample_ontology: Ontology
    ) -> None:
        """It holds whole rdflib graphs, so a long-lived server needs a bound."""
        from ontocast.tool.ontology_manager import _GRAPH_CACHE_MAX_ENTRIES

        for index in range(_GRAPH_CACHE_MAX_ENTRIES + 10):
            ontology_manager._cache_graph(sample_ontology, f"urn:graph:{index}")

        assert len(ontology_manager._graph_cache) == _GRAPH_CACHE_MAX_ENTRIES
        assert "urn:graph:0" not in ontology_manager._graph_cache
        assert (
            f"urn:graph:{_GRAPH_CACHE_MAX_ENTRIES + 9}" in ontology_manager._graph_cache
        )
