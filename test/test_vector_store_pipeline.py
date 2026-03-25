"""Unit tests for atomization, batching, and retriever expansion pipeline."""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from pydantic import PrivateAttr

from ontocast.config import EmbeddingConfig, QdrantConfig
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.sparql import SPARQLTool
from ontocast.tool.vector_store.atomizer import GraphAtomizer
from ontocast.tool.vector_store.core import (
    GraphAtom,
    OntologySearchHit,
    canonicalize_entity_role,
)
from ontocast.tool.vector_store.embedding import EmbeddingTool
from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever
from ontocast.tool.vector_store.qdrant import QdrantVectorStore
from ontocast.util import render_text_hash


class CountingEmbeddingTool(EmbeddingTool):
    """Embedding test double with deterministic vectors and call tracking."""

    calls: int = 0
    truncate_by_one: bool = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            digest = render_text_hash(text, digits=None)
            seed = int(digest[:16], 16)
            vector = [
                (((seed + i * 13) % 2000) / 1000.0) - 1.0
                for i in range(self.config.dimension)
            ]
            vectors.append(vector)
        if self.truncate_by_one and vectors:
            return vectors[:-1]
        return vectors


class StubVectorStore(QdrantVectorStore):
    """Vector store stub for retriever unit tests."""

    _atoms: list[GraphAtom] = PrivateAttr(default_factory=list)

    def set_atoms(self, atoms: Iterable[GraphAtom]) -> None:
        self._atoms = list(atoms)

    def search_patches(
        self,
        query: str,
        top_k: int = 10,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[GraphAtom]:
        del query, filter_iri, filter_version, filter_hash
        return self._atoms[:top_k]

    def search_patch_hits(
        self,
        query: str,
        top_k: int = 10,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHit]:
        del query, filter_iri, filter_version, filter_hash
        return [OntologySearchHit(atom=atom, score=1.0) for atom in self._atoms[:top_k]]

    def search_patch_hits_many(
        self,
        queries: list[str],
        top_k: int = 10,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[list[OntologySearchHit]]:
        del filter_iri, filter_version, filter_hash
        return [self.search_patch_hits(query=query, top_k=top_k) for query in queries]

    async def asearch_patch_hits_many(
        self,
        queries: list[str],
        top_k: int = 10,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[list[OntologySearchHit]]:
        return self.search_patch_hits_many(
            queries=queries,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )


class StubSPARQLTool(SPARQLTool):
    """SPARQL tool stub that records induced-subgraph requests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_entity_uris: list[str] = []
        self._last_ontology_iris: list[str] = []
        self._last_ontology_version_filters: dict[str, set[str]] | None = None
        self._last_ontology_hash_filters: dict[str, set[str]] | None = None

    @property
    def last_entity_uris(self) -> list[str]:
        return self._last_entity_uris

    @property
    def last_ontology_iris(self) -> list[str]:
        return self._last_ontology_iris

    @property
    def last_ontology_version_filters(self) -> dict[str, set[str]] | None:
        return self._last_ontology_version_filters

    @property
    def last_ontology_hash_filters(self) -> dict[str, set[str]] | None:
        return self._last_ontology_hash_filters

    def get_induced_subgraph(
        self,
        entity_uris: list[str],
        ontology_iris: list[str] | None = None,
        depth: int = 1,
        max_triples: int = 2000,
        ontology_version_filters: dict[str, set[str]] | None = None,
        ontology_hash_filters: dict[str, set[str]] | None = None,
    ) -> RDFGraph:
        del depth, max_triples
        self._last_entity_uris = entity_uris
        self._last_ontology_iris = ontology_iris or []
        self._last_ontology_version_filters = ontology_version_filters
        self._last_ontology_hash_filters = ontology_hash_filters
        graph = RDFGraph._from_turtle_str(
            """
            @prefix ex: <https://example.org/smoke#> .
            ex:Alpha ex:relatedTo ex:Beta .
            """
        )
        return graph


def _build_smoke_ontology() -> Ontology:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix ex: <https://example.org/smoke#> .

        ex: a owl:Ontology ;
            rdfs:label "Smoke Ontology" ;
            rdfs:comment "Ontology used for unit tests." .

        ex:Concept a rdfs:Class ;
            rdfs:label "Concept" .

        ex:relatedTo a rdf:Property ;
            rdfs:label "related to" ;
            rdfs:domain ex:Concept ;
            rdfs:range ex:Concept .

        ex:Alpha a ex:Concept ;
            rdfs:label "Alpha concept" ;
            ex:relatedTo ex:Beta .

        ex:Beta a ex:Concept ;
            rdfs:label "Beta concept" .
        """
    )
    return Ontology(graph=graph)


def test_atomizer_generates_representation_atoms_for_predicates() -> None:
    ontology = _build_smoke_ontology()
    atomizer = GraphAtomizer()
    atoms = atomizer.atomize(source=ontology, depth=1)

    assert atoms
    assert any(atom.entity_role == "predicate" for atom in atoms)
    assert all(atom.core_representation.strip() for atom in atoms)
    assert all(atom.neighborhood_representation.strip() for atom in atoms)
    assert all(atom.ontology_version == ontology.version for atom in atoms)
    assert "turtle" not in GraphAtom.model_fields


def test_embed_texts_batched_respects_batch_size() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    store = QdrantVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    vectors = store._embed_texts_batched(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert embedding.calls == 3


def test_embed_texts_batched_raises_on_mismatch() -> None:
    embedding = CountingEmbeddingTool(
        config=EmbeddingConfig(dimension=8), truncate_by_one=True
    )
    store = QdrantVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )

    try:
        store._embed_texts_batched(["alpha", "beta"])
    except ValueError as error:
        assert "mismatched vectors" in str(error)
    else:
        raise AssertionError("Expected ValueError for embedding/vector count mismatch")


def test_retriever_expands_graph_via_sparql_tool() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    atoms = [
        GraphAtom(
            atom_id="a1",
            ontology_iri="https://example.org/smoke",
            ontology_id="smoke",
            ontology_hash="hash1",
            ontology_version="1.0.0",
            iri="https://example.org/smoke#Alpha",
            entity_role="resource",
            core_representation="alpha concept",
            neighborhood_representation="alpha related to beta",
        ),
        GraphAtom(
            atom_id="a2",
            ontology_iri="https://example.org/smoke",
            ontology_id="smoke",
            ontology_hash="hash1",
            ontology_version="1.0.0",
            iri="https://example.org/smoke#relatedTo",
            entity_role="predicate",
            core_representation="related to predicate",
            neighborhood_representation="predicate related to links alpha and beta",
        ),
    ]
    vector_store.set_atoms(atoms)

    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store, sparql_tool=sparql_tool
    )
    graph, returned_atoms = retriever.retrieve(
        query="alpha", top_k=2, expand_sparql=True
    )

    assert len(returned_atoms) == 2
    assert len(graph) > 0
    assert sparql_tool.last_entity_uris == sorted({atom.iri for atom in atoms})
    assert sparql_tool.last_ontology_iris == ["https://example.org/smoke"]
    assert sparql_tool.last_ontology_version_filters == {
        "https://example.org/smoke": {"1.0.0"}
    }
    assert sparql_tool.last_ontology_hash_filters == {
        "https://example.org/smoke": {"hash1"}
    }


@pytest.mark.anyio
async def test_retriever_aretrieve_expands_graph_via_sparql_tool() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    atoms = [
        GraphAtom(
            atom_id="a1",
            ontology_iri="https://example.org/smoke",
            ontology_id="smoke",
            ontology_hash="hash1",
            ontology_version="1.0.0",
            iri="https://example.org/smoke#Alpha",
            entity_role="resource",
            core_representation="alpha concept",
            neighborhood_representation="alpha related to beta",
        ),
        GraphAtom(
            atom_id="a2",
            ontology_iri="https://example.org/smoke",
            ontology_id="smoke",
            ontology_hash="hash1",
            ontology_version="1.0.0",
            iri="https://example.org/smoke#relatedTo",
            entity_role="predicate",
            core_representation="related to predicate",
            neighborhood_representation="predicate related to links alpha and beta",
        ),
    ]
    vector_store.set_atoms(atoms)

    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store, sparql_tool=sparql_tool
    )
    graph, returned_atoms = await retriever.aretrieve(
        query="alpha", top_k=2, expand_sparql=True
    )

    assert len(returned_atoms) == 2
    assert len(graph) > 0
    assert sparql_tool.last_entity_uris == sorted({atom.iri for atom in atoms})
    assert sparql_tool.last_ontology_iris == ["https://example.org/smoke"]
    assert sparql_tool.last_ontology_version_filters == {
        "https://example.org/smoke": {"1.0.0"}
    }
    assert sparql_tool.last_ontology_hash_filters == {
        "https://example.org/smoke": {"hash1"}
    }


def test_canonicalize_entity_role_maps_synonyms() -> None:
    assert canonicalize_entity_role("predicate") == "predicate"
    assert canonicalize_entity_role("property") == "predicate"
    assert canonicalize_entity_role("class") == "resource"
    assert canonicalize_entity_role("instance") == "resource"
    assert canonicalize_entity_role("resource") == "resource"
    assert canonicalize_entity_role("unknown") is None


def test_ontology_atom_contract_iri_and_combined_representation() -> None:
    atom = GraphAtom(
        atom_id="a",
        ontology_iri="https://example.org/o",
        iri="https://example.org/o#A",
        entity_role="property",
        core_representation="core text",
        neighborhood_representation="neighbor text",
    )
    assert atom.entity_role == "predicate"
    assert atom.iri == "https://example.org/o#A"
    assert atom.ontology_iri == "https://example.org/o"
    assert atom.representation == "core text. neighbor text"


def test_search_hits_by_vector_returns_typed_scores() -> None:
    class _Point:
        def __init__(self, point_id: str, score: float, neighborhood: str) -> None:
            self.id = point_id
            self.score = score
            self.payload = {"neighborhood_representation": neighborhood}

    class _Store(QdrantVectorStore):
        def _query_named_vector(
            self,
            vector_name: str,
            vector: list[float],
            limit: int,
            search_filter,
        ):
            del vector, limit, search_filter
            if vector_name == "core":
                return [_Point("p1", 0.8, "neighbor"), _Point("p2", 0.4, "neighbor")]
            return [_Point("p1", 0.5, "neighbor"), _Point("p2", 0.2, "neighbor")]

        def _point_to_atom(self, point):
            return GraphAtom(
                atom_id=str(point.id),
                ontology_iri="https://example.org/o",
                iri=f"https://example.org/o#{point.id}",
                entity_role="resource",
                core_representation="core",
                neighborhood_representation="neighbor",
            )

    store = _Store(
        config=QdrantConfig(
            embedding_batch_size=2,
            upsert_batch_size=2,
            fusion_core_weight=0.7,
            fusion_neighborhood_weight=0.3,
        ),
        embedding=CountingEmbeddingTool(config=EmbeddingConfig(dimension=8)),
    )
    hits = store.search_hits_by_vector(
        core_vector=[0.0] * 8,
        neighborhood_vector=[0.0] * 8,
        top_k=2,
    )

    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert hits[0].atom.score == hits[0].score
