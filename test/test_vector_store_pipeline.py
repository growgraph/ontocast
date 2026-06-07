"""Unit tests for atomization, batching, and retriever expansion pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest
from pydantic import PrivateAttr

from ontocast.config import (
    CrossQueryMergeMode,
    EmbeddingConfig,
    PatchRetrievalConfig,
    QdrantConfig,
    QdrantDedupMode,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.sparql import SPARQLTool
from ontocast.tool.vector_store.atomizer import (
    STANDARD_VOCABULARY_NAMESPACE_PREFIXES,
    GraphAtomizer,
)
from ontocast.tool.vector_store.core import (
    BM25_VECTOR_NAME,
    GraphAtom,
    OntologySearchHit,
    OntologySearchHitsByChannel,
    canonicalize_entity_role,
)
from ontocast.tool.vector_store.embedding import EmbeddingTool, FastembedBm25SparseTool
from ontocast.tool.vector_store.patch_retriever import (
    OntologyPatchRetriever,
    _expand_ontology_iris_by_reference,
    _merge_hits_across_queries_hybrid,
    _merge_hits_across_queries_max_score,
    _mmr_rerank,
    _normalize_core_neighborhood_weights,
)
from ontocast.tool.vector_store.qdrant import QdrantVectorStore
from ontocast.util.hash import render_text_hash


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
    _override_hits_by_query: list[OntologySearchHitsByChannel] | None = PrivateAttr(
        default=None
    )
    _vectors: dict[str, tuple[list[float], list[float]]] = PrivateAttr(
        default_factory=dict
    )
    _afetch_vectors_calls: int = PrivateAttr(default=0)

    def set_atoms(self, atoms: Iterable[GraphAtom]) -> None:
        self._atoms = list(atoms)

    def set_hits_by_query(self, rows: list[OntologySearchHitsByChannel]) -> None:
        """Fixed scored hits per query (bypasses ``set_atoms`` ordering for ensemble tests)."""
        self._override_hits_by_query = rows

    def set_vectors(self, vectors: dict[str, tuple[list[float], list[float]]]) -> None:
        self._vectors = vectors

    @property
    def afetch_vectors_calls(self) -> int:
        return self._afetch_vectors_calls

    def search_patches(
        self,
        query: str,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[GraphAtom]:
        del query, filter_iri, filter_version, filter_hash
        k = self.config.top_k if top_k is None else top_k
        return self._atoms[:k]

    def search_patch_hits(
        self,
        query: str,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHit]:
        del query, filter_iri, filter_version, filter_hash
        k = self.config.top_k if top_k is None else top_k
        return [OntologySearchHit(atom=atom, score=1.0) for atom in self._atoms[:k]]

    def search_patch_hits_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHitsByChannel]:
        del filter_iri, filter_version, filter_hash
        return [
            OntologySearchHitsByChannel(
                core_hits=self.search_patch_hits(query=query, top_k=top_k),
                neighborhood_hits=[],
                bm25_hits=[],
            )
            for query in queries
        ]

    async def asearch_patch_hits_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHitsByChannel]:
        if self._override_hits_by_query is not None:
            return self._override_hits_by_query
        return self.search_patch_hits_many(
            queries=queries,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )

    async def afetch_vectors(
        self, atom_ids: list[str]
    ) -> dict[str, tuple[list[float], list[float]]]:
        self._afetch_vectors_calls += 1
        return {
            atom_id: self._vectors[atom_id]
            for atom_id in atom_ids
            if atom_id in self._vectors
        }


class StubSPARQLTool(SPARQLTool):
    """SPARQL tool stub that records induced-subgraph requests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_entity_uris: list[str] = []
        self._last_entity_relevance: dict[str, float] | None = None
        self._last_entity_roles: Mapping[str, str | None] | None = None
        self._last_ontology_iris: list[str] = []
        self._last_ontology_version_filters: dict[str, set[str]] | None = None
        self._last_ontology_hash_filters: dict[str, set[str]] | None = None
        self._last_max_total_triples: int | None = None
        self._last_estimated_triples_per_query: int | None = None
        self.induced_subgraph_calls: int = 0

    @property
    def last_entity_uris(self) -> list[str]:
        return self._last_entity_uris

    @property
    def last_ontology_iris(self) -> list[str]:
        return self._last_ontology_iris

    @property
    def last_entity_relevance(self) -> dict[str, float] | None:
        return self._last_entity_relevance

    @property
    def last_max_total_triples(self) -> int | None:
        return self._last_max_total_triples

    @property
    def last_estimated_triples_per_query(self) -> int | None:
        return self._last_estimated_triples_per_query

    @property
    def last_ontology_version_filters(self) -> dict[str, set[str]] | None:
        return self._last_ontology_version_filters

    @property
    def last_ontology_hash_filters(self) -> dict[str, set[str]] | None:
        return self._last_ontology_hash_filters

    def get_induced_subgraph(
        self,
        entity_uris: list[str],
        entity_relevance: dict[str, float] | None = None,
        entity_roles: Mapping[str, str | None] | None = None,
        ontology_iris: list[str] | None = None,
        depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
        ontology_version_filters: dict[str, set[str]] | None = None,
        ontology_hash_filters: dict[str, set[str]] | None = None,
        hub_seed_count: int = 8,
        ancestor_closure_depth: int = 3,
    ) -> RDFGraph:
        del depth, hub_seed_count, ancestor_closure_depth
        self.induced_subgraph_calls += 1
        self._last_entity_uris = entity_uris
        self._last_entity_relevance = entity_relevance
        self._last_entity_roles = entity_roles
        self._last_ontology_iris = ontology_iris or []
        self._last_ontology_version_filters = ontology_version_filters
        self._last_ontology_hash_filters = ontology_hash_filters
        self._last_max_total_triples = max_total_triples
        self._last_estimated_triples_per_query = estimated_triples_per_query
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
    assert any(atom.neighborhood_representation.strip() for atom in atoms)
    assert all(
        atom.neighborhood_representation.strip()
        for atom in atoms
        if atom.entity_role == "predicate"
    )
    assert all(atom.ontology_version == ontology.version for atom in atoms)
    assert "turtle" not in GraphAtom.model_fields


def test_atomizer_core_and_neighborhood_bias_structural_signal() -> None:
    """Core uses domain/range and drops generic OWL types; neighborhood uses clue phrasing."""
    ontology = _build_smoke_ontology()
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)

    related = next(a for a in atoms if a.iri.endswith("#relatedTo"))
    assert "object property" not in related.core_representation
    assert "applies to:" in related.core_representation.lower()
    assert "values restricted to:" in related.core_representation.lower()
    nh_prop = related.neighborhood_representation.lower()
    assert "it applies to concept" in nh_prop
    assert "it yields concept" in nh_prop

    alpha = next(a for a in atoms if a.iri.endswith("#Alpha"))
    nh_alpha = alpha.neighborhood_representation.lower()
    assert "it is a concept" in nh_alpha
    assert "related to beta" in nh_alpha

    atoms_d2 = GraphAtomizer().atomize(source=ontology, depth=2)
    related_d2 = next(a for a in atoms_d2 if a.iri.endswith("#relatedTo"))
    nh_rel_d2 = related_d2.neighborhood_representation.lower()
    assert "alpha" in nh_rel_d2 and "beta" in nh_rel_d2


def test_atomizer_class_neighborhood_includes_property_domain_range_clues_with_inverse() -> (
    None
):
    graph = RDFGraph._from_turtle_str(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <https://example.org/propdom#> .

        ex: a owl:Ontology .
        ex:Trial a owl:Class ; rdfs:label "Trial" .
        ex:Endpoint a owl:Class ; rdfs:label "Endpoint" .
        ex:forTrial a owl:ObjectProperty ;
            rdfs:label "for trial" ;
            rdfs:domain ex:Endpoint ;
            rdfs:range ex:Trial .
        ex:hasEndpoint a owl:ObjectProperty ;
            rdfs:label "has endpoint" ;
            rdfs:domain ex:Trial ;
            rdfs:range ex:Endpoint ;
            owl:inverseOf ex:forTrial .
        """
    )
    ontology = Ontology(graph=graph)
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)
    trial = next(a for a in atoms if a.iri.endswith("#Trial"))
    endpoint = next(a for a in atoms if a.iri.endswith("#Endpoint"))
    nh_trial = trial.neighborhood_representation.lower()
    nh_end = endpoint.neighborhood_representation.lower()
    assert "reverse" in nh_trial
    assert "endpoint" in nh_trial
    assert "for trial" in nh_trial
    assert "trial" in nh_end
    assert "endpoint" in nh_end
    assert "has endpoint" in nh_end


def _build_subclass_parent_label_ontology() -> Ontology:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix ex: <https://example.org/parentgloss#> .

        ex: a owl:Ontology .

        ex:Parent a rdfs:Class ;
            rdfs:label "Superclass label" .

        ex:Child a rdfs:Class ;
            rdfs:label "Child class" ;
            rdfs:subClassOf ex:Parent .

        ex:parentRel a rdf:Property ;
            rdfs:label "Parent relation label" .

        ex:childRel a rdf:Property ;
            rdfs:subPropertyOf ex:parentRel .

        ex:ThingA a rdfs:Class .
        ex:ThingB a rdfs:Class .
        ex:ThingA ex:childRel ex:ThingB .
        """
    )
    return Ontology(graph=graph)


def test_atomizer_excludes_standard_vocab_focal_iris_by_default() -> None:
    ontology = _build_smoke_ontology()
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)
    assert atoms
    for atom in atoms:
        assert not any(
            atom.iri.startswith(p) for p in STANDARD_VOCABULARY_NAMESPACE_PREFIXES
        )


def test_atomizer_multi_domain_namespaces_still_embedded_without_config() -> None:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix a: <https://a.example/ns#> .
        @prefix b: <https://b.example/ns#> .

        <https://a.example/ns> a owl:Ontology .

        a:ClassA a owl:Class ; rdfs:label "A class" .
        b:ClassB a owl:Class ; rdfs:label "B class" .
        """
    )
    ontology = Ontology(graph=graph)
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)
    iris = {a.iri for a in atoms}
    assert "https://a.example/ns#ClassA" in iris
    assert "https://b.example/ns#ClassB" in iris


def test_atomizer_embed_standard_vocab_iris_restores_vocab_focal_entities() -> None:
    ontology = _build_smoke_ontology()
    atoms = GraphAtomizer(embed_standard_vocab_iris=True).atomize(
        source=ontology, depth=1
    )
    assert any(
        a.iri.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#") for a in atoms
    )


def test_atomizer_extra_excluded_namespace_prefixes() -> None:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <https://example.org/embed#> .
        @prefix block: <https://blocked.example/vocab#> .

        ex: a owl:Ontology .

        block:Hidden a owl:Class ; rdfs:label "Should not focalize" .
        ex:Visible a owl:Class ; rdfs:label "Should focalize" .
        """
    )
    ontology = Ontology(graph=graph)
    atoms = GraphAtomizer(
        extra_excluded_namespace_prefixes=["https://blocked.example/vocab#"]
    ).atomize(source=ontology, depth=1)
    iris = {a.iri for a in atoms}
    assert "https://example.org/embed#Visible" in iris
    assert "https://blocked.example/vocab#Hidden" not in iris


def test_atomizer_hierarchy_clues_include_parent_label_gloss() -> None:
    """Parent IRIs in subclass / subproperty clues get rdfs:label when it adds signal."""
    ontology = _build_subclass_parent_label_ontology()
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)

    child = next(a for a in atoms if a.iri.endswith("#Child"))
    nh_class = child.neighborhood_representation.lower()
    assert "kind of" in nh_class
    assert "also described as" in nh_class
    assert "superclass label" in nh_class

    child_rel = next(a for a in atoms if a.iri.endswith("#childRel"))
    nh_prop = child_rel.neighborhood_representation.lower()
    assert "narrower form" in nh_prop
    assert "also described as" in nh_prop
    assert "parent relation label" in nh_prop


def test_atomizer_core_representation_includes_skos_alt_label() -> None:
    """skos:altLabel values appear in core text for ontology and facts atoms."""
    graph = RDFGraph._from_turtle_str(
        """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix ex: <https://example.org/alt#> .

        ex: a owl:Ontology .
        ex:Term a rdfs:Class ;
            skos:prefLabel "Preferred"@en ;
            skos:altLabel "Synonym A"@en ;
            skos:altLabel "Synonym B"@en .
        """
    )
    ontology = Ontology(graph=graph)
    atom = next(
        a
        for a in GraphAtomizer().atomize(source=ontology, depth=1)
        if a.iri.endswith("#Term")
    )
    core = atom.core_representation.lower()
    assert core.startswith("preferred")
    assert "synonym a" in core
    assert "synonym b" in core


def test_atomizer_minimal_representation_splits_iri_local_name() -> None:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix ex: <https://example.org/min#> .
        ex: a owl:Ontology .
        ex:MyVeryCoolClass a rdfs:Class ; rdfs:label "A label" .
        """
    )
    ontology = Ontology(graph=graph)
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)
    cool = next(a for a in atoms if a.iri.endswith("MyVeryCoolClass"))
    assert cool.core_representation.lower().startswith("a label")
    assert cool.minimal_representation == "my very cool class"


def test_embedding_config_default_bm25_model() -> None:
    cfg = EmbeddingConfig()
    assert cfg.bm25_model_name == "Qdrant/bm25"


def test_fastembed_bm25_sparse_tool_returns_qdrant_sparse_vectors() -> None:
    cfg = EmbeddingConfig()
    tool = FastembedBm25SparseTool(config=cfg)
    vectors = tool.embed_sparse(["alpha beta", "gamma delta"])
    assert len(vectors) == 2
    assert all(len(v.indices) == len(v.values) for v in vectors)
    assert all(len(v.indices) > 0 for v in vectors)


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


def test_bm25_sparse_vector_uses_dot_product_modifier_none() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    store = QdrantVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
        sparse_embedding=FastembedBm25SparseTool(config=embedding.config),
    )
    _, sparse_cfg = store._vectors_and_sparse_for_create()
    assert sparse_cfg is not None
    assert BM25_VECTOR_NAME in sparse_cfg
    assert sparse_cfg[BM25_VECTOR_NAME].modifier is None


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
    graph, source_iris = retriever.retrieve(query="alpha", top_k=2, expand_sparql=True)

    assert source_iris == ["https://example.org/smoke"]
    assert len(graph) > 0
    assert set(sparql_tool.last_entity_uris) == {atom.iri for atom in atoms}
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
    graph, source_iris = await retriever.aretrieve(
        query="alpha", top_k=2, expand_sparql=True
    )

    assert source_iris == ["https://example.org/smoke"]
    assert len(graph) > 0
    assert set(sparql_tool.last_entity_uris) == {atom.iri for atom in atoms}
    assert sparql_tool.last_ontology_iris == ["https://example.org/smoke"]
    assert sparql_tool.last_ontology_version_filters == {
        "https://example.org/smoke": {"1.0.0"}
    }
    assert sparql_tool.last_ontology_hash_filters == {
        "https://example.org/smoke": {"hash1"}
    }


@pytest.mark.anyio
async def test_aretrieve_ensemble_calls_induced_subgraph_once() -> None:
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
    graph, source_iris = await retriever.aretrieve_ensemble(
        queries=["alpha", "beta"],
        top_k=2,
        expand_sparql=True,
    )

    assert sparql_tool.induced_subgraph_calls == 1
    assert len(graph) > 0
    assert source_iris == ["https://example.org/smoke"]


def _scored_atom(atom_id: str, iri_local: str, score: float) -> OntologySearchHit:
    atom = GraphAtom(
        atom_id=atom_id,
        ontology_iri="https://example.org/smoke",
        ontology_id="smoke",
        ontology_hash="hash1",
        ontology_version="1.0.0",
        iri=f"https://example.org/smoke#{iri_local}",
        entity_role="resource",
        core_representation=f"core {atom_id}",
        neighborhood_representation="neighbor",
    )
    return OntologySearchHit(atom=atom, score=score)


def _channel_hits(
    core_hits: list[OntologySearchHit],
    neighborhood_hits: list[OntologySearchHit] | None = None,
    bm25_hits: list[OntologySearchHit] | None = None,
) -> OntologySearchHitsByChannel:
    return OntologySearchHitsByChannel(
        core_hits=core_hits,
        neighborhood_hits=neighborhood_hits or [],
        bm25_hits=bm25_hits or [],
    )


@pytest.mark.anyio
async def test_aretrieve_ensemble_per_query_ratio_keeps_weak_query_hits() -> None:
    """Weak-query hits survive vs a strong query because the cutoff is per-query relative."""
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    vector_store.set_hits_by_query(
        [
            _channel_hits(
                core_hits=[
                    _scored_atom("s1", "StrongTop", 0.95),
                    _scored_atom("s2", "StrongTail", 0.50),
                ]
            ),
            _channel_hits(
                core_hits=[
                    _scored_atom("w1", "WeakTop", 0.40),
                    _scored_atom("w2", "WeakMid", 0.35),
                ]
            ),
        ]
    )

    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(
            per_query_core_score_ratio=0.85,
            per_query_neighborhood_score_ratio=0.85,
            min_merged_max_score=0.0,
            merged_score_ratio=0.0,
            min_core_query_best_score=0.0,
            min_neighborhood_query_best_score=0.0,
        ),
    )
    await retriever.aretrieve_ensemble(
        queries=["q1", "q2"],
        top_k=4,
        expand_sparql=True,
    )

    # 0.50 < 0.95 * 0.85 -> strong tail out; weak query keeps 0.40 and 0.35 (floor 0.34).
    expected_iris = {
        "https://example.org/smoke#StrongTop",
        "https://example.org/smoke#WeakTop",
        "https://example.org/smoke#WeakMid",
    }
    assert set(sparql_tool.last_entity_uris) == expected_iris


@pytest.mark.anyio
async def test_aretrieve_ensemble_empty_when_merged_scores_below_floor() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    vector_store.set_hits_by_query(
        [
            _channel_hits(core_hits=[_scored_atom("a1", "A", 0.12)]),
            _channel_hits(core_hits=[_scored_atom("a2", "B", 0.11)]),
        ]
    )
    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(
            per_query_core_score_ratio=1.0,
            per_query_neighborhood_score_ratio=1.0,
            min_merged_max_score=2.0,
            merged_score_ratio=0.0,
            min_core_query_best_score=0.0,
            min_neighborhood_query_best_score=0.0,
        ),
    )
    graph, source_iris = await retriever.aretrieve_ensemble(
        queries=["q1", "q2"],
        top_k=2,
        expand_sparql=True,
    )
    assert len(graph) == 0
    assert source_iris == []
    assert sparql_tool.induced_subgraph_calls == 0


@pytest.mark.anyio
async def test_aretrieve_ensemble_drops_subquery_when_top_below_min_query_best() -> (
    None
):
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    vector_store.set_hits_by_query(
        [
            _channel_hits(core_hits=[_scored_atom("low", "LowEnt", 0.05)]),
            _channel_hits(core_hits=[_scored_atom("ok", "OkEnt", 0.80)]),
        ]
    )
    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(
            per_query_core_score_ratio=1.0,
            per_query_neighborhood_score_ratio=1.0,
            min_merged_max_score=0.0,
            merged_score_ratio=0.0,
            min_core_query_best_score=0.1,
            min_neighborhood_query_best_score=0.1,
        ),
    )
    await retriever.aretrieve_ensemble(
        queries=["q1", "q2"],
        top_k=2,
        expand_sparql=True,
    )
    assert sparql_tool.last_entity_uris == ["https://example.org/smoke#OkEnt"]


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


def test_search_hits_by_vector_returns_per_channel_typed_scores() -> None:
    class _Point:
        def __init__(self, point_id: str, score: float, neighborhood: str) -> None:
            self.id = point_id
            self.score = score
            self.payload = {"neighborhood_representation": neighborhood}

    class _Store(QdrantVectorStore):
        def _query_named_vector(
            self,
            vector_name: str,
            vector: Any,
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
    hits_by_channel = store.search_hits_by_vector(
        core_vector=[0.0] * 8,
        neighborhood_vector=[0.0] * 8,
        top_k=2,
    )

    assert len(hits_by_channel.core_hits) == 2
    assert len(hits_by_channel.neighborhood_hits) == 2
    assert hits_by_channel.core_hits[0].score >= hits_by_channel.core_hits[1].score
    assert (
        hits_by_channel.neighborhood_hits[0].score
        >= hits_by_channel.neighborhood_hits[1].score
    )
    assert hits_by_channel.core_hits[0].atom.score == hits_by_channel.core_hits[0].score


def test_search_hits_by_vector_dedupes_duplicate_iri_hits() -> None:
    class _Point:
        def __init__(
            self,
            point_id: str,
            score: float,
            iri: str,
            neighborhood: str = "neighbor",
        ) -> None:
            self.id = point_id
            self.score = score
            self.payload = {"neighborhood_representation": neighborhood, "iri": iri}

    class _Store(QdrantVectorStore):
        def _query_named_vector(
            self,
            vector_name: str,
            vector: Any,
            limit: int,
            search_filter,
        ):
            del vector, limit, search_filter
            if vector_name == "core":
                return [
                    _Point("v1", 0.91, "https://example.org/o#Same"),
                    _Point("v2", 0.86, "https://example.org/o#Same"),
                    _Point("v3", 0.80, "https://example.org/o#Other"),
                ]
            return [
                _Point("v4", 0.78, "https://example.org/o#Same"),
                _Point("v5", 0.74, "https://example.org/o#Other"),
            ]

        def _point_to_atom(self, point):
            return GraphAtom(
                atom_id=str(point.id),
                ontology_iri="https://example.org/o",
                iri=str(point.payload.get("iri", "")),
                entity_role="resource",
                core_representation="core",
                neighborhood_representation="neighbor",
            )

    store = _Store(
        config=QdrantConfig(
            embedding_batch_size=2,
            upsert_batch_size=2,
            dedup_mode=QdrantDedupMode.IRI,
            dedup_query_hits_by_iri=True,
        ),
        embedding=CountingEmbeddingTool(config=EmbeddingConfig(dimension=8)),
    )
    hits_by_channel = store.search_hits_by_vector(
        core_vector=[0.0] * 8,
        neighborhood_vector=[0.0] * 8,
        top_k=3,
    )
    assert [h.atom.iri for h in hits_by_channel.core_hits] == [
        "https://example.org/o#Same",
        "https://example.org/o#Other",
    ]
    assert [h.atom.iri for h in hits_by_channel.neighborhood_hits] == [
        "https://example.org/o#Same",
        "https://example.org/o#Other",
    ]


def test_point_id_for_atom_respects_dedup_mode() -> None:
    atom_a = GraphAtom(
        atom_id="atom-a",
        ontology_iri="https://example.org/o",
        ontology_version="1.0.0",
        ontology_hash="hash1",
        iri="https://example.org/o#Same",
        entity_role="resource",
        core_representation="core",
        neighborhood_representation="neighbor",
    )
    atom_b = GraphAtom(
        atom_id="atom-b",
        ontology_iri="https://example.org/o",
        ontology_version="1.0.0",
        ontology_hash="hash1",
        iri="https://example.org/o#Same",
        entity_role="resource",
        core_representation="core",
        neighborhood_representation="neighbor",
    )
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))

    strict_store = QdrantVectorStore(
        config=QdrantConfig(
            embedding_batch_size=2,
            upsert_batch_size=2,
            dedup_mode=QdrantDedupMode.IRI,
            dedup_include_version=True,
            dedup_include_hash=True,
        ),
        embedding=embedding,
    )
    relaxed_store = QdrantVectorStore(
        config=QdrantConfig(
            embedding_batch_size=2,
            upsert_batch_size=2,
            dedup_mode=QdrantDedupMode.ATOM_ID,
        ),
        embedding=embedding,
    )

    strict_id_a = strict_store._point_id_for_atom(atom_a)
    strict_id_b = strict_store._point_id_for_atom(atom_b)
    relaxed_id_a = relaxed_store._point_id_for_atom(atom_a)
    relaxed_id_b = relaxed_store._point_id_for_atom(atom_b)

    assert strict_id_a == strict_id_b
    assert relaxed_id_a != relaxed_id_b


def test_mmr_rerank_lambda_one_is_pure_relevance() -> None:
    atoms = [
        _scored_atom("a", "A", 0.91).atom,
        _scored_atom("b", "B", 0.83).atom,
        _scored_atom("c", "C", 0.74).atom,
    ]
    ranked = _mmr_rerank(
        atoms,
        vectors={},
        mmr_lambda=1.0,
        max_atoms=0,
        core_weight=0.7,
        neighborhood_weight=0.3,
    )
    assert [a.atom_id for a in ranked] == ["a", "b", "c"]


@pytest.mark.anyio
async def test_aretrieve_ensemble_mmr_promotes_diverse_candidates() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    vector_store.set_hits_by_query(
        [
            _channel_hits(
                core_hits=[
                    _scored_atom("a", "A", 0.95),
                    _scored_atom("b", "B", 0.93),
                    _scored_atom("c", "C", 0.90),
                ]
            )
        ]
    )
    vector_store.set_vectors(
        {
            "a": ([1.0, 0.0], [1.0, 0.0]),
            "b": ([0.98, 0.02], [0.98, 0.02]),
            "c": ([0.0, 1.0], [0.0, 1.0]),
        }
    )
    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(
            per_query_core_score_ratio=0.0,
            per_query_neighborhood_score_ratio=0.0,
            min_merged_max_score=0.0,
            merged_score_ratio=0.0,
            mmr_lambda=0.5,
            max_atoms=2,
        ),
    )
    await retriever.aretrieve_ensemble(
        queries=["q1"],
        top_k=3,
        expand_sparql=True,
    )
    assert set(sparql_tool.last_entity_uris) == {
        "https://example.org/smoke#A",
        "https://example.org/smoke#C",
    }


@pytest.mark.anyio
async def test_aretrieve_ensemble_rank_fusion_uses_rank_not_score() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(
            embedding_batch_size=2,
            upsert_batch_size=2,
            fusion_core_weight=0.5,
            fusion_neighborhood_weight=0.5,
            fusion_bm25_weight=0.0,
        ),
        embedding=embedding,
    )
    vector_store.set_hits_by_query(
        [
            _channel_hits(
                core_hits=[
                    _scored_atom("a", "A", 0.95),
                    _scored_atom("b", "B", 0.90),
                ],
                neighborhood_hits=[
                    _scored_atom("c", "C", 0.10),
                    _scored_atom("a", "A", 0.09),
                ],
            )
        ]
    )
    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(
            per_query_core_score_ratio=0.0,
            per_query_neighborhood_score_ratio=0.0,
            min_merged_max_score=0.0,
            merged_score_ratio=0.0,
            mmr_lambda=1.0,
            max_atoms=2,
        ),
    )
    await retriever.aretrieve_ensemble(
        queries=["q1"],
        top_k=2,
        expand_sparql=True,
    )
    assert set(sparql_tool.last_entity_uris) == {
        "https://example.org/smoke#A",
        "https://example.org/smoke#C",
    }


@pytest.mark.anyio
async def test_aretrieve_ensemble_lambda_one_skips_vector_fetch() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    vector_store.set_hits_by_query(
        [
            _channel_hits(
                core_hits=[
                    _scored_atom("a", "A", 0.95),
                    _scored_atom("b", "B", 0.90),
                ]
            )
        ]
    )
    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(
            per_query_core_score_ratio=0.0,
            per_query_neighborhood_score_ratio=0.0,
            min_merged_max_score=0.0,
            merged_score_ratio=0.0,
            mmr_lambda=1.0,
            max_atoms=2,
        ),
    )
    await retriever.aretrieve_ensemble(
        queries=["q1"],
        top_k=2,
        expand_sparql=True,
    )
    assert vector_store.afetch_vectors_calls == 0
    assert set(sparql_tool.last_entity_uris) == {
        "https://example.org/smoke#A",
        "https://example.org/smoke#B",
    }


@pytest.mark.anyio
async def test_aretrieve_ensemble_merged_score_ratio_trims_below_floor() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    vector_store.set_hits_by_query(
        [
            _channel_hits(
                core_hits=[
                    _scored_atom("a", "A", 0.95),
                    _scored_atom("b", "B", 0.84),
                ]
            )
        ]
    )
    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(
            per_query_core_score_ratio=0.0,
            per_query_neighborhood_score_ratio=0.0,
            min_merged_max_score=0.0,
            merged_score_ratio=0.9,
            mmr_lambda=1.0,
        ),
    )
    await retriever.aretrieve_ensemble(
        queries=["q1"],
        top_k=2,
        expand_sparql=True,
    )
    assert sparql_tool.last_entity_uris == ["https://example.org/smoke#A"]


@pytest.mark.anyio
async def test_aretrieve_ensemble_forwards_ranking_and_budget_controls() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    vector_store.set_hits_by_query(
        [
            _channel_hits(
                core_hits=[
                    _scored_atom("a", "A", 0.95),
                    _scored_atom("b", "B", 0.82),
                ]
            )
        ]
    )
    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(
            per_query_core_score_ratio=0.0,
            per_query_neighborhood_score_ratio=0.0,
            min_merged_max_score=0.0,
            merged_score_ratio=0.0,
            mmr_lambda=1.0,
        ),
    )
    await retriever.aretrieve_ensemble(
        queries=["q1"],
        top_k=2,
        expand_sparql=True,
        max_total_triples=77,
        estimated_triples_per_query=9,
    )

    assert sparql_tool.last_entity_uris == [
        "https://example.org/smoke#A",
        "https://example.org/smoke#B",
    ]
    assert sparql_tool.last_entity_relevance == {
        "https://example.org/smoke#A": pytest.approx(0.7 / 1.2),
        "https://example.org/smoke#B": pytest.approx(0.7 / 1.2 / 2),
    }
    assert sparql_tool.last_max_total_triples == 77
    assert sparql_tool.last_estimated_triples_per_query == 9


@pytest.mark.anyio
async def test_aretrieve_ensemble_patch_max_atoms_caps_output() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    vector_store.set_hits_by_query(
        [
            _channel_hits(
                core_hits=[
                    _scored_atom("a", "A", 0.95),
                    _scored_atom("b", "B", 0.93),
                    _scored_atom("c", "C", 0.92),
                ]
            )
        ]
    )
    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(
            per_query_core_score_ratio=0.0,
            per_query_neighborhood_score_ratio=0.0,
            min_merged_max_score=0.0,
            merged_score_ratio=0.0,
            mmr_lambda=1.0,
            max_atoms=2,
        ),
    )
    await retriever.aretrieve_ensemble(
        queries=["q1"],
        top_k=3,
        expand_sparql=True,
    )
    assert len(sparql_tool.last_entity_uris) == 2


def test_mmr_rerank_atoms_missing_vectors_fallback() -> None:
    atoms = [
        _scored_atom("a", "A", 0.95).atom,
        _scored_atom("b", "B", 0.90).atom,
        _scored_atom("c", "C", 0.89).atom,
    ]
    ranked = _mmr_rerank(
        atoms,
        vectors={
            "a": ([1.0, 0.0], [1.0, 0.0]),
            "c": ([0.99, 0.01], [0.99, 0.01]),
        },
        mmr_lambda=0.4,
        max_atoms=2,
        core_weight=0.7,
        neighborhood_weight=0.3,
    )
    # "b" has no vector, so it should rely on relevance and remain competitive.
    assert [a.atom_id for a in ranked] == ["a", "b"]


def test_normalize_core_neighborhood_weights_renormalizes_dense_lanes() -> None:
    qc = QdrantConfig(
        fusion_core_weight=0.7,
        fusion_neighborhood_weight=0.5,
        fusion_bm25_weight=0.2,
    )
    cw, nw = _normalize_core_neighborhood_weights(qc)
    assert abs(cw + nw - 1.0) < 1e-9
    assert cw > nw


def test_hybrid_merge_favors_dominant_single_window_hit() -> None:
    collected = [
        OntologySearchHit(atom=_scored_atom("dom", "Dominant", 1.2).atom, score=1.2),
        OntologySearchHit(atom=_scored_atom("w1", "Weak", 0.34).atom, score=0.34),
        OntologySearchHit(atom=_scored_atom("w2", "Weak", 0.33).atom, score=0.33),
        OntologySearchHit(atom=_scored_atom("w3", "Weak", 0.32).atom, score=0.32),
    ]
    merged = _merge_hits_across_queries_hybrid(
        collected,
        max_atoms_tier1=2,
        per_ontology_seed_quota=0,
        min_entity_score=0.3,
        max_atoms_total=2,
    )
    assert merged[0].atom.iri.endswith("#Dominant")
    assert merged[0].score == 1.2


def test_hybrid_merge_tier2_adds_per_ontology_coverage() -> None:
    matsci = "https://example.org/matsci"
    perov = "https://example.org/perov"

    def _hit(entity: str, onto: str, score: float) -> OntologySearchHit:
        atom = GraphAtom(
            atom_id=entity,
            ontology_iri=onto,
            iri=f"{onto}#{entity}",
            entity_role="resource",
            core_representation=entity,
            neighborhood_representation="",
            score=score,
        )
        return OntologySearchHit(atom=atom, score=score)

    collected = [
        _hit("M1", matsci, 0.95),
        _hit("M2", matsci, 0.90),
        _hit("M3", matsci, 0.85),
        _hit("P1", perov, 0.40),
    ]
    merged = _merge_hits_across_queries_hybrid(
        collected,
        max_atoms_tier1=2,
        per_ontology_seed_quota=1,
        min_entity_score=0.35,
        max_atoms_total=4,
    )
    iris = {hit.atom.iri for hit in merged}
    assert f"{matsci}#M1" in iris
    assert f"{matsci}#M2" in iris
    assert f"{perov}#P1" in iris


def test_max_score_merge_beats_rrf_frequency_bias() -> None:
    """Entity with one strong window outranks entity appearing weakly in many windows."""
    weak_repeated = [
        OntologySearchHit(
            atom=_scored_atom(f"w{i}", "Repeated", 0.31 + i * 0.01).atom,
            score=0.31 + i * 0.01,
        )
        for i in range(3)
    ]
    strong_once = [
        OntologySearchHit(atom=_scored_atom("s", "Strong", 1.0).atom, score=1.0),
    ]
    max_score = _merge_hits_across_queries_max_score(weak_repeated + strong_once)
    assert max_score[0].atom.iri.endswith("#Strong")


def test_expand_ontology_iris_by_reference_includes_cross_ontology_parent() -> None:
    matsci_iri = "https://growgraph.dev/ontologies/matsci-ontology"
    perov_iri = "https://growgraph.dev/ontologies/perovskitemat"
    matsci_graph = RDFGraph._from_turtle_str(
        f"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix matsci: <https://growgraph.dev/ontologies/matsci-ontology#> .
        @prefix perov: <https://growgraph.dev/ontologies/perovskitemat#> .

        <{matsci_iri}> a owl:Ontology .
        matsci:PerovskiteQD a owl:Class ;
            rdfs:subClassOf perov:PerovskiteNanocrystal .
        """
    )
    perov_graph = RDFGraph._from_turtle_str(
        f"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix perov: <https://growgraph.dev/ontologies/perovskitemat#> .

        <{perov_iri}> a owl:Ontology .
        perov:PerovskiteNanocrystal a owl:Class ;
            rdfs:label "Perovskite nanocrystal" .
        """
    )
    ontologies = [
        Ontology(iri=matsci_iri, graph=matsci_graph, title="matsci"),
        Ontology(iri=perov_iri, graph=perov_graph, title="perov"),
    ]
    expanded = _expand_ontology_iris_by_reference(
        ["https://growgraph.dev/ontologies/matsci-ontology#PerovskiteQD"],
        [matsci_iri],
        ontologies,
    )
    assert perov_iri in expanded


@pytest.mark.anyio
async def test_aretrieve_ensemble_rrf_mode_regression() -> None:
    embedding = CountingEmbeddingTool(config=EmbeddingConfig(dimension=8))
    vector_store = StubVectorStore(
        config=QdrantConfig(embedding_batch_size=2, upsert_batch_size=2),
        embedding=embedding,
    )
    vector_store.set_hits_by_query(
        [
            _channel_hits(core_hits=[_scored_atom("a", "A", 0.95)]),
            _channel_hits(core_hits=[_scored_atom("a", "A", 0.94)]),
            _channel_hits(core_hits=[_scored_atom("a", "A", 0.93)]),
            _channel_hits(core_hits=[_scored_atom("b", "B", 0.99)]),
        ]
    )
    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(
            cross_query_merge_mode=CrossQueryMergeMode.RRF,
            per_query_core_score_ratio=0.0,
            min_merged_max_score=0.0,
            merged_score_ratio=0.0,
            mmr_lambda=1.0,
            max_atoms=2,
        ),
    )
    await retriever.aretrieve_ensemble(
        queries=["q1", "q2", "q3", "q4"], top_k=1, expand_sparql=True
    )
    assert sparql_tool.last_entity_uris[0].endswith("#A")
