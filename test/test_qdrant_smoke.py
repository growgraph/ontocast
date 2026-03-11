"""Smoke test for Qdrant ontology atom indexing and retrieval."""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest

from ontocast.config import EmbeddingConfig, QdrantConfig
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.vector_store import (
    EmbeddingTool,
    OntologyPatchRetriever,
    QdrantVectorStore,
)
from ontocast.util import render_text_hash


class DeterministicEmbeddingTool(EmbeddingTool):
    """Lightweight deterministic embedding for smoke/integration tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = render_text_hash(text, digits=None)
            seed = int(digest[:16], 16)
            vector = [
                (((seed + i * 97) % 2000) / 1000.0) - 1.0
                for i in range(self.config.dimension)
            ]
            vectors.append(vector)
        return vectors


def _qdrant_available(uri: str, api_key: str | None) -> bool:
    candidates = [api_key] if api_key else [None, "abc123-qwe"]
    for candidate in candidates:
        headers = {"api-key": candidate} if candidate else None
        try:
            response = httpx.get(
                f"{uri.rstrip('/')}/collections",
                headers=headers,
                timeout=2.0,
            )
            if response.status_code == 200:
                return True
        except Exception:
            continue
    return False


def _build_smoke_ontology() -> Ontology:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix ex: <https://example.org/smoke#> .
        @prefix schema: <https://schema.org/> .

        ex: a owl:Ontology ;
            rdfs:label "Smoke Ontology" ;
            rdfs:comment "Ontology used for Qdrant smoke testing." .

        ex:Concept a rdfs:Class ;
            rdfs:label "Concept" ;
            rdfs:subClassOf schema:Thing .

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


def test_qdrant_vector_store_smoke() -> None:
    """Initialize Qdrant, index ontology atoms, and retrieve patch context."""
    uri = os.getenv("QDRANT_URI", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY", "abc123-qwe")
    if not _qdrant_available(uri=uri, api_key=api_key):
        pytest.skip(f"Qdrant is not reachable at {uri}")

    collection_name = f"ontocast_smoke_{uuid.uuid4().hex[:8]}"
    embedding_config = EmbeddingConfig(dimension=8)
    embedding_tool = DeterministicEmbeddingTool(config=embedding_config)
    vector_store = QdrantVectorStore(
        config=QdrantConfig(
            uri=uri,
            api_key=api_key,
            collection=collection_name,
            vector_size=embedding_config.dimension,
        ),
        embedding=embedding_tool,
    )

    ontology = _build_smoke_ontology()

    try:
        asyncio.run(vector_store.initialize())
        indexed = vector_store.index_ontology(ontology=ontology)
        assert indexed > 0

        hits = vector_store.search_patches(query="alpha concept relation", top_k=5)
        assert len(hits) > 0
        assert any(hit.ontology_iri == ontology.iri for hit in hits)
        assert all(hit.ontology_version == ontology.version for hit in hits)
        assert all(hit.core_representation for hit in hits)
        assert all(hit.neighborhood_representation for hit in hits)

        filtered_version_hits = vector_store.search_patches(
            query="alpha concept relation",
            top_k=5,
            filter_version=ontology.version,
        )
        assert len(filtered_version_hits) > 0
        assert all(
            hit.ontology_version == ontology.version for hit in filtered_version_hits
        )

        retriever = OntologyPatchRetriever(vector_store=vector_store, sparql_tool=None)
        patch_graph, atoms = retriever.retrieve(query="beta concept", top_k=3)
        assert len(atoms) > 0
        assert len(patch_graph) == 0

        vector_store.delete_ontology(ontology.iri)
        filtered_hits = vector_store.search_patches(
            query="alpha concept relation",
            top_k=5,
            filter_iri=ontology.iri,
        )
        assert filtered_hits == []
    finally:
        vector_store.client.delete_collection(collection_name=collection_name)
