"""Smoke test for Qdrant via ToolBox: ingest TTL, index atoms, retrieve patches."""

from __future__ import annotations

import asyncio

from ontocast.config import Config, EmbeddingConfig, LLMConfig, PathConfig, ToolConfig
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.toolbox import ToolBox
from test.qdrant_util import DeterministicEmbeddingTool, QdrantSessionTestContext


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


def _build_toolbox(ctx: QdrantSessionTestContext) -> ToolBox:
    embedding_config = EmbeddingConfig(dimension=8, model_name="pytest-smoke")
    tool_config = ToolConfig(
        llm_config=LLMConfig(),
        path_config=PathConfig(
            working_directory=ctx.working_directory,
            ontology_directory=ctx.ontology_directory,
        ),
        embedding=embedding_config,
        qdrant=ctx.qdrant_config,
    )
    return ToolBox(Config(tool_config=tool_config))


def test_qdrant_vector_store_smoke(
    qdrant_session_test_context: QdrantSessionTestContext,
) -> None:
    """Ingest ontology via ToolBox; Qdrant indexing and patch retrieval."""
    ctx = qdrant_session_test_context
    embedding_config = EmbeddingConfig(dimension=8, model_name="pytest-smoke")
    tools = _build_toolbox(ctx)

    assert tools.vector_store is not None
    assert tools.patch_retriever is not None
    deterministic = DeterministicEmbeddingTool(config=embedding_config)
    tools.embedding_tool = deterministic
    tools.vector_store.embedding = deterministic

    ontology = _build_smoke_ontology()
    ttl_bytes = ontology.graph.serialize(format="turtle").encode("utf-8")

    async def _run() -> Ontology:
        # _materialize_ontology indexes only when vector_store_ready; initialize() on
        # the store alone does not set it (update_tenancy would but would change collection names).
        if tools.vector_store is not None:
            await tools.vector_store.initialize()
            tools.vector_store_ready = True
            tools.vector_store_last_error = None
        return await tools.ingest_ontology_ttl(ttl_bytes)

    ingested = asyncio.run(_run())

    vector_store = tools.vector_store
    indexed_iri = ingested.iri

    hits = vector_store.search_patches(query="alpha concept relation", top_k=5)
    assert len(hits) > 0
    assert len({hit.iri for hit in hits}) == len(hits)
    assert any(hit.ontology_iri == indexed_iri for hit in hits)
    assert all(hit.ontology_version == ingested.version for hit in hits)
    assert all(hit.core_representation for hit in hits)
    assert any(hit.neighborhood_representation for hit in hits)
    predicate_hits = [h for h in hits if h.entity_role == "predicate"]
    assert predicate_hits
    assert all(h.neighborhood_representation for h in predicate_hits)

    filtered_version_hits = vector_store.search_patches(
        query="alpha concept relation",
        top_k=5,
        filter_version=ingested.version,
    )
    assert len(filtered_version_hits) > 0
    assert all(
        hit.ontology_version == ingested.version for hit in filtered_version_hits
    )

    patch_graph, source_iris = tools.patch_retriever.retrieve(
        query="beta concept", top_k=3
    )
    assert len(source_iris) > 0
    assert len(patch_graph) > 0

    vector_store.delete_ontology(indexed_iri)
    filtered_hits = vector_store.search_patches(
        query="alpha concept relation",
        top_k=5,
        filter_iri=indexed_iri,
    )
    assert filtered_hits == []
