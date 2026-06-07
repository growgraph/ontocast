"""Tests for InMemoryTripleStoreManager."""

from __future__ import annotations

import asyncio

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager


def _sample_ontology() -> Ontology:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <https://example.org/test#> .

        <https://example.org/test> a owl:Ontology ;
            rdfs:label "Test Ontology" .

        ex:Thing a rdfs:Class ;
            rdfs:label "Thing" .
        """
    )
    return Ontology(graph=graph, iri="https://example.org/test")


def test_in_memory_roundtrip_ontology() -> None:
    manager = InMemoryTripleStoreManager()
    ontology = _sample_ontology()
    assert manager.serialize(ontology) is True
    fetched = manager.fetch_ontologies()
    assert len(fetched) == 1
    assert fetched[0].iri == ontology.iri
    assert len(fetched[0].graph) == len(ontology.graph)


def test_in_memory_serializes_facts_graph() -> None:
    manager = InMemoryTripleStoreManager()
    facts = RDFGraph._from_turtle_str(
        """
        @prefix ex: <https://example.org/facts#> .
        ex:s ex:p ex:o .
        """
    )
    graph_uri = "https://example.org/facts/graph1"
    assert manager.serialize(facts, graph_uri=graph_uri) is True
    partition = manager._active_partition()
    assert len(partition.facts) > 0


def test_in_memory_tenancy_isolation() -> None:
    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        onto_a = _sample_ontology()

        await manager.update_tenancy("tenant_a", "project_a")
        manager.serialize(onto_a)

        await manager.update_tenancy("tenant_b", "project_b")
        assert manager.fetch_ontologies() == []

        await manager.update_tenancy("tenant_a", "project_a")
        fetched = manager.fetch_ontologies()
        assert len(fetched) == 1
        assert fetched[0].iri == onto_a.iri

    asyncio.run(main())


def test_in_memory_clean_and_clean_tenancy() -> None:
    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        ontology = _sample_ontology()
        manager.serialize(ontology)
        assert manager.fetch_ontologies()

        await manager.clean()
        assert manager.fetch_ontologies() == []

        manager.serialize(ontology)
        await manager.update_tenancy("other", "proj")
        manager.serialize(ontology)
        await manager.clean_tenancy("other", "proj")

        await manager.update_tenancy("ontocast", "test")
        assert manager.fetch_ontologies()

    asyncio.run(main())


def test_in_memory_drop_ontology_graphs() -> None:
    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        ontology = _sample_ontology()
        manager.serialize(ontology)
        assert manager.fetch_ontologies()

        await manager.drop_all_ontology_graphs_for_iri(ontology.iri)
        assert manager.fetch_ontologies() == []

    asyncio.run(main())


def test_in_memory_supports_tenancy_partition() -> None:
    manager = InMemoryTripleStoreManager()
    assert manager.supports_tenancy_partition() is True
