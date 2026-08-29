"""Tests for InMemoryTripleStoreManager."""

from __future__ import annotations

import asyncio

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import RDFS

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager
from ontocast.tool.triple_manager.util import dedupe_terminal_ontologies


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


def _versioned_ontology(iri: str, extra_classes: int, parents: list[str]) -> Ontology:
    """Build an ontology whose content -- and therefore hash -- varies with size."""
    classes = "\n".join(
        f'        ex:Thing{n} a rdfs:Class ; rdfs:label "Thing {n}" .'
        for n in range(extra_classes)
    )
    graph = RDFGraph._from_turtle_str(
        f"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <{iri}#> .

        <{iri}> a owl:Ontology ;
            rdfs:label "Versioned Ontology" .
{classes}
        """
    )
    return Ontology(graph=graph, iri=iri, parent_hashes=parents)


def test_in_memory_supports_sparql_select() -> None:
    assert InMemoryTripleStoreManager().supports_sparql_select() is True


def test_aselect_returns_lexical_values_and_omits_unbound() -> None:
    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        manager.serialize(_sample_ontology())

        rows = await manager.aselect(
            """
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?onto ?label ?missing WHERE {
              GRAPH ?g {
                ?onto a owl:Ontology .
                OPTIONAL { ?onto rdfs:label ?label }
                OPTIONAL { ?onto rdfs:seeAlso ?missing }
              }
            }
            """
        )

        assert len(rows) == 1
        # IRIs and literals both arrive as their lexical form, no <> or quotes.
        assert rows[0]["onto"] == "https://example.org/test"
        assert rows[0]["label"] == "Test Ontology"
        assert "missing" not in rows[0]

    asyncio.run(main())


def test_aselect_rejects_non_select_query() -> None:
    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        manager.serialize(_sample_ontology())
        with pytest.raises(TypeError):
            await manager.aselect("ASK { ?s ?p ?o }")

    asyncio.run(main())


def test_ontology_catalog_matches_materialized_catalog() -> None:
    """Headers must reproduce lineage well enough to pick the same terminals.

    Everything that avoids materializing graphs rests on this equivalence.
    """

    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        iri = "https://example.org/versioned"
        v1 = _versioned_ontology(iri, extra_classes=1, parents=[])
        v2 = _versioned_ontology(iri, extra_classes=2, parents=[str(v1.hash)])
        standalone = _sample_ontology()
        for onto in (v1, v2, standalone):
            manager.serialize(onto)

        headers = await manager.afetch_ontology_catalog()
        # One header per stored *version*, not per terminal ontology.
        assert len(headers) == 3
        by_graph = {header.graph_uri: header for header in headers}
        assert by_graph[v1.versioned_iri].hash == v1.hash
        assert by_graph[v2.versioned_iri].hash == v2.hash
        assert by_graph[v2.versioned_iri].parent_hashes == [str(v1.hash)]
        assert by_graph[v1.versioned_iri].iri == iri
        assert by_graph[v2.versioned_iri].version == v2.version

        terminal_headers = dedupe_terminal_ontologies(headers)
        terminal_ontologies = dedupe_terminal_ontologies(
            await manager.afetch_ontologies()
        )
        assert {header.graph_uri for header in terminal_headers} == {
            onto.versioned_iri for onto in terminal_ontologies
        }
        # v1 is a parent of v2, so only v2 survives.
        assert v2.versioned_iri in {header.graph_uri for header in terminal_headers}
        assert v1.versioned_iri not in {header.graph_uri for header in terminal_headers}

    asyncio.run(main())


def test_afetch_ontologies_by_iri_restricts_and_materializes() -> None:
    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        wanted = _sample_ontology()
        other = _versioned_ontology(
            "https://example.org/other", extra_classes=1, parents=[]
        )
        manager.serialize(wanted)
        manager.serialize(other)

        selected = await manager.afetch_ontologies_by_iri([wanted.iri])
        assert [onto.iri for onto in selected] == [wanted.iri]
        assert len(selected[0].graph) == len(wanted.graph)

        # Empty means "no restriction", matching the induced-subgraph filter.
        assert len(await manager.afetch_ontologies_by_iri([])) == 2
        assert (
            await manager.afetch_ontologies_by_iri(["https://example.org/absent"]) == []
        )

    asyncio.run(main())


def test_catalog_reads_respect_tenancy() -> None:
    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        ontology = _sample_ontology()

        await manager.update_tenancy("tenant_a", "project_a")
        manager.serialize(ontology)
        assert len(await manager.afetch_ontology_catalog()) == 1

        await manager.update_tenancy("tenant_b", "project_b")
        assert await manager.afetch_ontology_catalog() == []
        assert await manager.afetch_ontologies_by_iri([ontology.iri]) == []

        await manager.update_tenancy("tenant_a", "project_a")
        assert len(await manager.afetch_ontology_catalog()) == 1

    asyncio.run(main())


def test_catalog_normalizes_non_semantic_version() -> None:
    """A store holding ``owl:versionInfo "1.0"`` must not fail header validation."""

    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        graph = RDFGraph._from_turtle_str(
            """
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <https://example.org/loose#> .

            <https://example.org/loose> a owl:Ontology ;
                owl:versionInfo "1.0" .
            ex:Thing a rdfs:Class .
            """
        )
        manager.serialize_graph(
            graph,
            graph_uri="https://example.org/loose",
            store="ontologies",
        )

        headers = await manager.afetch_ontology_catalog()
        assert [header.version for header in headers] == ["1.0.0"]

    asyncio.run(main())


def test_in_memory_supports_sparql_construct() -> None:
    assert InMemoryTripleStoreManager().supports_sparql_construct() is True


def test_aconstruct_returns_typed_terms() -> None:
    """Unlike aselect rows, a CONSTRUCT result carries real RDF terms."""

    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        manager.serialize(_sample_ontology())

        graph = await manager.aconstruct(
            "CONSTRUCT { ?s ?p ?o } WHERE { GRAPH ?g { ?s ?p ?o } }"
        )
        # Serialization persists author @prefix bindings as one sh:declare
        # blank node (3 triples) on top of the ontology's own content.
        assert len(graph) == len(_sample_ontology().graph) + 3
        labels = list(
            graph.objects(
                URIRef("https://example.org/test#Thing"),
                RDFS.label,
            )
        )
        assert labels == [Literal("Thing")]

    asyncio.run(main())


def test_aconstruct_rejects_select_query() -> None:
    """A wrong query kind must fail cleanly, not leak an unsendable result handle."""

    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        manager.serialize(_sample_ontology())
        with pytest.raises(TypeError):
            await manager.aconstruct("SELECT ?s WHERE { GRAPH ?g { ?s ?p ?o } }")

    asyncio.run(main())


def test_aconstruct_respects_tenancy() -> None:
    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        manager.serialize(_sample_ontology())
        query = "CONSTRUCT { ?s ?p ?o } WHERE { GRAPH ?g { ?s ?p ?o } }"
        assert len(await manager.aconstruct(query)) > 0

        await manager.update_tenancy("other", "project")
        assert len(await manager.aconstruct(query)) == 0

    asyncio.run(main())
