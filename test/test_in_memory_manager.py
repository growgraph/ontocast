"""Tests for InMemoryTripleStoreManager."""

from __future__ import annotations

import asyncio

import pyoxigraph as ox
import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID

from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_header import CatalogContractError
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.triple_manager.in_memory import (
    InMemoryTripleStoreManager,
    _rdflib_graph_to_quads,
)
from ontocast.tool.triple_manager.util import head_headers
from test.ontology_helpers import next_version, stamped_ontology

pytestmark = pytest.mark.unit

_SAMPLE_IRI = "https://example.org/test"


def _sample_ontology() -> Ontology:
    return stamped_ontology(
        f"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <{_SAMPLE_IRI}#> .

        <{_SAMPLE_IRI}> a owl:Ontology ;
            rdfs:label "Test Ontology" .

        ex:Thing a rdfs:Class ;
            rdfs:label "Thing" .
        """,
        ontology_id="test",
    )


def _versioned_ontology(
    iri: str, extra_classes: int, base: Ontology | None
) -> Ontology:
    """A root, or the stamped child of ``base``, whose content grows with size."""
    graph = RDFGraph()
    graph.add((URIRef(iri), RDF.type, OWL.Ontology))
    graph.add((URIRef(iri), RDFS.label, Literal("Versioned Ontology")))
    for n in range(extra_classes):
        thing = URIRef(f"{iri}#Thing{n}")
        graph.add((thing, RDF.type, RDFS.Class))
        graph.add((thing, RDFS.label, Literal(f"Thing {n}")))
    if base is None:
        return stamped_ontology(graph, ontology_id=iri.rsplit("/", 1)[-1])
    return next_version(base, graph)


def test_in_memory_roundtrip_ontology() -> None:
    manager = InMemoryTripleStoreManager()
    ontology = _sample_ontology()
    manager.serialize(ontology)
    fetched = manager.fetch_ontologies()
    assert len(fetched) == 1
    assert fetched[0].iri == ontology.iri
    assert fetched[0].version_iri == ontology.version_iri
    assert fetched[0].ontology_id == ontology.ontology_id
    assert len(fetched[0].graph) == len(ontology.graph)


def test_in_memory_refuses_an_unstamped_ontology() -> None:
    manager = InMemoryTripleStoreManager()
    unstamped = Ontology(graph=_sample_ontology().graph, iri=_SAMPLE_IRI)
    with pytest.raises(ValueError, match="not stamped as a version"):
        manager.serialize(unstamped)
    assert manager.fetch_ontologies() == []


def test_in_memory_serializes_facts_graph() -> None:
    manager = InMemoryTripleStoreManager()
    facts = RDFGraph._from_turtle_str(
        """
        @prefix ex: <https://example.org/facts#> .
        ex:s ex:p ex:o .
        """
    )
    graph_uri = "https://example.org/facts/graph1"
    manager.serialize(facts, graph_uri=graph_uri)
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
        assert rows[0]["onto"] == _SAMPLE_IRI
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
    """Headers must reproduce lineage well enough to pick the same heads.

    Everything that avoids materializing graphs rests on this equivalence.
    """

    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        iri = "https://example.org/versioned"
        v1 = _versioned_ontology(iri, extra_classes=1, base=None)
        v2 = _versioned_ontology(iri, extra_classes=2, base=v1)
        standalone = _sample_ontology()
        for onto in (v1, v2, standalone):
            manager.serialize(onto)

        headers = await manager.afetch_ontology_catalog()
        # One header per stored *version*, not per head.
        assert len(headers) == 3
        by_graph = {header.graph_uri: header for header in headers}
        assert v1.version_iri and v2.version_iri
        assert by_graph[v1.version_iri].version_iri == v1.version_iri
        assert by_graph[v1.version_iri].prior_version_iris == []
        assert by_graph[v2.version_iri].prior_version_iris == [v1.version_iri]
        assert by_graph[v1.version_iri].iri == iri
        assert by_graph[v2.version_iri].version == v2.version
        assert by_graph[v2.version_iri].ontology_id == v2.ontology_id

        head_graphs = {header.graph_uri for header in head_headers(headers)}
        materialized = {onto.version_iri for onto in await manager.afetch_ontologies()}
        assert head_graphs == materialized
        # v1 is a prior of v2, so only v2 survives.
        assert v2.version_iri in head_graphs
        assert v1.version_iri not in head_graphs

    asyncio.run(main())


def test_fetch_ontologies_returns_only_heads() -> None:
    """Two stored versions of one ontology materialize as its one head."""
    manager = InMemoryTripleStoreManager()
    iri = "https://example.org/versioned"
    v1 = _versioned_ontology(iri, extra_classes=1, base=None)
    v2 = _versioned_ontology(iri, extra_classes=2, base=v1)
    manager.serialize(v1)
    manager.serialize(v2)

    fetched = manager.fetch_ontologies()

    assert [onto.version_iri for onto in fetched] == [v2.version_iri]
    assert fetched[0].prior_version_iris == [v1.version_iri]
    assert fetched[0].hash == v2.hash


def test_a_non_version_graph_in_the_ontologies_dataset_is_a_fault() -> None:
    """A graph without the version header is a contract error, never a gap."""

    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        manager.serialize(_sample_ontology())
        foreign = RDFGraph._from_turtle_str(
            """
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            <https://example.org/foreign> a owl:Ontology ;
                owl:versionInfo "1.0.0" .
            """
        )
        manager.serialize_graph(
            foreign, graph_uri="https://example.org/foreign", store="ontologies"
        )

        with pytest.raises(CatalogContractError) as excinfo:
            await manager.afetch_ontology_catalog()
        assert excinfo.value.graph_uri == "https://example.org/foreign"
        with pytest.raises(CatalogContractError):
            await manager.afetch_ontologies()

    asyncio.run(main())


def test_afetch_ontologies_by_iri_restricts_and_materializes() -> None:
    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        wanted = _sample_ontology()
        other = _versioned_ontology("https://example.org/other", 1, base=None)
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
    """An author's ``owl:versionInfo "1.0"`` is stored and read as ``1.0.0``."""

    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        loose = stamped_ontology(
            """
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <https://example.org/loose#> .

            <https://example.org/loose> a owl:Ontology ;
                owl:versionInfo "1.0" .
            ex:Thing a rdfs:Class .
            """,
            ontology_id="loose",
        )
        manager.serialize(loose)

        headers = await manager.afetch_ontology_catalog()
        assert [header.version for header in headers] == ["1.0.0"]

    asyncio.run(main())


def test_in_memory_supports_sparql_construct() -> None:
    assert InMemoryTripleStoreManager().supports_sparql_construct() is True


def test_aconstruct_returns_typed_terms() -> None:
    """Unlike aselect rows, a CONSTRUCT result carries real RDF terms."""

    async def main() -> None:
        manager = InMemoryTripleStoreManager()
        sample = _sample_ontology()
        manager.serialize(sample)

        graph = await manager.aconstruct(
            "CONSTRUCT { ?s ?p ?o } WHERE { GRAPH ?g { ?s ?p ?o } }"
        )
        # The stored graph is the stamped version graph, header and prefix
        # declarations included.
        assert len(graph) == len(sample.graph)
        labels = list(graph.objects(URIRef(f"{_SAMPLE_IRI}#Thing"), RDFS.label))
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


def test_unserializable_object_names_term_position_and_graph() -> None:
    """A bad term must be diagnosable, not a bare AssertionError.

    ``to_ox`` maps ``urn:x-rdflib:default`` to a DefaultGraph, which is not a
    term, so a graph carrying it in a triple position used to trip an
    unmessaged assert far from the code that built the graph.
    """
    graph = RDFGraph()
    graph.add((URIRef("urn:s"), URIRef("urn:p"), DATASET_DEFAULT_GRAPH_ID))
    with pytest.raises(TypeError) as excinfo:
        _rdflib_graph_to_quads(graph, ox.NamedNode("urn:g"))
    message = str(excinfo.value)
    assert "object" in message
    assert "urn:x-rdflib:default" in message
    assert "urn:g" in message


def test_unserializable_subject_names_position() -> None:
    graph = RDFGraph()
    graph.add((Literal("a"), URIRef("urn:p"), Literal("b")))
    with pytest.raises(TypeError) as excinfo:
        _rdflib_graph_to_quads(graph, ox.NamedNode("urn:g"))
    assert "subject" in str(excinfo.value)


def test_valid_graph_still_converts() -> None:
    graph = RDFGraph()
    graph.add((URIRef("urn:s"), RDFS.label, Literal("x")))
    quads = _rdflib_graph_to_quads(graph, ox.NamedNode("urn:g"))
    assert len(quads) == 1
    assert quads[0].graph_name == ox.NamedNode("urn:g")
