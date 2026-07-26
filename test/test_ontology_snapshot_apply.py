"""Tests for OntologySnapshot and namespace-ownership apply."""

from __future__ import annotations

from rdflib import OWL, RDF, URIRef

from ontocast.onto.enum import OntologyAssemblyMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_apply import (
    complement_inserts,
    partition_inserts_by_namespace,
)
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import OntologyManager


def _onto_graph(iri: str, local: str, prefix: str) -> RDFGraph:
    g = RDFGraph()
    ns = f"{iri}#"
    g.bind(prefix, ns)
    g.add((URIRef(iri), RDF.type, OWL.Ontology))
    g.add((URIRef(f"{ns}{local}"), RDF.type, OWL.Class))
    return g


def test_ontology_snapshot_from_ontology_copies_graph_and_source() -> None:
    iri = "https://example.org/a"
    ontology = Ontology(graph=_onto_graph(iri, "A", "exa"), iri=iri)
    snap = OntologySnapshot.from_ontology(
        ontology, assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY
    )
    assert snap.source_iris == [iri]
    assert len(snap.graph) == len(ontology.graph)
    assert snap.graph is not ontology.graph
    assert "iri" not in OntologySnapshot.model_fields


def test_ontology_snapshot_from_graph_preserves_sources() -> None:
    g = RDFGraph()
    g.add(
        (
            URIRef("https://example.org/a#A"),
            RDF.type,
            OWL.Class,
        )
    )
    snap = OntologySnapshot.from_graph(
        g,
        source_iris=["https://example.org/a", "https://example.org/b"],
        assembly_mode=OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE,
    )
    assert snap.source_iris == ["https://example.org/a", "https://example.org/b"]
    assert not snap.is_empty()
    assert "Source catalog IRIs" in snap.describe_for_prompt()


def test_complement_inserts_subtracts_snapshot_triples() -> None:
    snap = RDFGraph()
    s = URIRef("https://example.org/a#Existing")
    snap.add((s, RDF.type, OWL.Class))
    inserts = RDFGraph()
    inserts.add((s, RDF.type, OWL.Class))
    new = URIRef("https://example.org/a#New")
    inserts.add((new, RDF.type, OWL.Class))
    result = complement_inserts(inserts, snap)
    assert (new, RDF.type, OWL.Class) in result
    assert (s, RDF.type, OWL.Class) not in result


def test_partition_inserts_by_namespace_routes_to_owner() -> None:
    iri_a = "https://example.org/a"
    iri_b = "https://example.org/b"
    mgr = OntologyManager()
    mgr.add_ontology(
        Ontology(graph=_onto_graph(iri_a, "A", "exa"), iri=iri_a),
        skip_vector_index=True,
    )
    mgr.add_ontology(
        Ontology(graph=_onto_graph(iri_b, "B", "exb"), iri=iri_b),
        skip_vector_index=True,
    )

    inserts = RDFGraph()
    inserts.add((URIRef(f"{iri_a}#NewA"), RDF.type, OWL.Class))
    inserts.add((URIRef(f"{iri_b}#NewB"), RDF.type, OWL.Class))
    inserts.add(
        (
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
            RDF.type,
            OWL.DatatypeProperty,
        )
    )

    partitioned, unattributed = partition_inserts_by_namespace(
        inserts,
        writable_iris=[iri_a, iri_b],
        ontology_manager=mgr,
    )
    assert iri_a in partitioned
    assert iri_b in partitioned
    assert (URIRef(f"{iri_a}#NewA"), RDF.type, OWL.Class) in partitioned[iri_a]
    assert (URIRef(f"{iri_b}#NewB"), RDF.type, OWL.Class) in partitioned[iri_b]
    assert unattributed >= 1
