"""Unit tests for the retrieval ground-truth attribution helpers."""

from __future__ import annotations

from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from test.retrieval_gt import owner_index, owner_of


def _versioned_header_ontology() -> Ontology:
    """QUDT-style vocabulary: header IRI is not a prefix of its term IRIs."""
    graph = RDFGraph()
    graph.bind("unit", "http://example.org/vocab/unit/")
    graph.add((URIRef("http://example.org/2.1/vocab/unit"), RDF.type, OWL.Ontology))
    graph.add(
        (
            URIRef("http://example.org/vocab/unit/MilliEV"),
            RDF.type,
            OWL.NamedIndividual,
        )
    )
    graph.add(
        (
            URIRef("http://example.org/vocab/unit/MilliEV"),
            RDFS.label,
            Literal("millielectronvolt"),
        )
    )
    return Ontology(graph=graph, iri="http://example.org/2.1/vocab/unit")


def test_owner_of_namespace_containment() -> None:
    graph = RDFGraph()
    graph.add((URIRef("https://example.org/onto"), RDF.type, OWL.Ontology))
    graph.add((URIRef("https://example.org/onto#Thing"), RDF.type, OWL.Class))
    ontology = Ontology(graph=graph, iri="https://example.org/onto")
    owners = owner_index([ontology])

    assert owner_of("https://example.org/onto#Thing", owners) == ontology.iri


def test_owner_of_falls_back_to_subject_membership() -> None:
    """A term outside the header namespace attributes via graph membership.

    Without the fallback, every term of a versioned-header vocabulary resolves
    to None and silently disappears from per-ontology attribution.
    """
    ontology = _versioned_header_ontology()
    owners = owner_index([ontology])
    term = "http://example.org/vocab/unit/MilliEV"

    # Containment alone cannot attribute the term…
    assert owner_of(term, owners) is None
    # …but subject membership can.
    assert owner_of(term, owners, [ontology]) == ontology.iri


def test_owner_of_unknown_iri_stays_unowned() -> None:
    ontology = _versioned_header_ontology()
    owners = owner_index([ontology])

    assert owner_of("http://elsewhere.org/Term", owners, [ontology]) is None
