"""Ontology identity is minted by code: the id rule, the IRI rule, the pair."""

import pytest
from pydantic import TypeAdapter, ValidationError
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.identity import (
    OntologyId,
    OntologyIdentity,
    mint_ontology_iri,
    validate_ontology_iri,
)
from ontocast.onto.ontology_header import mint_ontology
from ontocast.onto.rdfgraph import RDFGraph

pytestmark = pytest.mark.unit

_ID = TypeAdapter(OntologyId)


def test_mint_ontology_iri_is_the_id_under_the_domain() -> None:
    assert mint_ontology_iri("https://example.org", "tmi") == "https://example.org/tmi"
    assert (
        mint_ontology_iri("https://example.org/ontologies", "tmi")
        == "https://example.org/ontologies/tmi"
    )


@pytest.mark.parametrize(
    "domain", ["https://example.org/", "https://example.org#x", "example.org"]
)
def test_mint_ontology_iri_refuses_a_malformed_domain(domain: str) -> None:
    with pytest.raises(ValueError):
        mint_ontology_iri(domain, "tmi")


@pytest.mark.parametrize("value", ["Tmi1!", "t", "doc", "cd", "rdf"])
def test_ontology_id_refuses_malformed_and_reserved_ids(value: str) -> None:
    with pytest.raises(ValidationError):
        _ID.validate_python(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("tmi", "tmi"), ("my-onto_2", "my-onto_2"), (" TMI ", "tmi")],
)
def test_ontology_id_accepts_valid_ids_folded(value: str, expected: str) -> None:
    assert _ID.validate_python(value) == expected


def test_validate_ontology_iri_refuses_a_fragment() -> None:
    with pytest.raises(ValueError, match="fragment"):
        validate_ontology_iri("https://example.org/tmi#")


def test_validate_ontology_iri_refuses_a_query() -> None:
    with pytest.raises(ValueError, match="query"):
        validate_ontology_iri("https://example.org/tmi?v=1")


def test_validate_ontology_iri_accepts_a_trailing_slash() -> None:
    assert (
        validate_ontology_iri("https://example.org/tmi/") == "https://example.org/tmi/"
    )


def test_validate_ontology_iri_refuses_a_relative_iri() -> None:
    with pytest.raises(ValueError, match="absolute"):
        validate_ontology_iri("tmi")


def test_mint_ontology_drops_a_model_written_header() -> None:
    identity = OntologyIdentity(
        ontology_id="tmi", iri="https://example.org/tmi", title="TMI"
    )
    graph = RDFGraph()
    graph.bind("tmi", identity.namespace)
    model_header = URIRef(f"{identity.namespace}Ontology")
    graph.add((model_header, RDF.type, OWL.Ontology))
    graph.add((model_header, RDFS.label, Literal("tmi ontology")))
    graph.add((model_header, OWL.versionInfo, Literal("0.0.1")))
    term = URIRef(f"{identity.namespace}Thing")
    graph.add((term, RDF.type, OWL.Class))
    header_triples = len(list(graph.triples((model_header, None, None))))
    assert header_triples == 3

    artifact = mint_ontology(identity, graph)

    assert len(artifact.graph) == len(graph) - header_triples
    assert list(artifact.graph.triples((None, RDF.type, OWL.Ontology))) == []
    assert (term, RDF.type, OWL.Class) in artifact.graph
    assert artifact.version_iri is None and artifact.bases == []
    assert artifact.ontology_id == "tmi" and artifact.iri == identity.iri
    assert len(graph) == header_triples + 1, "the input graph is untouched"


def test_identity_namespace_is_the_iri_plus_hash() -> None:
    identity = OntologyIdentity(
        ontology_id="tmi", iri="https://example.org/tmi", title="TMI"
    )
    assert identity.namespace == "https://example.org/tmi#"


def test_identity_namespace_is_the_iri_itself_when_it_ends_with_a_slash() -> None:
    identity = OntologyIdentity(
        ontology_id="tmi", iri="https://example.org/tmi/", title="TMI"
    )
    assert identity.namespace == "https://example.org/tmi/"


def test_identity_refuses_a_reserved_id_and_a_fragment_iri() -> None:
    with pytest.raises(ValidationError):
        OntologyIdentity(ontology_id="owl", iri="https://example.org/owl", title="x")
    with pytest.raises(ValidationError):
        OntologyIdentity(ontology_id="tmi", iri="https://example.org/tmi#", title="x")
