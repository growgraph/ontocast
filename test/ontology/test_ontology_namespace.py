"""An adopted ontology's namespace is where its terms live, not ``<iri>#``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from rdflib import OWL, RDF, RDFS, URIRef

from ontocast.onto.constants import VANN
from ontocast.onto.ontology_header import adopt_ontology, read_header
from ontocast.onto.ontology_namespace import ontology_namespace
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager
from ontocast.tool.vector_store.atomizer import GraphAtomizer

pytestmark = pytest.mark.unit

T0 = datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc)

PREFIXES = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix vann: <http://purl.org/vocab/vann/> .
"""

# Terms under "<iri>/", as OM-2 publishes them.
SLASH_TTL = (
    PREFIXES
    + """
<https://example.org/units> a owl:Ontology ; rdfs:label "Units" .
<https://example.org/units/Unit> a owl:Class ; rdfs:label "unit" .
<https://example.org/units/Measure> a owl:Class ; rdfs:label "measure" .
"""
)

# The header pair W3C PROV-O publishes: a bare "prov#" header and a described
# "prov-o#" one, with the terms under "prov#".
PROV_O_TTL = (
    PREFIXES
    + """
@prefix prov: <http://www.w3.org/ns/prov#> .
<http://www.w3.org/ns/prov#> a owl:Ontology .
<http://www.w3.org/ns/prov-o#> a owl:Ontology ;
    rdfs:label "W3C PROVenance Interchange Ontology (PROV-O)"@en ;
    rdfs:seeAlso <http://www.w3.org/TR/prov-o/> ;
    owl:versionInfo "Recommendation version 2013-04-30"@en .
prov:Activity a owl:Class ; rdfs:label "Activity" .
prov:Agent a owl:Class ; rdfs:label "Agent" .
prov:wasAssociatedWith a owl:ObjectProperty ; rdfs:label "wasAssociatedWith" ;
    rdfs:domain prov:Activity ; rdfs:range prov:Agent .
"""
)


def _graph(ttl: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=ttl, format="turtle")
    return graph


def test_slash_namespace_is_kept_and_bound_to_the_prefix() -> None:
    adopted = adopt_ontology(_graph(SLASH_TTL), now=T0)

    assert adopted.iri == "https://example.org/units"
    assert adopted.namespace == "https://example.org/units/"
    assert adopted.prefix == "units"
    subject = URIRef(adopted.iri)
    assert str(adopted.graph.value(subject, VANN.preferredNamespaceUri)) == (
        "https://example.org/units/"
    )


def test_declared_namespace_wins_when_terms_live_there() -> None:
    ttl = SLASH_TTL.replace(
        'rdfs:label "Units" .',
        'rdfs:label "Units" ; vann:preferredNamespaceUri "https://example.org/units/" .',
    ).replace(
        "<https://example.org/units/Measure>", "<https://other.example/x#Measure>"
    )
    graph = _graph(ttl)
    graph.add((URIRef("https://other.example/x#Kind"), RDF.type, OWL.Class))

    assert ontology_namespace(graph, "https://example.org/units") == (
        "https://example.org/units/"
    )


def test_declared_namespace_without_terms_gives_way_to_the_terms() -> None:
    graph = _graph(
        SLASH_TTL.replace(
            'rdfs:label "Units" .',
            'rdfs:label "Units" ; vann:preferredNamespaceUri "https://example.org/units#" .',
        )
    )

    assert ontology_namespace(graph, "https://example.org/units") == (
        "https://example.org/units/"
    )


def test_w3c_prov_o_header_pair_adopts() -> None:
    adopted = adopt_ontology(_graph(PROV_O_TTL), now=T0)

    assert adopted.iri == "http://www.w3.org/ns/prov-o"
    assert adopted.namespace == "http://www.w3.org/ns/prov#"
    headers = set(adopted.graph.subjects(RDF.type, OWL.Ontology))
    assert headers == {URIRef("http://www.w3.org/ns/prov-o")}
    # The author's extras move with the header to its canonical IRI.
    assert (
        URIRef("http://www.w3.org/ns/prov-o"),
        RDFS.seeAlso,
        URIRef("http://www.w3.org/TR/prov-o/"),
    ) in adopted.graph


def test_two_described_headers_are_refused() -> None:
    ttl = PROV_O_TTL.replace(
        "<http://www.w3.org/ns/prov#> a owl:Ontology .",
        '<http://www.w3.org/ns/prov#> a owl:Ontology ; rdfs:label "PROV" .',
    )
    with pytest.raises(ValueError, match="exactly one owl:Ontology"):
        adopt_ontology(_graph(ttl), now=T0)


@pytest.mark.anyio
async def test_namespace_survives_the_store() -> None:
    adopted = adopt_ontology(_graph(SLASH_TTL), now=T0)
    assert adopted.version_iri is not None
    manager = InMemoryTripleStoreManager()
    await manager.async_init()
    manager.serialize(adopted)

    header = (await manager.afetch_ontology_catalog())[0]
    fetched = (await manager.afetch_ontologies_by_iri([adopted.iri]))[0]

    assert header.namespace == "https://example.org/units/"
    assert fetched.namespace == "https://example.org/units/"
    assert read_header(adopted.graph, adopted.version_iri).namespace == (
        "https://example.org/units/"
    )


def test_atomizer_indexes_an_adopted_slash_ontology() -> None:
    adopted = adopt_ontology(_graph(SLASH_TTL), now=T0)

    iris = {a.iri for a in GraphAtomizer().atomize(source=adopted)}

    assert {
        "https://example.org/units/Unit",
        "https://example.org/units/Measure",
    } <= iris
