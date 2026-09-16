"""Content-hash stability across a triple-store round trip.

Regression guard for catalog identity drift: the hash written at load time must
equal the hash recomputed after the graph has been through the store, or the
content-addressed ``versioned_iri`` identity the catalog is built on breaks and
``select_relevant_ontologies`` falls back a rung on every retrieval.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import XSD

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.triple_manager.in_memory import (
    InMemoryTripleStoreManager,
    _list_named_graph_uris,
)

pytestmark = pytest.mark.unit

SUBJECT = URIRef("urn:test:subject")
PREDICATE = URIRef("urn:test:predicate")

# An ontology exercising both measured store rewrites: xsd:decimal lexical
# canonicalization, and the integer-subtype collapse that OWL qualified
# cardinality axioms trigger. Also carries a long xsd:double lexical, which
# rdflib's stock Turtle writer would round.
DRIFT_PRONE_TURTLE = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <https://example.org/drift#> .

<https://example.org/drift> a owl:Ontology ;
    owl:versionInfo "1.0.0" ;
    rdfs:label "Drift Probe" .

ex:conversionFactor a owl:DatatypeProperty .
ex:planckish a owl:DatatypeProperty .

ex:Unit a owl:Class ;
    ex:conversionFactor "10.0"^^xsd:decimal ;
    ex:planckish "1.602176634e-22"^^xsd:double .

ex:Measurement a owl:Class ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty ex:hasUnit ;
        owl:onClass ex:Unit ;
        owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger
    ] .
"""


def _literal_graph(lexical: str, datatype: URIRef) -> RDFGraph:
    graph = RDFGraph()
    graph.add((SUBJECT, PREDICATE, Literal(lexical, datatype=datatype)))
    return graph


def _hash_of(lexical: str, datatype: URIRef) -> str:
    return _literal_graph(lexical, datatype).hash()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("left", "right", "datatype"),
    [
        ("10.0", "10", XSD.decimal),
        ("0.100", "0.1", XSD.decimal),
        ("+5", "5", XSD.decimal),
        ("1", "1", XSD.nonNegativeInteger),
        ("07", "7", XSD.int),
        ("true", "1", XSD.boolean),
    ],
)
def test_hash_ignores_lexical_variance(left: str, right: str, datatype: URIRef) -> None:
    """Equal values hash equally regardless of how they were written."""
    assert _hash_of(left, datatype) == _hash_of(right, datatype)


@pytest.mark.unit
def test_hash_ignores_integer_subtype() -> None:
    """The store collapses integer subtypes, so the hash must too."""
    assert _hash_of("1", XSD.nonNegativeInteger) == _hash_of("1", XSD.integer)
    assert _hash_of("1", XSD.positiveInteger) == _hash_of("1", XSD.integer)


@pytest.mark.unit
def test_hash_still_separates_distinct_values() -> None:
    """Canonicalization must not collapse genuinely different content."""
    assert _hash_of("10.1", XSD.decimal) != _hash_of("10.2", XSD.decimal)
    assert _hash_of("1", XSD.integer) != _hash_of("2", XSD.integer)
    assert _hash_of("1.602176634e-22", XSD.double) != _hash_of(
        "1.602176635e-22", XSD.double
    )


@pytest.mark.unit
def test_hash_does_not_mutate_the_graph() -> None:
    """Hashing is read-only: what is stored and shown to the LLM is untouched."""
    graph = _literal_graph("10.0", XSD.decimal)
    graph.hash()
    stored = next(iter(graph.objects(SUBJECT, PREDICATE)))
    assert isinstance(stored, Literal)
    assert str(stored) == "10.0"
    assert stored.datatype == XSD.decimal


@pytest.mark.unit
def test_canonical_turtle_preserves_double_precision() -> None:
    """rdflib's plain-literal shorthand rounds doubles to 7 significant digits."""
    graph = _literal_graph("1.602176634e-22", XSD.double)
    turtle = graph.serialize_canonical_turtle()
    assert "1.602176634e-22" in turtle

    reparsed = RDFGraph()
    reparsed.parse(data=turtle, format="turtle")
    value = next(iter(reparsed.objects(SUBJECT, PREDICATE)))
    assert str(value) == "1.602176634e-22"
    assert graph.hash() == reparsed.hash()


def _round_trip(ontology: Ontology) -> tuple[Ontology, int]:
    """Persist ``ontology``, read it back, write it again; report graph count."""

    async def main() -> tuple[Ontology, int]:
        manager = InMemoryTripleStoreManager()
        await manager.async_init()
        manager.serialize(ontology)
        fetched = await manager.afetch_ontologies_by_iri([ontology.iri])
        assert fetched, f"{ontology.iri} did not come back from the store"
        # Re-serializing is what a restart does; a drifting hash creates a
        # second named graph for the same ontology.
        manager.serialize(fetched[0])
        graphs = _list_named_graph_uris(manager._active_partition().ontologies)
        return fetched[0], len(graphs)

    return asyncio.run(main())


@pytest.mark.unit
def test_drift_prone_ontology_survives_store_round_trip() -> None:
    ontology = Ontology(graph=RDFGraph._from_turtle_str(DRIFT_PRONE_TURTLE))
    assert ontology.hash

    fetched, graph_count = _round_trip(ontology)

    assert fetched.hash == ontology.hash
    assert graph_count == 1, "hash drift created a duplicate named graph"


def _shipped_ontologies() -> list[Path]:
    """The TTL fixtures under `test/data/ontologies`, resolved from this file.

    A cwd-relative glob here silently produced *zero* parameters whenever pytest
    ran from anywhere but the repo root, so the round-trip guard vanished without
    a single failure. Resolve from `__file__` and let the emptiness assertion
    below catch a genuine disappearance.
    """
    return sorted(
        (Path(__file__).resolve().parent / "data" / "ontologies").glob("*.ttl")
    )


SHIPPED_ONTOLOGIES = _shipped_ontologies()


@pytest.mark.unit
def test_shipped_ontologies_are_discoverable() -> None:
    assert SHIPPED_ONTOLOGIES, (
        "test/data/ontologies is empty -- the round trip below is vacuous"
    )


@pytest.mark.unit
@pytest.mark.parametrize("ttl_path", SHIPPED_ONTOLOGIES, ids=lambda p: p.stem)
def test_shipped_ontologies_survive_store_round_trip(ttl_path: Path) -> None:
    ontology = Ontology.from_file(ttl_path)
    assert ontology.hash

    fetched, graph_count = _round_trip(ontology)

    assert fetched.hash == ontology.hash
    assert graph_count == 1


@pytest.mark.unit
def test_stale_hash_literal_is_refreshed() -> None:
    """``<iri>#<hash>`` and the advertised ``hash:`` literal cannot disagree."""
    from rdflib.namespace import DCTERMS

    ontology = Ontology(graph=RDFGraph._from_turtle_str(DRIFT_PRONE_TURTLE))
    onto_iri = URIRef(ontology.iri)
    ontology.graph.remove((onto_iri, DCTERMS.identifier, None))
    ontology.graph.add((onto_iri, DCTERMS.identifier, Literal("hash:stale")))

    ontology.sync_properties_to_graph()

    identifiers = {
        str(obj)
        for obj in ontology.graph.objects(onto_iri, DCTERMS.identifier)
        if str(obj).startswith("hash:")
    }
    assert identifiers == {f"hash:{ontology.hash}"}
