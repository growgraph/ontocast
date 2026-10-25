"""Stamped ontologies for tests: the version contract in one call."""

from __future__ import annotations

from datetime import datetime, timezone

from rdflib import OWL, RDF, URIRef

from ontocast.onto.identity import OntologyIdentity, mint_ontology_iri
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_header import (
    adopt_ontology,
    mint_ontology,
    stamp_version,
)
from ontocast.onto.rdfgraph import RDFGraph

DOMAIN = "https://example.org"


def stamped_ontology(
    source: str | RDFGraph,
    *,
    ontology_id: str | None = None,
    iri: str | None = None,
    title: str | None = None,
    description: str = "",
    now: datetime | None = None,
) -> Ontology:
    """A stamped root version of the ontology in ``source``.

    A graph that declares an ``owl:Ontology`` is adopted (its IRI and header
    win; ``ontology_id`` overrides the id). A headerless graph is minted under
    ``iri`` (default ``https://example.org/<ontology_id>``), which needs
    ``ontology_id``.
    """
    graph = (
        RDFGraph._from_turtle_str(source) if isinstance(source, str) else source.copy()
    )
    declared = any(
        isinstance(s, URIRef)
        for s, _, _ in graph.triples((None, RDF.type, OWL.Ontology))
    )
    if declared:
        return adopt_ontology(graph, ontology_id=ontology_id, now=now)
    if ontology_id is None:
        raise ValueError("a headerless graph needs an ontology_id to mint under")
    identity = OntologyIdentity(
        ontology_id=ontology_id,
        iri=iri or mint_ontology_iri(DOMAIN, ontology_id),
        title=title or ontology_id,
        description=description,
    )
    stamped = stamp_version(mint_ontology(identity, graph), now=now)
    assert stamped is not None
    return stamped


def next_version(
    base: Ontology, graph: RDFGraph, *, now: datetime | None = None
) -> Ontology:
    """The stamped child of ``base`` over ``graph`` (its content must differ)."""
    stamped = stamp_version(
        base.derive_updated_version(graph), now=now or datetime.now(timezone.utc)
    )
    assert stamped is not None, "the graph must differ from the base's"
    return stamped
