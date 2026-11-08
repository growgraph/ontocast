"""Deterministic clean-up of a document's ontology delta before it is written.

Each rule removes triples a render should not have produced and returns how
many it removed, for ``ontology_reduce_metrics``:

- :func:`drop_external_subjects`: triples about a term of an external ontology.
  An external ontology is never written to; its terms are used by IRI only.
- :func:`drop_example_namespace_triples`: triples naming an IRI under a
  reserved example domain (``example.org``, ``.com``, ``.net``) outside the
  deployment's own domain: the namespace of prompt examples, which a model
  reaches for when it lacks a real one.
- :func:`normalize_datatype_ranges`: a property with several datatype ranges
  means their intersection, which is empty for two different datatypes; one
  range is kept.
- :func:`drop_vacuous_upper_terms`: ``owl:Thing`` / ``schema:Thing`` as a
  range or domain, or as a parent beside another parent, says nothing and
  reads as a modelling decision.
"""

from __future__ import annotations

from collections.abc import Iterable

from rdflib import OWL, RDF, RDFS, XSD, Graph, URIRef

from ontocast.onto.iri_policy import split_namespace_local

#: Reserved by RFC 2606 for documentation; never a real vocabulary.
EXAMPLE_NAMESPACE_STEMS: tuple[str, ...] = tuple(
    f"{scheme}://{host}/"
    for scheme in ("http", "https")
    for host in ("example.org", "example.com", "example.net")
)

#: The class every class is a subclass of, in the vocabularies runs use.
THING_IRIS: frozenset[URIRef] = frozenset(
    {
        OWL.Thing,
        URIRef("http://schema.org/Thing"),
        URIRef("https://schema.org/Thing"),
    }
)

#: Datatype ranges in the order one is kept when a property has several: the
#: language-tagged and plain string forms first, since a model that gives two
#: ranges usually means "text"; then numbers from wide to narrow; then time.
DATATYPE_PRECEDENCE: tuple[URIRef, ...] = (
    RDF.langString,
    XSD.string,
    XSD.decimal,
    XSD.double,
    XSD.float,
    XSD.integer,
    XSD.nonNegativeInteger,
    XSD.positiveInteger,
    XSD.dateTime,
    XSD.date,
    XSD.gYearMonth,
    XSD.gYear,
    XSD.boolean,
    XSD.anyURI,
)


def _is_datatype(node: object) -> bool:
    return isinstance(node, URIRef) and (
        str(node).startswith(str(XSD))
        or node in (RDF.langString, RDF.PlainLiteral, RDFS.Literal)
    )


def drop_external_subjects(graph: Graph, external_stems: Iterable[str]) -> int:
    """Remove the triples whose subject lies in an external namespace.

    ``external_stems`` are namespace stems without a trailing ``/`` or ``#``
    (an ontology IRI, its namespace). A subject is external when its own
    namespace stem is one of them or lies under one (``<stem>/...``).
    """
    stems = tuple(stem.rstrip("/#") for stem in external_stems if stem)
    if not stems:
        return 0

    def external(node: object) -> bool:
        if not isinstance(node, URIRef):
            return False
        namespace, _ = split_namespace_local(str(node))
        stem = (namespace or "").rstrip("/#")
        return bool(stem) and any(
            stem == root or stem.startswith((f"{root}/", f"{root}#")) for root in stems
        )

    doomed = [triple for triple in graph if external(triple[0])]
    for triple in doomed:
        graph.remove(triple)
    return len(doomed)


def drop_example_namespace_triples(graph: Graph, current_domain: str) -> int:
    """Remove the triples that name an IRI under a reserved example domain.

    IRIs under ``current_domain`` are the deployment's own and are kept even
    when it is an example domain (a test or demo deployment).
    """
    own = f"{current_domain.rstrip('/')}/"

    def stray(node: object) -> bool:
        text = str(node)
        return (
            isinstance(node, URIRef)
            and text.startswith(EXAMPLE_NAMESPACE_STEMS)
            and not text.startswith(own)
        )

    doomed = [triple for triple in graph if any(stray(node) for node in triple)]
    for triple in doomed:
        graph.remove(triple)
    return len(doomed)


def normalize_datatype_ranges(graph: Graph) -> int:
    """Keep one datatype range per property; return the properties changed.

    ``rdfs:Literal`` goes when a concrete datatype is present; of several
    concrete ones, the first in :data:`DATATYPE_PRECEDENCE` stays (else the
    lowest IRI).
    """
    changed = 0
    for prop in sorted(set(graph.subjects(RDFS.range, None)), key=str):
        ranges = [r for r in graph.objects(prop, RDFS.range) if _is_datatype(r)]
        if len(ranges) < 2:
            continue
        concrete = [r for r in ranges if r != RDFS.Literal] or ranges
        rank = {iri: i for i, iri in enumerate(DATATYPE_PRECEDENCE)}
        keep = min(concrete, key=lambda r: (rank.get(r, len(rank)), str(r)))
        for r in ranges:
            if r != keep:
                graph.remove((prop, RDFS.range, r))
        changed += 1
    return changed


def drop_vacuous_upper_terms(graph: Graph) -> int:
    """Remove Thing as a range or domain, and as a parent beside another parent."""
    doomed: list[tuple] = []
    for thing in THING_IRIS:
        doomed.extend(graph.triples((None, RDFS.range, thing)))
        doomed.extend(graph.triples((None, RDFS.domain, thing)))
        for cls in graph.subjects(RDFS.subClassOf, thing):
            if any(
                parent not in THING_IRIS
                for parent in graph.objects(cls, RDFS.subClassOf)
            ):
                doomed.append((cls, RDFS.subClassOf, thing))
    for triple in doomed:
        graph.remove(triple)
    return len(doomed)


__all__ = [
    "DATATYPE_PRECEDENCE",
    "EXAMPLE_NAMESPACE_STEMS",
    "THING_IRIS",
    "drop_example_namespace_triples",
    "drop_external_subjects",
    "drop_vacuous_upper_terms",
    "normalize_datatype_ranges",
]
