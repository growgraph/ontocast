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
- :func:`drop_non_term_subjects`: an ontology update declares terms and value
  sets. A subject in the facts namespace, an individual (typed, but neither as
  a term nor as a SKOS concept), or an untyped subject that nothing declares
  belongs in the facts graph, not in an ontology.
- :func:`drop_terms_minted_outside_home`: every new term of a document lives
  under its home namespace; a term newly declared under another namespace of
  the deployment is a second copy of one the document already named, and one
  declared in another vocabulary is that vocabulary's to write.

Every rule removes a blank-node structure (a restriction, a list) whole or not at all:
removing one of its statements, or every statement pointing at it, removes it
with the statements pointing at it (:func:`ontocast.onto.blank_structures.remove_with_structures`).
"""

from __future__ import annotations

from collections.abc import Iterable

from rdflib import OWL, RDF, RDFS, SKOS, XSD, Graph, URIRef

from ontocast.onto.blank_structures import remove_with_structures
from ontocast.onto.identity import is_owned_ontology
from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.well_known import well_known_vocabulary

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

#: ``rdf:type`` objects that make a subject something an ontology may declare.
TERM_TYPES: frozenset[URIRef] = frozenset(
    {
        OWL.Class,
        RDFS.Class,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        RDF.Property,
        RDFS.Datatype,
        OWL.Ontology,
        SKOS.Concept,
        SKOS.ConceptScheme,
    }
)

#: Predicates only a class or a property carries: an untyped subject with one
#: of them is a term whose declaration the model left implicit.
TERM_AXIOMS: frozenset[URIRef] = frozenset(
    {
        RDFS.subClassOf,
        RDFS.subPropertyOf,
        RDFS.domain,
        RDFS.range,
        OWL.equivalentClass,
        OWL.equivalentProperty,
        OWL.inverseOf,
        OWL.disjointWith,
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

    return remove_with_structures(graph, [t for t in graph if external(t[0])])


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

    return remove_with_structures(
        graph, [triple for triple in graph if any(stray(node) for node in triple)]
    )


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


def _drop_subjects(graph: Graph, subjects: Iterable[URIRef]) -> int:
    return remove_with_structures(
        graph, [triple for s in subjects for triple in graph.triples((s, None, None))]
    )


def drop_non_term_subjects(
    graph: Graph, *, declared: Iterable[str], facts_namespace: str
) -> int:
    """Remove the triples whose subject is not a term or a value set.

    A named subject stays when it is typed as one of :data:`TERM_TYPES`, or is
    untyped and either in ``declared`` (catalog terms and ontology IRIs) or
    carries one of :data:`TERM_AXIOMS`. A subject under ``facts_namespace``
    never stays, whatever its type.
    """
    known = set(declared)
    doomed: set[URIRef] = set()
    for s in set(graph.subjects()):
        if not isinstance(s, URIRef):
            continue
        if str(s).startswith(facts_namespace):
            doomed.add(s)
            continue
        types = set(graph.objects(s, RDF.type))
        if types & TERM_TYPES:
            continue
        if types:
            doomed.add(s)
        elif str(s) not in known and not any(
            (s, p, None) in graph for p in TERM_AXIOMS
        ):
            doomed.add(s)
    return _drop_subjects(graph, doomed)


def drop_terms_minted_outside_home(
    graph: Graph,
    *,
    home_namespace: str,
    catalog_terms: Iterable[str],
    current_domain: str,
) -> int:
    """Remove the statements about a term newly declared outside ``home_namespace``.

    A term is newly declared when the catalog does not hold it and the graph
    either types it as one of :data:`TERM_TYPES` (ontology headers aside) or,
    untyped, gives it one of :data:`TERM_AXIOMS`. A term of a well-known
    vocabulary is never newly declared: an axiom about one is a reuse, and
    the partition decides where it goes.

    Under ``current_domain`` such a term is a second copy of one the document
    already named, so every triple naming it goes, and no axiom points at a
    term that is not written. Outside it the term is another vocabulary's: the
    statements about it go and a reference to it stays, as it would had the
    render not declared it.
    """
    catalog = set(catalog_terms)

    def outside(s: object) -> bool:
        return (
            isinstance(s, URIRef)
            and not str(s).startswith(home_namespace)
            and str(s) not in catalog
        )

    typed = {
        s
        for s, o in graph.subject_objects(RDF.type)
        if o in TERM_TYPES and o != OWL.Ontology and outside(s)
    }
    untyped = {
        s
        for p in TERM_AXIOMS
        for s in graph.subjects(p, None)
        if outside(s)
        and not (set(graph.objects(s, RDF.type)) & TERM_TYPES)
        and well_known_vocabulary(str(s)) is None
    }
    strays = typed | untyped
    owned = {s for s in strays if is_owned_ontology(str(s), current_domain)}
    return remove_with_structures(
        graph, [t for t in graph if t[0] in strays or t[2] in owned]
    )


__all__ = [
    "DATATYPE_PRECEDENCE",
    "EXAMPLE_NAMESPACE_STEMS",
    "TERM_AXIOMS",
    "TERM_TYPES",
    "THING_IRIS",
    "drop_example_namespace_triples",
    "drop_external_subjects",
    "drop_non_term_subjects",
    "drop_terms_minted_outside_home",
    "drop_vacuous_upper_terms",
    "normalize_datatype_ranges",
]
