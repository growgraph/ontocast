"""The ``# ONTOLOGY`` chapter as a term sheet rather than a serialized graph.

In a facts prompt the ontology is read-only context: the model reads it to pick
terms, and emits its graph in the wire format, never a patch against the chapter
it read. That asymmetry is what makes a non-RDF chapter admissible here and
inadmissible in the ontology loop, whose output *is* a patch against these very
statements.

Freed from being a graph, the chapter can drop everything RDF spends bytes on
that no extractor reads: a JSON node wrapper or a Turtle subject block per term,
a repeated predicate IRI per statement, and prose written for a human browsing
the ontology. What is left is what lets a model use a term -- its name, the
surface forms a document might spell it with, what it is, where it sits in the
hierarchy, what it connects, and the contract saying when it applies.

The completion pass established the format (:mod:`ontocast.prompt.complete_facts`)
on a narrower vocabulary; this renders the whole snapshot in it.
"""

from __future__ import annotations

from rdflib import OWL, RDF, RDFS, SKOS, Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph

#: Rendered in this order, each under its own heading. Individuals come last:
#: they are the longest section on a unit-bearing catalog and the least
#: structural, so a truncated read still reaches classes and properties.
_CLASS_TYPES: frozenset[URIRef] = frozenset({OWL.Class, RDFS.Class})
_PROPERTY_TYPES: frozenset[URIRef] = frozenset(
    {
        RDF.Property,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        OWL.FunctionalProperty,
        OWL.InverseFunctionalProperty,
        OWL.SymmetricProperty,
        OWL.TransitiveProperty,
    }
)

#: Types that say only "this is a term" and are noise on an individual's line.
_UNINFORMATIVE_TYPES: frozenset[URIRef] = frozenset(
    {OWL.NamedIndividual, RDFS.Resource, OWL.Thing}
)


def qname_for(graph: RDFGraph, term: URIRef) -> str:
    """Prefixed name for ``term``, falling back to the full IRI.

    Args:
        graph: Graph whose namespace manager supplies the prefixes.
        term: The IRI to shorten.

    Returns:
        A prefixed name such as ``ex:PowderSample``, or the IRI itself
        when no bound prefix covers it.
    """
    try:
        return graph.namespace_manager.qname(term)
    except Exception:
        return str(term)


def _texts(graph: RDFGraph, subject: URIRef, predicate: URIRef) -> list[str]:
    """Literal objects of ``predicate`` on ``subject``, whitespace-normalised."""
    return [
        " ".join(str(obj).split())
        for obj in graph.objects(subject, predicate)
        if isinstance(obj, Literal)
    ]


def _first_text(graph: RDFGraph, subject: URIRef, *predicates: URIRef) -> str:
    """First literal found across ``predicates``, in the order given."""
    for predicate in predicates:
        for text in _texts(graph, subject, predicate):
            return text
    return ""


def _iris(graph: RDFGraph, subject: URIRef, predicate: URIRef) -> list[str]:
    """Sorted qnames of the IRI objects of ``predicate`` on ``subject``.

    Blank-node objects are skipped: an ``owl:Restriction`` has no name to print,
    and the term sheet has no syntax for an anonymous class expression. The
    class it restricts is already named by the term's own line.
    """
    return sorted(
        qname_for(graph, obj)
        for obj in graph.objects(subject, predicate)
        if isinstance(obj, URIRef)
    )


def _partition(graph: RDFGraph) -> tuple[list[URIRef], list[URIRef], list[URIRef]]:
    """Split named subjects into classes, properties and individuals.

    Classification is by declared ``rdf:type``, with two structural fallbacks so
    a catalog that under-declares still lands its terms in the right section: a
    subject carrying ``rdfs:domain``/``rdfs:range``/``rdfs:subPropertyOf`` is a
    property, and one carrying ``rdfs:subClassOf`` is a class.
    """
    classes: list[URIRef] = []
    properties: list[URIRef] = []
    individuals: list[URIRef] = []
    for subject in sorted(
        {s for s in graph.subjects() if isinstance(s, URIRef)}, key=str
    ):
        types = {t for t in graph.objects(subject, RDF.type) if isinstance(t, URIRef)}
        if types & _PROPERTY_TYPES or any(
            (subject, p, None) in graph
            for p in (RDFS.domain, RDFS.range, RDFS.subPropertyOf)
        ):
            properties.append(subject)
        elif types & _CLASS_TYPES or (subject, RDFS.subClassOf, None) in graph:
            classes.append(subject)
        else:
            individuals.append(subject)
    return classes, properties, individuals


def _term_line(graph: RDFGraph, subject: URIRef, *, kind: str) -> list[str]:
    """The one-or-two lines describing ``subject``.

    Args:
        graph: The snapshot being rendered.
        subject: Term to describe.
        kind: ``"class"``, ``"property"`` or ``"individual"`` -- selects which
            relations are worth printing for this term.

    Returns:
        The term's line, plus an indented ``note:`` line when it carries a
        usage contract.
    """
    label = _first_text(graph, subject, RDFS.label, SKOS.prefLabel)
    parts = [f"  {qname_for(graph, subject)}"]
    if label:
        parts.append(f'"{label}"')

    if kind == "class":
        parents = _iris(graph, subject, RDFS.subClassOf) + _iris(
            graph, subject, OWL.equivalentClass
        )
        if parents:
            parts.append(f"< {', '.join(sorted(set(parents)))}")
    elif kind == "property":
        domain = _iris(graph, subject, RDFS.domain)
        codomain = _iris(graph, subject, RDFS.range)
        parts.append(f"{'/'.join(domain) or '?'} -> {'/'.join(codomain) or '?'}")
        super_properties = _iris(graph, subject, RDFS.subPropertyOf)
        if super_properties:
            parts.append(f"< {', '.join(super_properties)}")
    else:
        types = [
            qname_for(graph, obj)
            for obj in graph.objects(subject, RDF.type)
            if isinstance(obj, URIRef) and obj not in _UNINFORMATIVE_TYPES
        ]
        if types:
            parts.append(f": {', '.join(sorted(types))}")

    # Every surface form, shortest first. There is deliberately no count cap
    # here: alternative labels are the cheapest content in the sheet (a couple
    # of dozen characters each) and the most direct thing a document match has
    # to work with -- an alphabetical top-N on qqval:Approximate cuts exactly
    # the "~", "\u223c" and "\u2248" a paper actually prints. TextCaps.total_budget
    # is the one mechanism that bounds them, and only on a catalog that needs it.
    alt = sorted(set(_texts(graph, subject, SKOS.altLabel)), key=lambda a: (len(a), a))
    if alt:
        parts.append(f"~ {'; '.join(alt)}")

    lines = ["  ".join(parts)]
    note = _first_text(graph, subject, SKOS.scopeNote, SKOS.definition)
    if note:
        lines.append(f"      note: {note}")
    return lines


_HEADER = """Every term available to you, one per line:
  `qname "label"` names it; `< ...` is its parent; `A -> B` is a property's
  domain and range; `: T` is an individual's type; `~ ...` are alternative
  surface forms the source text may use; `note:` is a usage contract.
Use these terms and only these. A term absent from this sheet does not exist."""


def build_ontology_term_sheet(graph: RDFGraph) -> str:
    """Render ``graph`` as the body of a term-sheet ontology chapter.

    Terms are emitted in sorted IRI order within each section, so the same
    snapshot renders byte-identically on every call and across processes -- which
    is what lets a provider's prefix cache serve the chapter more than once.
    Unlike the RDF chapters, this never depends on blank-node identifiers, whose
    labels rdflib mints at random.

    Args:
        graph: The unit's ontology snapshot.

    Returns:
        The chapter body, or ``""`` when the snapshot names no term.
    """
    classes, properties, individuals = _partition(graph)
    sections = [
        ("Classes", classes, "class"),
        ("Properties  (domain -> range)", properties, "property"),
        ("Individuals  (units, vocabulary values)", individuals, "individual"),
    ]
    if not any(terms for _, terms, _ in sections):
        return ""

    lines = [_HEADER]
    for heading, terms, kind in sections:
        if not terms:
            continue
        lines.append(f"\n## {heading}")
        for term in terms:
            lines.extend(_term_line(graph, term, kind=kind))
    return "\n".join(lines)
