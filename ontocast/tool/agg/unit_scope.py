"""Unit-scoped fact IRIs: cross-unit coreference as a merge decision.

Units render facts independently, and each mints instance IRIs from the
text in front of it, so two units that both write ``cd:temperature_value``
produce one IRI for what may be two different measurements. Aggregation
keys its entity collection by IRI, which fuses the two nodes before any
merge guard sees them, and the validation gate's un-merge repair only
dissolves clusters of two or more *source* IRIs -- a name collision is a
singleton there and can never be split.

Scoping rewrites every minted fact IRI of a unit graph to carry the unit
index as a suffix on its local name::

    <ns>temperature_value  ->  <ns>temperature_value__u3

The suffix stays inside the local name (no new path segment), so namespace
splitting keeps working and the entity remains a fact under its namespace.
Two units naming the same thing now arrive as two source entities that earn
their merge like any other alias pair: clustered by the embedding model and
validated by the symbolic guards, where disjoint literal values or
conflicting functional objects keep them apart. The suffix is stripped
wherever a local name is read -- normal forms, structured-id fallbacks,
representative selection, final minting -- so a served graph never carries
it.

Only *instances* are scoped. An IRI the unit itself uses as a predicate, as
a type, or as a schema-relation target is vocabulary the unit refers to
rather than mints; its cross-unit identity is by name.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rdflib import OWL, RDF, RDFS, Node, URIRef

from ontocast.onto.iri_policy import normalize_namespace_iri, split_namespace_local
from ontocast.onto.rdfgraph import RDFGraph

#: Separator between a minted local name and the index of the unit that minted it.
UNIT_SCOPE_MARKER = "__u"

_UNIT_SCOPE_RE = re.compile(rf"{re.escape(UNIT_SCOPE_MARKER)}(-?\d+)$")

#: Types whose subjects are vocabulary terms and are never scoped.
_SCHEMA_ROLE_TYPES: frozenset[URIRef] = frozenset(
    {
        RDFS.Class,
        OWL.Class,
        RDF.Property,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
    }
)

#: Predicates whose objects are vocabulary terms and are never scoped.
_SCHEMA_TARGET_PREDICATES: frozenset[URIRef] = frozenset(
    {
        RDF.type,
        RDFS.subClassOf,
        RDFS.subPropertyOf,
        RDFS.domain,
        RDFS.range,
        OWL.equivalentClass,
        OWL.equivalentProperty,
        OWL.inverseOf,
    }
)


def strip_unit_scope(name: str) -> str:
    """Return *name* without a trailing unit-scope suffix.

    Accepts a bare local name or a full IRI string: the suffix is terminal
    either way, and a name that carries none is returned unchanged.

    Args:
        name: Local name or IRI string.

    Returns:
        The name with any ``__u<index>`` suffix removed.
    """
    return _UNIT_SCOPE_RE.sub("", name)


def scope_local_name(local_name: str, unit_index: int) -> str:
    """Return *local_name* carrying the scope of *unit_index*.

    An existing scope suffix is replaced rather than stacked, so scoping a
    unit graph twice is a no-op -- the validation gate re-aggregates units
    that were already scoped by the first aggregation pass.

    Args:
        local_name: Local name as minted by the unit.
        unit_index: Position of the minting unit in the document.

    Returns:
        ``<local_name>__u<unit_index>``.
    """
    return f"{strip_unit_scope(local_name)}{UNIT_SCOPE_MARKER}{unit_index}"


def unit_scope_index(name: str) -> int | None:
    """Return the unit index carried by *name*, or None when it is unscoped.

    Args:
        name: Local name or IRI string.

    Returns:
        The index encoded in the trailing scope suffix, if any.
    """
    match = _UNIT_SCOPE_RE.search(name)
    return int(match.group(1)) if match else None


def unscoped_iri(iri: URIRef) -> URIRef:
    """Return *iri* with any unit-scope suffix removed from its local name.

    Args:
        iri: Possibly scoped IRI.

    Returns:
        The same object when unscoped, otherwise a new IRI without the suffix.
    """
    text = str(iri)
    stripped = strip_unit_scope(text)
    return iri if stripped == text else URIRef(stripped)


def _vocabulary_terms(graph: RDFGraph) -> set[URIRef]:
    """IRIs the graph uses as vocabulary: predicates, types, schema targets."""
    terms: set[URIRef] = set()
    for subject, predicate, obj in graph:
        if isinstance(predicate, URIRef):
            terms.add(predicate)
        if predicate in _SCHEMA_TARGET_PREDICATES and isinstance(obj, URIRef):
            terms.add(obj)
        if (
            predicate == RDF.type
            and obj in _SCHEMA_ROLE_TYPES
            and isinstance(subject, URIRef)
        ):
            terms.add(subject)
    return terms


def scope_fact_iris(
    graph: RDFGraph,
    unit_index: int,
    fact_namespaces: Sequence[str | URIRef],
) -> dict[URIRef, URIRef]:
    """Suffix every minted fact IRI in *graph* with the unit index, in place.

    A term is rewritten when it sits in subject or object position, lives
    under one of *fact_namespaces*, and is not vocabulary of the graph itself
    (see :func:`_vocabulary_terms`). Every occurrence of a rewritten IRI is
    replaced, so the graph stays internally consistent; predicates and
    literals are untouched. Idempotent for the same unit index.

    Args:
        graph: Unit facts graph, modified in place.
        unit_index: Position of the unit in the document.
        fact_namespaces: Namespaces holding minted instances (the configured
            facts base and the unit's document IRI); empty entries are
            ignored.

    Returns:
        Mapping from each rewritten source IRI to its scoped IRI.
    """
    namespaces = tuple(
        normalize_namespace_iri(str(namespace), context="facts")
        for namespace in fact_namespaces
        if namespace
    )
    if not namespaces or len(graph) == 0:
        return {}

    vocabulary = _vocabulary_terms(graph)
    mapping: dict[URIRef, URIRef] = {}

    def scoped(term: Node) -> Node:
        if not isinstance(term, URIRef) or term in vocabulary:
            return term
        cached = mapping.get(term)
        if cached is not None:
            return cached
        text = str(term)
        if not any(text.startswith(namespace) for namespace in namespaces):
            return term
        namespace, local = split_namespace_local(text)
        if namespace is None or not local:
            return term
        target = URIRef(f"{namespace}{scope_local_name(local, unit_index)}")
        if target == term:
            return term
        mapping[term] = target
        return target

    updates: list[tuple[tuple[Node, Node, Node], tuple[Node, Node, Node]]] = []
    for subject, predicate, obj in graph:
        new_subject = scoped(subject)
        new_obj = scoped(obj)
        if new_subject != subject or new_obj != obj:
            updates.append(
                ((subject, predicate, obj), (new_subject, predicate, new_obj))
            )
    for old, new in updates:
        graph.remove(old)
        graph.add(new)
    return mapping
