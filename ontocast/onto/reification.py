"""RDF 1.2 statements about statements, held beside an rdflib graph.

Provenance is RDF 1.2 reification: ``_:r rdf:reifies <<( s p o )>>`` plus
``_:r prov:wasDerivedFrom <chunk>``. rdflib has no node type for a triple term,
so an :class:`~ontocast.onto.rdfgraph.RDFGraph` keeps every ``rdf:reifies``
statement here, in a pyoxigraph ``Dataset``, which keeps literal lexical forms.
The reifier's other arcs are ordinary triples in the rdflib graph; the two
halves share the reifier by blank-node id or IRI.

:func:`to_ox` and :func:`from_ox` convert terms between rdflib and pyoxigraph.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Iterator, Mapping

import pyoxigraph as ox
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import XSD
from rdflib.term import Node

from ontocast.onto.constants import RDF_REIFIES

Triple = tuple[Node, Node, Node]
OxSubject = ox.NamedNode | ox.BlankNode
OxObject = ox.NamedNode | ox.BlankNode | ox.Literal | ox.Triple

_REIFIES = ox.NamedNode(str(RDF_REIFIES))
_XSD_STRING = str(XSD.string)


class ProvenanceFormatError(ValueError):
    """A graph carrying RDF 1.2 provenance was asked for a format without triple terms."""


def _blank_node(identifier: str) -> ox.BlankNode:
    try:
        return ox.BlankNode(identifier)
    except ValueError:
        # An id pyoxigraph rejects maps to a stable one, so both halves of a
        # graph still agree on the reifier.
        return ox.BlankNode("b" + hashlib.sha1(identifier.encode()).hexdigest()[:24])


def to_ox(term: Node) -> ox.NamedNode | ox.BlankNode | ox.Literal:
    """The pyoxigraph term for an rdflib term, lexical form unchanged.

    Raises:
        TypeError: ``term`` is not an IRI, blank node or literal.
    """
    if isinstance(term, URIRef):
        return ox.NamedNode(str(term))
    if isinstance(term, BNode):
        return _blank_node(str(term))
    if isinstance(term, Literal):
        if term.language:
            return ox.Literal(str(term), language=term.language)
        if term.datatype is not None:
            return ox.Literal(str(term), datatype=ox.NamedNode(str(term.datatype)))
        return ox.Literal(str(term))
    raise TypeError(f"{term!r} (rdflib type {type(term).__name__}) is not an RDF term")


def from_ox(term: object) -> Node:
    """The rdflib term for a pyoxigraph term.

    An ``xsd:string`` literal becomes a plain literal, as rdflib's own parsers
    produce it.

    Raises:
        TypeError: ``term`` is a triple term or not a term at all.
    """
    if isinstance(term, ox.NamedNode):
        return URIRef(term.value)
    if isinstance(term, ox.BlankNode):
        return BNode(term.value)
    if isinstance(term, ox.Literal):
        if term.language:
            return Literal(term.value, lang=term.language)
        if term.datatype.value == _XSD_STRING:
            return Literal(term.value)
        return Literal(term.value, datatype=URIRef(term.datatype.value))
    raise TypeError(f"{term!r} has no rdflib form")


def ox_triple(triple: Triple) -> ox.Triple | None:
    """The triple term for an rdflib triple, or ``None`` when one position
    cannot hold its term (a literal subject, say)."""
    try:
        subject, predicate, obj = (to_ox(position) for position in triple)
    except TypeError:
        return None
    if not isinstance(subject, (ox.NamedNode, ox.BlankNode)):
        return None
    if not isinstance(predicate, ox.NamedNode):
        return None
    return ox.Triple(subject, predicate, obj)


def triple_from_ox(term: ox.Triple) -> Triple:
    """The rdflib triple a triple term states."""
    return from_ox(term.subject), from_ox(term.predicate), from_ox(term.object)


class Reifications:
    """The ``rdf:reifies`` statements of one graph: reifier and reified triple."""

    def __init__(self) -> None:
        self._dataset = ox.Dataset()

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[tuple[Node, Triple]]:
        for quad in self._dataset:
            obj = quad.object
            if isinstance(obj, ox.Triple):
                yield from_ox(quad.subject), triple_from_ox(obj)

    def add(self, reifier: Node, triple: Triple) -> None:
        """Record that ``reifier`` reifies ``triple``.

        Raises:
            TypeError: ``reifier`` is not an IRI or blank node, or ``triple``
                cannot be a triple term.
        """
        subject = to_ox(reifier)
        if not isinstance(subject, (ox.NamedNode, ox.BlankNode)):
            raise TypeError(f"reifier {reifier!r} must be an IRI or a blank node")
        term = ox_triple(triple)
        if term is None:
            raise TypeError(f"{triple!r} cannot be a triple term")
        self.add_ox(subject, term)

    def add_ox(self, reifier: OxSubject, term: ox.Triple) -> None:
        """Record a reification given as pyoxigraph terms."""
        self._dataset.add(ox.Quad(reifier, _REIFIES, term, ox.DefaultGraph()))

    def reifiers(self) -> set[Node]:
        """Every reifier."""
        return {from_ox(quad.subject) for quad in self._dataset}

    def reifiers_of(self, triple: Triple) -> set[Node]:
        """The reifiers of ``triple``."""
        term = ox_triple(triple)
        if term is None:
            return set()
        return {from_ox(quad.subject) for quad in self._dataset.quads_for_object(term)}

    def quads(self) -> list[ox.Quad]:
        """The statements as quads in the default graph."""
        return list(self._dataset)

    def copy(self) -> Reifications:
        """An independent copy."""
        copied = Reifications()
        copied.update(self)
        return copied

    def update(self, other: Reifications) -> None:
        """Add every statement of ``other``."""
        for quad in other._dataset:
            self._dataset.add(quad)

    def clear(self) -> None:
        """Remove every statement."""
        self._dataset.clear()

    def remove_reifiers(self, reifiers: Collection[Node]) -> int:
        """Remove the statements of ``reifiers``; returns how many."""
        removed = 0
        for reifier in reifiers:
            subject = to_ox(reifier)
            if not isinstance(subject, (ox.NamedNode, ox.BlankNode)):
                continue
            for quad in list(self._dataset.quads_for_subject(subject)):
                self._dataset.remove(quad)
                removed += 1
        return removed

    def drop_mentioning(self, nodes: Collection[Node]) -> set[Node]:
        """Remove the reifications whose triple mentions any of ``nodes``.

        Returns:
            The reifiers removed, so the caller can drop their other arcs.
        """
        targets = {to_ox(node) for node in nodes if not isinstance(node, tuple)}
        if not targets:
            return set()
        doomed = [
            quad
            for quad in self._dataset
            if isinstance(quad.object, ox.Triple)
            and targets
            & {quad.object.subject, quad.object.predicate, quad.object.object}
        ]
        for quad in doomed:
            self._dataset.remove(quad)
        return {from_ox(quad.subject) for quad in doomed}

    def retarget(self, replacements: Mapping[Triple, Triple]) -> int:
        """Repoint reifications from a replaced triple onto its replacement.

        The reifier is kept, so its other arcs move with it.

        Returns:
            The number of statements repointed.
        """
        moved = 0
        for removed, replacement in replacements.items():
            old_term, new_term = ox_triple(removed), ox_triple(replacement)
            if old_term is None or new_term is None or old_term == new_term:
                continue
            for quad in list(self._dataset.quads_for_object(old_term)):
                reifier = quad.subject
                if not isinstance(reifier, (ox.NamedNode, ox.BlankNode)):
                    continue
                self._dataset.remove(quad)
                self.add_ox(reifier, new_term)
                moved += 1
        return moved

    def rename(self, rename_term: Callable[[Node], Node]) -> int:
        """Apply ``rename_term`` to every term, as the graph's own rewrite does.

        Returns:
            The number of statements changed.
        """
        changed: list[tuple[ox.Quad, ox.Quad]] = []
        for quad in self._dataset:
            obj = quad.object
            if not isinstance(obj, ox.Triple):
                continue
            reifier = from_ox(quad.subject)
            triple = triple_from_ox(obj)
            new_reifier = rename_term(reifier)
            new_triple = (
                rename_term(triple[0]),
                rename_term(triple[1]),
                rename_term(triple[2]),
            )
            if new_reifier == reifier and new_triple == triple:
                continue
            subject = to_ox(new_reifier)
            term = ox_triple(new_triple)
            if not isinstance(subject, (ox.NamedNode, ox.BlankNode)) or term is None:
                continue
            changed.append((quad, ox.Quad(subject, _REIFIES, term, ox.DefaultGraph())))
        for old, new in changed:
            self._dataset.remove(old)
            self._dataset.add(new)
        return len(changed)


__all__ = [
    "ProvenanceFormatError",
    "Reifications",
    "from_ox",
    "ox_triple",
    "to_ox",
    "triple_from_ox",
]
