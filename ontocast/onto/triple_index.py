"""Stable per-triple ids for the graph a critic is asked to review.

A critique is only actionable if the loop can find the triples it names. The
critic used to name them by *requoting* their text into ``incorrect_value``,
which asks the model to reproduce graph content from memory. Measured across a
large corpus of cached critiques that reproduction succeeds a minority of the
time for ``REMOVE`` and barely half the time for ``REPLACE``: the payload comes
back as prose, or as a plausible-but-invented IRI, or -- most often -- as a
node-shaped quote spanning several triples with one predicate slightly wrong,
which fails an all-triples-present guard as a whole. Authoring *new* content in
``correct_value`` has no such problem, because nothing has to match.

So the fix is not to give the critic a better quoting syntax; it is to stop
asking it to quote. The graph chapter carries an id per triple, the critic cites
ids, and the loop resolves them by lookup. What the model cannot reliably
reproduce, it no longer has to.

The index is built once per critic call, held on the unit state, and checked
against the graph by fingerprint before any id is resolved -- the loop mutates
the graph between passes, so a fix carried forward as residual must not silently
resolve against a later numbering.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from rdflib.namespace import NamespaceManager
from rdflib.term import BNode, Literal, Node, URIRef

from ontocast.onto.rdfgraph import RDFGraph

Triple = tuple[Node, Node, Node]

#: Sort rank per term kind, so a ``Literal`` and a ``URIRef`` with the same
#: lexical form cannot tie and swap places between two builds of the same graph.
_TERM_RANK: dict[type, int] = {URIRef: 0, BNode: 1, Literal: 2}

RDF_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")


def _term_key(term: Node, nsmgr: NamespaceManager) -> tuple[int, str]:
    return (_TERM_RANK.get(type(term), 3), format_term(term, nsmgr))


def format_term(term: Node, nsmgr: NamespaceManager) -> str:
    """Render one term for the indexed listing.

    Literals go through ``Literal.n3``, never through the Turtle serializer's
    abbreviating path: rdflib's writer renders ``xsd:double``/``float``/
    ``decimal`` via ``%e`` and loses precision, which is why
    :data:`~ontocast.onto.rdfgraph.LOSSLESS_TURTLE_FORMAT` exists at all. The
    critic must see the value the graph actually holds, or it will report a
    rounding artifact as a defect.
    """
    if isinstance(term, BNode):
        return f"_:{term}"
    try:
        return term.n3(nsmgr)
    except Exception:
        # A term with an unbindable namespace still needs an id; falling back to
        # the raw form keeps it addressable rather than dropping it from the
        # listing, which would leave a triple the critic can see but not cite.
        return str(term)


@dataclass(frozen=True)
class IndexedTriple:
    """One line of the listing: a statement, and its id when it has one.

    A statement without an id is shown but not addressable. That is how the
    ontology chapter draws its read-only boundary: the retrieved catalog is
    context the critic must read and must not delete, so it simply has no number
    to cite.
    """

    triple: Triple
    triple_id: int | None


@dataclass(frozen=True)
class TripleIndex:
    """Ids ``1..N`` over exactly the triples a prompt chapter shows.

    ``fingerprint`` identifies the graph state the ids were assigned against.
    It is a digest of the same rendered lines the listing is built from, not
    :meth:`RDFGraph.hash`: that one runs URDNA2015 to get a canonical form
    stable across triple-store round trips, which is the right identity for the
    catalog and far more work than is needed to answer "is this still the graph
    I numbered?".
    """

    by_id: dict[int, Triple]
    ids: dict[Triple, int]
    fingerprint: str
    scope_size: int
    #: Subjects in listing order with their statements -- the render order is
    #: part of the contract, so it is recorded rather than recomputed.
    order: list[tuple[Node, list[IndexedTriple]]] = field(default_factory=list)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Accept instances only; this never crosses a serialization boundary.

        The index holds rdflib terms and is valid only for one graph state, so it
        is carried on the state as a within-call reference table and excluded
        from every dump. There is nothing to parse it *from*.
        """
        return core_schema.is_instance_schema(cls)

    def __len__(self) -> int:
        return len(self.by_id)

    @property
    def is_empty(self) -> bool:
        return not self.by_id

    def resolve(self, triple_id: int) -> Triple | None:
        """Return the triple for ``triple_id``, or ``None`` if it is unknown."""
        return self.by_id.get(triple_id)

    def matches(self, graph: RDFGraph) -> bool:
        """Whether ``graph`` is still in the state these ids were assigned to."""
        return self.fingerprint == fingerprint_graph(graph)


def _sorted_triples(graph: RDFGraph, nsmgr: NamespaceManager) -> list[Triple]:
    # rdflib's iteration order is store-dependent -- RDFGraph branches on
    # OxigraphStore in several places -- so the order is imposed here rather
    # than inherited. Predicates sort with rdf:type first so a subject's block
    # opens with what it is.
    def key(triple: Triple) -> tuple:
        subject, predicate, obj = triple
        return (
            _term_key(subject, nsmgr),
            (0, "") if predicate == RDF_TYPE else (1, format_term(predicate, nsmgr)),
            _term_key(obj, nsmgr),
        )

    return sorted(graph, key=key)


def fingerprint_graph(graph: RDFGraph) -> str:
    """Digest of a graph's rendered triples, used to detect drift between passes."""
    nsmgr = graph.namespace_manager
    digest = hashlib.sha256()
    for subject, predicate, obj in _sorted_triples(graph, nsmgr):
        digest.update(
            f"{format_term(subject, nsmgr)}\t"
            f"{format_term(predicate, nsmgr)}\t"
            f"{format_term(obj, nsmgr)}\n".encode()
        )
    return digest.hexdigest()


def build_triple_index(
    graph: RDFGraph, *, scope: RDFGraph | None = None
) -> TripleIndex:
    """Assign ids to ``graph``'s triples, grouped and ordered by subject.

    Args:
        graph: The graph exactly as the prompt will show it. For a chapter that
            condenses before serializing, pass the *condensed* graph: an id on a
            triple the critic never sees is a delete ordered blind.
        scope: When given, only triples also present here receive an id. The
            rest are still listed -- the critic needs them to judge -- but cannot
            be cited, and so cannot be removed. This is what makes a catalog
            delete structurally inexpressible on the ontology side rather than
            merely reported after the fact.

    Returns:
        TripleIndex: Ids, the reverse lookup, the listing order, and the
        fingerprint of the state they were assigned against.
    """
    nsmgr = graph.namespace_manager
    by_id: dict[int, Triple] = {}
    ids: dict[Triple, int] = {}
    order: list[tuple[Node, list[IndexedTriple]]] = []
    digest = hashlib.sha256()

    next_id = 0
    current_subject: Node | None = None
    current_block: list[IndexedTriple] = []
    for triple in _sorted_triples(graph, nsmgr):
        subject, predicate, obj = triple
        digest.update(
            f"{format_term(subject, nsmgr)}\t"
            f"{format_term(predicate, nsmgr)}\t"
            f"{format_term(obj, nsmgr)}\n".encode()
        )
        triple_id: int | None = None
        if scope is None or triple in scope:
            next_id += 1
            triple_id = next_id
            by_id[triple_id] = triple
            ids[triple] = triple_id
        if subject != current_subject:
            current_subject = subject
            current_block = []
            order.append((subject, current_block))
        current_block.append(IndexedTriple(triple=triple, triple_id=triple_id))

    return TripleIndex(
        by_id=by_id,
        ids=ids,
        fingerprint=digest.hexdigest(),
        scope_size=len(by_id),
        order=order,
    )
