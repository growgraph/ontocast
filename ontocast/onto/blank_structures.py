"""Blank nodes as structures: an OWL restriction, an RDF list, an anonymous axiom.

A blank node has no name of its own, so a statement about it means nothing
apart from the statements that reach it. A *structure* is a set of blank nodes
connected to each other through statements, with every statement whose subject
is one of them. Its *inbound* statements are those whose named subject points
at one of its nodes (``home:C rdfs:subClassOf _:r``), and their subjects are
its *anchors*. A structure is filed, kept and removed as one piece: half a
restriction is not a restriction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rdflib import BNode, Graph, URIRef
from rdflib.term import Node

Triple = tuple[Node, Node, Node]


@dataclass(frozen=True)
class BlankStructure:
    """Connected blank nodes, their statements and what points at them."""

    nodes: frozenset[BNode]
    #: Statements whose subject is a node of the structure.
    triples: frozenset[Triple]
    #: Statements whose named subject has a node of the structure as object.
    inbound: frozenset[Triple]

    @property
    def anchors(self) -> frozenset[URIRef]:
        """The named subjects that point at the structure."""
        return frozenset(s for s, _, _ in self.inbound if isinstance(s, URIRef))

    @property
    def references(self) -> frozenset[URIRef]:
        """The named IRIs the structure's own statements use, as predicate or object."""
        return frozenset(
            node
            for _, p, o in self.triples
            for node in (p, o)
            if isinstance(node, URIRef)
        )


def blank_structures(graph: Graph) -> list[BlankStructure]:
    """Every blank-node structure of ``graph``, in no particular order."""
    parent: dict[BNode, BNode] = {}

    def find(node: BNode) -> BNode:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def add(node: BNode) -> None:
        parent.setdefault(node, node)

    for s, _, o in graph:
        if isinstance(s, BNode):
            add(s)
        if isinstance(o, BNode):
            add(o)
        if isinstance(s, BNode) and isinstance(o, BNode):
            a, b = find(s), find(o)
            if a != b:
                parent[a] = b

    if not parent:
        return []

    nodes: dict[BNode, set[BNode]] = {}
    for node in parent:
        nodes.setdefault(find(node), set()).add(node)
    triples: dict[BNode, set[Triple]] = {root: set() for root in nodes}
    inbound: dict[BNode, set[Triple]] = {root: set() for root in nodes}
    for triple in graph:
        s, _, o = triple
        if isinstance(s, BNode):
            triples[find(s)].add(triple)
        elif isinstance(o, BNode):
            inbound[find(o)].add(triple)
    return [
        BlankStructure(
            nodes=frozenset(members),
            triples=frozenset(triples[root]),
            inbound=frozenset(inbound[root]),
        )
        for root, members in nodes.items()
    ]


def structure_closure(graph: Graph, doomed: Iterable[Triple]) -> set[Triple]:
    """``doomed`` plus every blank-node structure its removal would leave broken.

    A structure goes whole, with the statements pointing at it, when one of its
    own statements is doomed (it would be left partial) or when every statement
    pointing at it is doomed (it would be left orphaned). An unanchored
    structure goes only when one of its own statements is doomed. Statements of
    ``doomed`` that are not in ``graph`` are kept in the result.
    """
    closure = set(doomed)
    for structure in blank_structures(graph):
        partial = bool(structure.triples & closure)
        orphaned = bool(structure.inbound) and structure.inbound <= closure
        if partial or orphaned:
            closure |= structure.triples | structure.inbound
    return closure


def remove_with_structures(graph: Graph, doomed: Iterable[Triple]) -> int:
    """Remove ``doomed`` and every structure it would leave broken.

    See :func:`structure_closure`. Returns the number of statements removed.
    """
    removed = {triple for triple in structure_closure(graph, doomed) if triple in graph}
    for triple in removed:
        graph.remove(triple)
    return len(removed)


__all__ = [
    "BlankStructure",
    "blank_structures",
    "remove_with_structures",
    "structure_closure",
]
