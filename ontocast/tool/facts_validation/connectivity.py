"""How the aggregated facts hang together.

:func:`facts_connectivity` counts the connected components of the served
graph and the instances linked to nothing; :func:`instance_links` and
:func:`links_lost` count the links the gate's repairs removed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel
from rdflib import BNode, URIRef
from rdflib.namespace import OWL, PROV, RDF
from rdflib.term import Node

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation.terms import fact_instances

# Predicates that link nothing: typing, and alias or lineage handles.
_NON_LINKS: frozenset[URIRef] = frozenset({RDF.type, OWL.sameAs})


def _is_link(predicate: Node, obj: Node) -> bool:
    return (
        isinstance(obj, (URIRef, BNode))
        and predicate not in _NON_LINKS
        and not str(predicate).startswith(str(PROV))
    )


def instance_links(
    graph: RDFGraph, fact_namespaces: Sequence[str]
) -> set[tuple[Node, Node, Node]]:
    """Triples that link an instance (or a blank node) to another node.

    Typing, ``owl:sameAs`` and ``prov:`` arcs are not links, and neither is a
    literal value.
    """
    instances = fact_instances(graph, fact_namespaces)
    return {
        (s, p, o)
        for s, p, o in graph
        if _is_link(p, o)
        and (s in instances or isinstance(s, BNode))
        and (o in instances or isinstance(o, BNode))
    }


class FactsConnectivity(BaseModel):
    """How the instances of a facts graph hang together."""

    instances: int = 0
    #: Connected groups of instances (the document node joins what it links).
    components: int = 0
    #: Instances linked to no other instance and not to the document.
    isolated: int = 0


def facts_connectivity(
    graph: RDFGraph, fact_namespaces: Sequence[str], document: URIRef | None = None
) -> FactsConnectivity:
    """Count the connected components and isolated instances of ``graph``.

    Instances (:func:`~ontocast.tool.facts_validation.terms.fact_instances`)
    are joined by any link (:func:`_is_link`), directly or through blank nodes;
    the ``document`` node, when given, is not an instance but joins what it
    links (cited works, metadata), as a reader of the graph sees it.
    Provenance reifiers and chunk nodes join nothing.
    """
    instances = fact_instances(graph, fact_namespaces) - {document}
    if not instances:
        return FactsConnectivity()
    members: set[Node] = set(instances)
    if document is not None:
        members.add(document)
    parent: dict[Node, Node] = {}

    def find(node: Node) -> Node:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def joinable(node: Node) -> bool:
        return node in members or isinstance(node, BNode)

    for s, p, o in graph:
        if _is_link(p, o) and joinable(s) and joinable(o) and s != o:
            parent[find(s)] = find(o)

    sizes: dict[Node, int] = {}
    for node in members:
        root = find(node)
        sizes[root] = sizes.get(root, 0) + 1
    roots_with_instances = {find(node) for node in instances}
    return FactsConnectivity(
        instances=len(instances),
        components=len(roots_with_instances),
        isolated=sum(1 for node in instances if sizes[find(node)] == 1),
    )


def links_lost(before: Iterable[tuple[Node, Node, Node]], after: RDFGraph) -> int:
    """How many of the links ``before`` the graph ``after`` no longer holds."""
    return sum(1 for triple in before if triple not in after)


__all__ = [
    "FactsConnectivity",
    "facts_connectivity",
    "instance_links",
    "links_lost",
]
