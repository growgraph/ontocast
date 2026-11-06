"""How the aggregated facts hang together.

:func:`facts_connectivity` counts the connected components of the served
graph and the instances linked to nothing; :func:`isolated_instances` and
:func:`silent_instances` name them; :func:`twin_labels` names the labels
several instances share; :func:`instance_links` and :func:`links_lost` count
the links the gate's repairs removed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from pydantic import BaseModel
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from rdflib.term import Node

from ontocast.onto.annotation import is_annotation
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import is_scaffolding, scaffolding_nodes
from ontocast.tool.facts_validation.terms import fact_instances

# Predicates that link nothing: typing and alias handles.
_NON_LINKS: frozenset[URIRef] = frozenset({RDF.type, OWL.sameAs})


def _is_link(triple: tuple[Node, Node, Node], scaffolding: set[Node]) -> bool:
    _, predicate, obj = triple
    return (
        isinstance(obj, (URIRef, BNode))
        and predicate not in _NON_LINKS
        and not is_scaffolding(triple, scaffolding)
    )


def instance_links(
    graph: RDFGraph, fact_namespaces: Sequence[str]
) -> set[tuple[Node, Node, Node]]:
    """Triples that link an instance (or a blank node) to another node.

    Typing, ``owl:sameAs`` and the pipeline's provenance scaffolding are not
    links, and neither is a literal value. A domain property is a link in any
    vocabulary, PROV-O included.
    """
    instances = fact_instances(graph, fact_namespaces)
    scaffolding = scaffolding_nodes(graph)
    return {
        (s, p, o)
        for s, p, o in graph
        if _is_link((s, p, o), scaffolding)
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


def _components(
    graph: RDFGraph, instances: set[URIRef], document: URIRef | None
) -> tuple[Callable[[Node], Node], dict[Node, int]]:
    """Union the instances (and ``document``) over links, through blank nodes.

    Returns the root finder and each root's member count; blank nodes join but
    are not members.
    """
    members: set[Node] = set(instances)
    if document is not None:
        members.add(document)
    scaffolding = scaffolding_nodes(graph)
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
        if _is_link((s, p, o), scaffolding) and joinable(s) and joinable(o) and s != o:
            parent[find(s)] = find(o)

    sizes: dict[Node, int] = {}
    for node in members:
        root = find(node)
        sizes[root] = sizes.get(root, 0) + 1
    return find, sizes


def facts_connectivity(
    graph: RDFGraph, fact_namespaces: Sequence[str], document: URIRef | None = None
) -> FactsConnectivity:
    """Count the connected components and isolated instances of ``graph``.

    Instances (:func:`~ontocast.tool.facts_validation.terms.fact_instances`)
    are joined by any link (:func:`instance_links`' notion), directly or
    through blank nodes; the ``document`` node, when given, is not an instance
    but joins what it links (cited works, metadata), as a reader of the graph
    sees it. Provenance reifiers and chunk nodes join nothing.
    """
    instances = fact_instances(graph, fact_namespaces) - {document}
    if not instances:
        return FactsConnectivity()
    find, sizes = _components(graph, instances, document)
    return FactsConnectivity(
        instances=len(instances),
        components=len({find(node) for node in instances}),
        isolated=sum(1 for node in instances if sizes[find(node)] == 1),
    )


def isolated_instances(
    graph: RDFGraph, fact_namespaces: Sequence[str], document: URIRef | None = None
) -> set[URIRef]:
    """Instances linked to no other instance and not to ``document``."""
    instances = {
        node
        for node in fact_instances(graph, fact_namespaces) - {document}
        if isinstance(node, URIRef)
    }
    if not instances:
        return set()
    find, sizes = _components(graph, instances, document)
    return {node for node in instances if sizes[find(node)] == 1}


def silent_instances(
    graph: RDFGraph, fact_namespaces: Sequence[str], document: URIRef | None = None
) -> set[URIRef]:
    """Isolated instances that state nothing: no value of their own either.

    A value is a blank-node object (a structured value) or a literal on a
    property that is not an annotation (:func:`~ontocast.onto.annotation.is_annotation`).
    What is left is a type and a name: a mention, not a fact.
    """
    scaffolding = scaffolding_nodes(graph)

    def states_a_value(instance: URIRef) -> bool:
        for triple in graph.triples((instance, None, None)):
            _, predicate, obj = triple
            if predicate in _NON_LINKS or is_scaffolding(triple, scaffolding):
                continue
            if isinstance(obj, BNode) or (
                isinstance(obj, Literal) and not is_annotation(predicate)
            ):
                return True
        return False

    return {
        instance
        for instance in isolated_instances(graph, fact_namespaces, document)
        if not states_a_value(instance)
    }


def twin_labels(
    graph: RDFGraph, fact_namespaces: Sequence[str], document: URIRef | None = None
) -> dict[str, int]:
    """Labels two or more instances share, with how many share each.

    A label is compared case-folded with its whitespace collapsed. Twins are
    the duplicates aggregation left, or distinct things with one name; the
    count says how many identities one name still carries.
    """
    holders: dict[str, set[URIRef]] = {}
    for instance in fact_instances(graph, fact_namespaces) - {document}:
        if not isinstance(instance, URIRef):
            continue
        for label in graph.objects(instance, RDFS.label):
            if isinstance(label, Literal):
                key = " ".join(str(label).split()).casefold()
                if key:
                    holders.setdefault(key, set()).add(instance)
    return {
        label: len(instances)
        for label, instances in sorted(holders.items())
        if len(instances) > 1
    }


def links_lost(before: Iterable[tuple[Node, Node, Node]], after: RDFGraph) -> int:
    """How many of the links ``before`` the graph ``after`` no longer holds."""
    return sum(1 for triple in before if triple not in after)


__all__ = [
    "FactsConnectivity",
    "facts_connectivity",
    "instance_links",
    "isolated_instances",
    "links_lost",
    "silent_instances",
    "twin_labels",
]
