"""How the aggregated facts hang together.

:func:`facts_connectivity` counts the connected components of the served
graph and the instances linked to nothing; :func:`isolated_instances` and
:func:`silent_instances` name them, :func:`states_a_value` says whether an
instance states anything, :func:`isolated_report` lists the isolated ones;
:func:`twin_labels` names the labels
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
from ontocast.onto.model import FactsIsolatedInstance
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import FactScope
from ontocast.tool.facts_validation.statements import StatementContract

# Predicates that link nothing: typing and alias handles.
_NON_LINKS: frozenset[URIRef] = frozenset({RDF.type, OWL.sameAs})


def _links(scope: FactScope) -> Iterable[tuple[Node, Node, Node]]:
    """The fact triples whose object is a node, typing and aliases left out."""
    for triple in scope.triples():
        _, predicate, obj = triple
        if isinstance(obj, (URIRef, BNode)) and predicate not in _NON_LINKS:
            yield triple


def instance_links(
    graph: RDFGraph, fact_namespaces: Sequence[str]
) -> set[tuple[Node, Node, Node]]:
    """Triples that link an instance (or a blank node) to another node.

    Typing, ``owl:sameAs`` and the pipeline's provenance scaffolding are not
    links, and neither is a literal value. A domain property is a link in any
    vocabulary, PROV-O included.
    """
    scope = FactScope.of(graph, fact_namespaces)
    return {
        (s, p, o)
        for s, p, o in _links(scope)
        if (s in scope.instances or isinstance(s, BNode))
        and (o in scope.instances or isinstance(o, BNode))
    }


class FactsConnectivity(BaseModel):
    """How the instances of a facts graph hang together."""

    instances: int = 0
    #: Connected groups of instances (the document node joins what it links).
    components: int = 0
    #: Instances linked to no other instance and not to the document.
    isolated: int = 0


def _components(
    scope: FactScope, instances: set[URIRef], document: URIRef | None
) -> tuple[Callable[[Node], Node], dict[Node, int]]:
    """Union the instances (and ``document``) over links, through blank nodes.

    Returns the root finder and each root's member count; blank nodes join but
    are not members.
    """
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

    for s, _, o in _links(scope):
        if joinable(s) and joinable(o) and s != o:
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

    Instances (:attr:`~ontocast.onto.scaffolding.FactScope.instances`)
    are joined by any link (:func:`instance_links`' notion), directly or
    through blank nodes; the ``document`` node, when given, is not an instance
    but joins what it links (cited works, metadata), as a reader of the graph
    sees it. Provenance reifiers and chunk nodes join nothing.
    """
    scope = FactScope.of(graph, fact_namespaces, document)
    instances = set(scope.instances) - {document}
    if not instances:
        return FactsConnectivity()
    find, sizes = _components(scope, instances, document)
    return FactsConnectivity(
        instances=len(instances),
        components=len({find(node) for node in instances}),
        isolated=sum(1 for node in instances if sizes[find(node)] == 1),
    )


def isolated_instances(
    graph: RDFGraph, fact_namespaces: Sequence[str], document: URIRef | None = None
) -> set[URIRef]:
    """Instances linked to no other instance and not to ``document``."""
    scope = FactScope.of(graph, fact_namespaces, document)
    instances = set(scope.instances) - {document}
    if not instances:
        return set()
    find, sizes = _components(scope, instances, document)
    return {node for node in instances if sizes[find(node)] == 1}


def states_a_value(scope: FactScope, node: Node) -> bool:
    """Whether ``node`` states a value of its own.

    A value is a blank-node object (a structured value, before values are
    named) or a literal on a property that is not an annotation
    (:func:`~ontocast.onto.annotation.is_annotation`). Typing, ``owl:sameAs``
    and provenance scaffolding state nothing. What states no value is a type
    and a name: a mention, not a fact.
    """
    for _, predicate, obj in scope.triples((node, None, None)):
        if predicate in _NON_LINKS:
            continue
        if isinstance(obj, BNode) or (
            isinstance(obj, Literal) and not is_annotation(predicate)
        ):
            return True
    return False


def silent_instances(
    graph: RDFGraph, fact_namespaces: Sequence[str], document: URIRef | None = None
) -> set[URIRef]:
    """Isolated instances that state nothing (:func:`states_a_value`)."""
    scope = FactScope.of(graph, fact_namespaces, document)
    return {
        instance
        for instance in isolated_instances(graph, fact_namespaces, document)
        if not states_a_value(scope, instance)
    }


def isolated_report(
    graph: RDFGraph, isolated: Iterable[URIRef]
) -> list[FactsIsolatedInstance]:
    """The isolated instances, ordered by IRI, each with its label and kind.

    ``holds_value`` when it states a value (:func:`states_a_value`): a
    statement whose subject the graph never names. ``other`` when it states
    none.
    """
    scope = FactScope.of(graph, None)
    report: list[FactsIsolatedInstance] = []
    for node in sorted(isolated, key=str):
        label = graph.value(node, RDFS.label)
        report.append(
            FactsIsolatedInstance(
                iri=str(node),
                label=str(label) if label is not None else "",
                kind="holds_value" if states_a_value(scope, node) else "other",
            )
        )
    return report


def twin_labels(
    graph: RDFGraph,
    fact_namespaces: Sequence[str],
    document: URIRef | None = None,
    *,
    contract: StatementContract,
) -> dict[str, int]:
    """Labels two or more instances of one role share, with how many share each.

    A label is compared case-folded with its whitespace collapsed, and only
    between instances that could be one thing: statements with statements,
    the properties statements observe with properties, other instances with
    other instances. Values are left out, since a value is identified by what
    it states, not by its label; an observation labelled like the property it
    observes is no twin of it, and neither are two instances one links to the
    other (a series and its edition). Twins are the duplicates aggregation left, or
    distinct things with one name; the count says how many identities one
    name still carries.
    """
    observed = _observed_properties(graph, contract)
    holders: dict[tuple[str, str], set[URIRef]] = {}
    for instance in FactScope.of(graph, fact_namespaces, document).instances - {
        document
    }:
        role = _twin_role(graph, instance, contract, observed)
        if role is None:
            continue
        for label in graph.objects(instance, RDFS.label):
            if isinstance(label, Literal):
                key = " ".join(str(label).split()).casefold()
                if key:
                    holders.setdefault((key, role), set()).add(instance)
    counts: dict[str, int] = {}
    for (label, _), instances in sorted(holders.items()):
        twins = {
            instance
            for instance in instances
            if any(
                other != instance and not _linked(graph, instance, other)
                for other in instances
            )
        }
        if twins:
            counts[label] = counts.get(label, 0) + len(twins)
    return counts


def _linked(graph: RDFGraph, a: URIRef, b: URIRef) -> bool:
    """Whether a triple other than an annotation joins ``a`` and ``b``."""
    return any(
        not is_annotation(predicate)
        for predicate in (*graph.predicates(a, b), *graph.predicates(b, a))
    )


def _observed_properties(graph: RDFGraph, contract: StatementContract) -> set[Node]:
    """Nodes a statement's ``property`` link names, in either direction."""
    if "property" not in contract.roles:
        return set()
    links = contract.links("property")
    found: set[Node] = set()
    for link in links.forward:
        found.update(graph.objects(None, URIRef(link)))
    for link in links.inverse:
        found.update(graph.subjects(URIRef(link), None))
    return found


def _twin_role(
    graph: RDFGraph, instance: URIRef, contract: StatementContract, observed: set[Node]
) -> str | None:
    """The role ``instance`` is compared within, or ``None`` for a value."""
    types = set(graph.objects(instance, RDF.type))
    if contract.is_value(types):
        return None
    if contract.is_statement(types):
        return "statement"
    if instance in observed or any(
        str(cls) in contract.property_classes for cls in types
    ):
        return "property"
    return "instance"


def links_lost(before: Iterable[tuple[Node, Node, Node]], after: RDFGraph) -> int:
    """How many of the links ``before`` the graph ``after`` no longer holds."""
    return sum(1 for triple in before if triple not in after)


__all__ = [
    "FactsConnectivity",
    "facts_connectivity",
    "instance_links",
    "isolated_instances",
    "isolated_report",
    "links_lost",
    "silent_instances",
    "states_a_value",
    "twin_labels",
]
