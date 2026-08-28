"""Structural telemetry for a facts graph.

Connectivity summaries answer "did extraction produce one knowledge graph or a
pile of islands?" without any domain vocabulary: nodes are the fact-namespace
IRIs, edges are IRI-to-IRI statements outside the annotation substrate. Used
by the run manifest so fragmentation regressions are visible per document
instead of requiring an offline notebook.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field
from rdflib import RDF, RDFS, URIRef

from ontocast.onto.rdfgraph import RDFGraph

#: Predicates that describe a node rather than relate two entities; edges via
#: these say nothing about knowledge-graph cohesion.
_ANNOTATION_PREDICATES = {RDF.type, RDFS.label, RDFS.comment, RDFS.seeAlso}


class GraphShapeMetrics(BaseModel):
    """Connectivity summary of one facts graph."""

    nodes: int = Field(description="Fact-namespace IRIs present in the graph.")
    edges: int = Field(description="Distinct IRI-to-IRI links between them.")
    components: int = Field(description="Connected components (undirected).")
    largest_component: int = Field(description="Node count of the largest one.")
    isolated_nodes: int = Field(description="Nodes with no entity link at all.")


def facts_graph_shape_metrics(
    graph: RDFGraph, fact_namespaces: Sequence[str]
) -> GraphShapeMetrics:
    """Compute connectivity metrics over the graph's fact-namespace entities.

    Objects outside the fact namespaces (catalog reference individuals,
    units) still connect the fact nodes that share them — linking through a
    catalog individual is by design, not fragmentation.
    """
    namespaces = tuple(ns for ns in fact_namespaces if ns)

    def is_fact_node(term: object) -> bool:
        return isinstance(term, URIRef) and str(term).startswith(namespaces)

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    nodes: set[str] = set()
    linked: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for subject, predicate, obj in graph:
        subject_is_fact = is_fact_node(subject)
        if subject_is_fact:
            nodes.add(str(subject))
        if predicate in _ANNOTATION_PREDICATES or not isinstance(obj, URIRef):
            continue
        object_is_fact = is_fact_node(obj)
        if object_is_fact:
            nodes.add(str(obj))
        if not (subject_is_fact or object_is_fact) or str(subject) == str(obj):
            continue
        # A shared non-fact object (catalog individual) still merges the fact
        # nodes pointing at it, so it participates in union-find but is not
        # counted as a node.
        for term in (str(subject), str(obj)):
            parent.setdefault(term, term)
        union(str(subject), str(obj))
        if subject_is_fact and object_is_fact:
            edges.add((str(subject), str(obj)))
        if subject_is_fact:
            linked.add(str(subject))
        if object_is_fact:
            linked.add(str(obj))

    components = {find(node) for node in parent if node in nodes}
    component_sizes: dict[str, int] = {}
    for node in nodes:
        if node in parent:
            root = find(node)
            component_sizes[root] = component_sizes.get(root, 0) + 1
    isolated = len(nodes - linked)
    return GraphShapeMetrics(
        nodes=len(nodes),
        edges=len(edges),
        components=len(components) + isolated,
        largest_component=max(component_sizes.values(), default=1 if nodes else 0),
        isolated_nodes=isolated,
    )
