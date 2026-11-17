"""The nodes OntoCast mints around extracted facts.

The pipeline records where each fact came from with standard vocabulary:
content-unit (chunk) nodes typed ``prov:Entity`` and ``schema:Text``, and one
reifier per asserted triple (``rdf:reifies`` in RDF 1.2, ``rdf:Statement`` in
the classic export) pointing at its chunks with ``prov:wasDerivedFrom``.

Scaffolding is identified by these *nodes*, never by namespace: a facts graph
may use PROV-O, schema.org or Dublin Core as domain vocabulary, and those
triples are facts like any other. The document node is not scaffolding: its
title, date and source are facts about the document.
"""

from __future__ import annotations

from rdflib import FOAF, RDF, Graph
from rdflib.term import Node

from ontocast.onto.constants import PROV, RDF_REIFIES

_SCHEMA_TEXT = (
    "http://schema.org/Text",
    "https://schema.org/Text",
)


def chunk_nodes(graph: Graph) -> set[Node]:
    """Content-unit nodes: typed both ``prov:Entity`` and ``schema:Text``."""
    entities = set(graph.subjects(RDF.type, PROV.Entity))
    return {
        node
        for node in entities
        if any(str(t) in _SCHEMA_TEXT for t in graph.objects(node, RDF.type))
    }


def reifier_nodes(graph: Graph) -> set[Node]:
    """Statement reifiers, in either provenance profile.

    RDF 1.2 reifiers are those of an
    :class:`~ontocast.onto.rdfgraph.RDFGraph`'s reifications, or subjects of
    ``rdf:reifies`` in a graph read otherwise; classic ones are typed
    ``rdf:Statement``.
    """
    from ontocast.onto.rdfgraph import RDFGraph

    nodes = set(graph.subjects(RDF_REIFIES, None)) | set(
        graph.subjects(RDF.type, RDF.Statement)
    )
    if isinstance(graph, RDFGraph) and graph.has_reifications:
        nodes |= graph.reifications.reifiers()
    return nodes


def document_nodes(graph: Graph) -> set[Node]:
    """The document node the pipeline mints: typed ``prov:Entity`` and ``foaf:Document``."""
    return set(graph.subjects(RDF.type, PROV.Entity)) & set(
        graph.subjects(RDF.type, FOAF.Document)
    )


def scaffolding_nodes(graph: Graph) -> set[Node]:
    """Chunk nodes and reifiers: the bookkeeping that carries no fact."""
    return chunk_nodes(graph) | reifier_nodes(graph)


def is_scaffolding(triple: tuple[Node, Node, Node], nodes: set[Node]) -> bool:
    """True when ``triple`` describes, or points at, a scaffolding node.

    Args:
        triple: An ``(s, p, o)`` triple.
        nodes: :func:`scaffolding_nodes` of the graph the triple belongs to.
    """
    subject, _, obj = triple
    return subject in nodes or obj in nodes


__all__ = [
    "chunk_nodes",
    "document_nodes",
    "is_scaffolding",
    "reifier_nodes",
    "scaffolding_nodes",
]
