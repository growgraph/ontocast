"""The nodes OntoCast mints around extracted facts.

The pipeline records where each fact came from with standard vocabulary:
content-unit (chunk) nodes typed ``prov:Entity`` and ``schema:Text``, and one
reifier per asserted triple (``rdf:reifies`` in RDF 1.2, ``rdf:Statement`` in
the classic export) pointing at its chunks with ``prov:wasDerivedFrom``.

Scaffolding is identified by these *nodes*, never by namespace: a facts graph
may use PROV-O, schema.org or Dublin Core as domain vocabulary, and those
triples are facts like any other. The document node is not scaffolding: its
title, date and source are facts about the document.

:class:`FactScope` is the one definition of a fact triple and a fact instance
every check reads.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from rdflib import FOAF, OWL, RDF, BNode, Graph, URIRef
from rdflib.term import Node

from ontocast.onto.constants import PROV, RDF_REIFIES

if TYPE_CHECKING:
    from ontocast.onto.rdfgraph import RDFGraph

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


def document_nodes(graph: Graph, document: Node | None = None) -> set[Node]:
    """The document node the pipeline mints: typed ``prov:Entity`` and ``foaf:Document``.

    Given the document's IRI, only that node: a work the text cites, typed the
    same way, is a fact, never the document being read.
    """
    found = set(graph.subjects(RDF.type, PROV.Entity)) & set(
        graph.subjects(RDF.type, FOAF.Document)
    )
    return found if document is None else found & {document}


def scaffolding_nodes(graph: Graph) -> set[Node]:
    """Chunk nodes and reifiers: the bookkeeping that carries no fact."""
    return chunk_nodes(graph) | reifier_nodes(graph)


#: Objects of these predicates are alias handles, not instances.
_HANDLE_PREDICATES: frozenset[Node] = frozenset({OWL.sameAs})

Triple = tuple[Node, Node, Node]


class FactScope(BaseModel):
    """The facts of a graph: what every check iterates and counts.

    A fact triple neither describes nor points at a scaffolding node (a chunk
    or a reifier). A fact instance is an IRI that is not minted by the
    pipeline (scaffolding, the document node) and is the subject of a fact
    triple -- or, with ``fact_namespaces``, an IRI object in one of them of a
    fact triple other than ``rdf:type`` and ``owl:sameAs``: the namespace is
    what says an IRI the graph only points at is an instance. Build with
    :meth:`of`; the graph must not change while the scope is in use.

    Attributes:
        graph: The graph.
        fact_namespaces: Namespaces holding minted instances; ``None`` admits
            every namespace, an empty tuple none.
        scaffolding: Its chunk nodes and reifiers.
        documents: Its document nodes.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    graph: Graph
    fact_namespaces: tuple[str, ...] | None
    scaffolding: frozenset[Node]
    documents: frozenset[Node]

    @classmethod
    def of(
        cls,
        graph: Graph,
        fact_namespaces: Sequence[str] | None,
        document: Node | None = None,
    ) -> FactScope:
        """The fact scope of ``graph``.

        Args:
            graph: Facts graph for one unit or the aggregated document.
            fact_namespaces: Namespaces holding minted instances, empty ones
                ignored; ``None`` admits every namespace.
            document: The IRI of the document the graph was extracted from,
                when known (:func:`document_nodes`).
        """
        return cls(
            graph=graph,
            fact_namespaces=None
            if fact_namespaces is None
            else tuple(ns for ns in fact_namespaces if ns),
            scaffolding=frozenset(scaffolding_nodes(graph)),
            documents=frozenset(document_nodes(graph, document)),
        )

    def is_fact(self, triple: Triple) -> bool:
        """Whether ``triple`` neither describes nor points at scaffolding."""
        subject, _, obj = triple
        return subject not in self.scaffolding and obj not in self.scaffolding

    def triples(
        self, pattern: tuple[Node | None, Node | None, Node | None] = (None, None, None)
    ) -> Iterator[Triple]:
        """The fact triples matching ``pattern``."""
        for triple in self.graph.triples(pattern):
            if self.is_fact(triple):
                yield triple

    def facts(self) -> RDFGraph:
        """The fact triples as a new graph, with the namespace bindings.

        RDF 1.2 reifications are not copied. ``graph`` is not modified.
        """
        from ontocast.onto.rdfgraph import RDFGraph

        facts = RDFGraph()
        for triple in self.triples():
            facts.add(triple)
        for prefix, namespace in self.graph.namespaces():
            facts.bind(prefix, namespace)
        return facts

    def in_namespaces(self, node: Node) -> bool:
        """Whether ``node`` is an IRI in the fact namespaces."""
        if not isinstance(node, URIRef):
            return False
        if self.fact_namespaces is None:
            return True
        return str(node).startswith(self.fact_namespaces)

    def holds(self, node: Node) -> bool:
        """Whether ``node`` is a fact node: an instance or a blank node of the facts.

        A blank node holds unless it is scaffolding (a blank reifier): a
        value written inline belongs to the facts that carry it.
        """
        if isinstance(node, BNode):
            return node not in self.scaffolding
        return node in self.instances

    @cached_property
    def subjects(self) -> frozenset[URIRef]:
        """The fact instances that are the subject of a fact triple."""
        minted = self.scaffolding | self.documents
        return frozenset(
            subject
            for subject in {s for s, _, _ in self.triples()}
            if isinstance(subject, URIRef)
            and subject not in minted
            and self.in_namespaces(subject)
        )

    @cached_property
    def instances(self) -> frozenset[URIRef]:
        """The fact instances: :attr:`subjects` and, with fact namespaces, the
        IRI objects in them."""
        if self.fact_namespaces is None:
            return self.subjects
        minted = self.scaffolding | self.documents
        referenced = {
            obj
            for _, predicate, obj in self.triples()
            if isinstance(obj, URIRef)
            and predicate != RDF.type
            and predicate not in _HANDLE_PREDICATES
            and obj not in minted
            and self.in_namespaces(obj)
        }
        return self.subjects | referenced


__all__ = [
    "FactScope",
    "chunk_nodes",
    "document_nodes",
    "reifier_nodes",
    "scaffolding_nodes",
]
