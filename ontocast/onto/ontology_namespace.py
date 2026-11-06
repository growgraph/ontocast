"""The namespace an ontology defines its terms in.

An ontology's IRI does not say where its terms live: OM-2 is
``…/om-2`` with terms under ``…/om-2/``, PROV-O is ``…/prov-o`` with terms
under ``…/prov#``. The namespace is read from the graph instead: the one the
header declares (``vann:preferredNamespaceUri``) when terms live there, else
the one in which the graph describes the most terms.
"""

from __future__ import annotations

from collections import Counter

from rdflib import Graph, URIRef

from ontocast.onto.constants import VANN
from ontocast.onto.iri_policy import normalize_namespace_iri, split_namespace_local


def described_namespaces(
    graph: Graph, *, exclude: URIRef | None = None
) -> frozenset[str]:
    """The namespaces in which ``graph`` describes the most terms (ties kept).

    A term is described when it is the subject of a triple. Terms an ontology
    only reuses (a standard vocabulary's annotation properties, a borrowed
    superclass) are described less often than its own, so they lose.

    Args:
        graph: An ontology graph.
        exclude: A subject not counted, typically the ontology header.
    """
    counts: Counter[str] = Counter()
    for subject in set(graph.subjects()):
        if not isinstance(subject, URIRef) or subject == exclude:
            continue
        namespace, _ = split_namespace_local(str(subject))
        if namespace:
            counts[namespace] += 1
    if not counts:
        return frozenset()
    top = max(counts.values())
    return frozenset(ns for ns, n in counts.items() if n == top)


def ontology_namespace(graph: Graph, iri: str) -> str:
    """The namespace the ontology ``iri`` defines its terms in.

    In order: the header's ``vann:preferredNamespaceUri`` when ``graph``
    describes a term under it; else :func:`described_namespaces`, a tie going
    to the namespace under the ontology IRI, then to the lexically first; else
    ``<iri>#``, the namespace OntoCast mints terms under.
    """
    header = URIRef(iri)
    subjects = [str(s) for s in graph.subjects() if isinstance(s, URIRef)]
    for declared in sorted(
        str(o) for o in graph.objects(header, VANN.preferredNamespaceUri)
    ):
        if any(s != iri and s.startswith(declared) for s in subjects):
            return declared
    candidates = described_namespaces(graph, exclude=header)
    if candidates:
        stem = iri.rstrip("/#")
        return min(candidates, key=lambda ns: (not ns.startswith(stem), ns))
    return normalize_namespace_iri(iri, context="ontology")


__all__ = ["described_namespaces", "ontology_namespace"]
