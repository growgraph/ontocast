from __future__ import annotations

from rdflib import DCTERMS, URIRef

from ontocast.onto.constants import PROV, RDF_REIFIES
from ontocast.onto.rdfgraph import RDFGraph


def strip_provenance_triples_for_embedding(graph: RDFGraph) -> RDFGraph:
    """Return a copy of *graph* suitable for embedding generation.

    Provenance/reification triples can dominate embedding neighborhoods and
    cause RAG retrieval to focus on document provenance rather than asserted
    facts. This helper removes a minimal set of provenance-related triples
    based on RDF predicate vocabulary.
    """
    result = RDFGraph()
    for prefix, namespace in graph.namespaces():
        if prefix:
            result.bind(prefix, namespace)

    prov_prefix = str(PROV)
    for s, p, o in graph:
        if not isinstance(p, URIRef):
            result.add((s, p, o))
            continue

        if p == DCTERMS.source:
            continue
        if p == RDF_REIFIES:
            continue
        if str(p).startswith(prov_prefix):
            continue

        result.add((s, p, o))

    return result
