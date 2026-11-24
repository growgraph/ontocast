from __future__ import annotations

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import FactScope


def strip_provenance_triples_for_embedding(graph: RDFGraph) -> RDFGraph:
    """Return a copy of *graph* suitable for embedding generation.

    Provenance scaffolding (chunk nodes, reifiers) and the document node would
    dominate embedding neighborhoods and make retrieval match document
    provenance rather than asserted facts, so every triple touching them is
    dropped. Domain triples stay whatever their vocabulary: an ontology's own
    PROV-O or Dublin Core annotations are text worth embedding.
    """
    result = RDFGraph()
    for prefix, namespace in graph.namespaces():
        if prefix:
            result.bind(prefix, namespace)

    scope = FactScope.of(graph, None)
    for triple in scope.triples():
        subject, _, obj = triple
        if subject not in scope.documents and obj not in scope.documents:
            result.add(triple)
    return result
