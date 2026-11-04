from __future__ import annotations

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import document_nodes, is_scaffolding, scaffolding_nodes


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

    dropped = scaffolding_nodes(graph) | document_nodes(graph)
    for triple in graph:
        if not is_scaffolding(triple, dropped):
            result.add(triple)
    return result
