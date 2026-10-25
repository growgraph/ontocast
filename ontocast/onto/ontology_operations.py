"""Algorithmic operations over ontology versions."""

import logging
from collections import defaultdict

from rdflib import URIRef

from ontocast.onto.ontology import Ontology, OntologyHeader
from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)


def merge_ontologies(onto1: Ontology, onto2: Ontology) -> Ontology:
    """The union of two versions of one ontology, as an unstamped merge artifact.

    Takes every triple of both, reports contradictions (one subject and
    predicate with different objects) and keeps both sides of each. The result
    has both inputs as bases, so stamping it yields a version with two priors.

    Raises:
        ValueError: the inputs are versions of different ontologies, or either
            was never stamped.
    """
    if onto1.iri != onto2.iri:
        raise ValueError(
            f"Cannot merge ontologies with different IRIs: {onto1.iri} != {onto2.iri}"
        )
    bases: list[OntologyHeader] = []
    for onto in (onto1, onto2):
        bases.append(
            onto.bases[0]
            if len(onto.bases) == 1 and onto.bases[0].version_iri == onto.version_iri
            else OntologyHeader.from_ontology(onto)
        )

    merged_graph = onto1.graph + onto2.graph
    contradictions = _find_contradictions(onto1.graph, onto2.graph)
    if contradictions:
        logger.warning("Found %d contradictions in merge", len(contradictions))
        for (s, p), (objects1, objects2) in contradictions.items():
            logger.debug("Contradiction: %s %s -> %s vs %s", s, p, objects1, objects2)

    merged = Ontology(
        ontology_id=onto1.ontology_id or onto2.ontology_id,
        iri=onto1.iri,
        title=onto1.title or onto2.title,
        description=onto1.description or onto2.description,
        graph=merged_graph,
        bases=bases,
    )
    merged.content_hash()
    logger.info(
        "Merged %s and %s -> %s",
        onto1.version_iri,
        onto2.version_iri,
        merged.hash,
    )
    return merged


def _find_contradictions(
    graph1: RDFGraph, graph2: RDFGraph
) -> dict[tuple[URIRef, URIRef], tuple[set, set]]:
    """Subject-predicate pairs whose objects differ between the two graphs."""
    objects1: dict[tuple, set] = defaultdict(set)
    objects2: dict[tuple, set] = defaultdict(set)
    for s, p, o in graph1:
        objects1[(s, p)].add(o)
    for s, p, o in graph2:
        objects2[(s, p)].add(o)
    contradictions: dict[tuple[URIRef, URIRef], tuple[set, set]] = {}
    for key in objects1.keys() & objects2.keys():
        if objects1[key] != objects2[key]:
            contradictions[key] = (objects1[key], objects2[key])
    return contradictions
