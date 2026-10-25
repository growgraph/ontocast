"""Shared helpers for triple store backends: the catalog read, strictly."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TypeVar

from rdflib import Graph

from ontocast.onto.ontology import (
    Ontology,
    OntologyHeader,
    OntologyPropertiesWithLineage,
)
from ontocast.onto.ontology_header import (
    CatalogContractError,
    catalog_heads,
    header_from_fields,
    ontology_from_version_graph,
)
from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)

#: Anything carrying version lineage: a header or a materialized ontology.
LineageT = TypeVar("LineageT", bound=OntologyPropertiesWithLineage)

#: One row per ``(named graph, prior version)``; fold with
#: :func:`headers_from_select_rows`. Mirrors the contract in
#: :mod:`ontocast.onto.ontology_header`.
ONTOLOGY_HEADER_QUERY = """
PREFIX owl:     <http://www.w3.org/2002/07/owl#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX vann:    <http://purl.org/vocab/vann/>
PREFIX oc:      <https://growgraph.dev/ontocast#>
SELECT DISTINCT ?g ?onto ?versionIri ?version ?prior ?prefix ?title ?description
                ?created ?modified ?removed
WHERE {
  GRAPH ?g {
    ?onto a owl:Ontology .
    OPTIONAL { ?onto owl:versionIRI ?versionIri }
    OPTIONAL { ?onto owl:versionInfo ?version }
    OPTIONAL { ?onto owl:priorVersion ?prior }
    OPTIONAL { ?onto vann:preferredNamespacePrefix ?prefix }
    OPTIONAL { ?onto dcterms:title ?title }
    OPTIONAL { ?onto dcterms:description ?description }
    OPTIONAL { ?onto dcterms:created ?created }
    OPTIONAL { ?onto dcterms:modified ?modified }
    OPTIONAL { ?g a oc:RemovedVersion . BIND(true AS ?removed) }
  }
}
"""

#: Lists every named graph holding at least one triple.
LIST_NAMED_GRAPHS_QUERY = """
SELECT DISTINCT ?g WHERE {
  GRAPH ?g { ?s ?p ?o }
}
"""


def deterministic_turtle_serialization(graph: Graph) -> str:
    """Create a deterministic Turtle serialization of an RDF graph."""
    prefix_lines = [
        f"@prefix {p}: <{ns}> ."
        for p, ns in sorted(graph.namespace_manager.namespaces())
    ]
    triples_sorted = sorted(graph, key=lambda t: (str(t[0]), str(t[1]), str(t[2])))
    triple_lines = [
        f"{s.n3(graph.namespace_manager)} {p.n3(graph.namespace_manager)} {o.n3(graph.namespace_manager)} ."
        for s, p, o in triples_sorted
    ]
    return "\n".join(prefix_lines + [""] + triple_lines)


def ontology_from_named_graph(graph_uri: str, graph: Graph) -> Ontology:
    """Build the :class:`Ontology` stored as the version graph ``graph_uri``.

    Raises:
        CatalogContractError: the graph is not a version of an ontology. A
            foreign graph in an ontologies dataset is a fault, never a silent
            gap in the catalog.
    """
    deterministic_graph = RDFGraph()
    deterministic_graph.parse(
        data=deterministic_turtle_serialization(graph), format="turtle"
    )
    for prefix, namespace in graph.namespaces():
        if prefix:
            deterministic_graph.bind(prefix, namespace)
    ontology = ontology_from_version_graph(graph_uri, deterministic_graph)
    logger.debug(
        "Loaded ontology %s version %s from graph %s",
        ontology.iri,
        ontology.version,
        graph_uri,
    )
    return ontology


def headers_from_select_rows(rows: Iterable[dict[str, str]]) -> list[OntologyHeader]:
    """Fold :data:`ONTOLOGY_HEADER_QUERY` rows into one header per named graph.

    ``?prior`` is multi-valued, so the query returns a small cross product; rows
    sharing a graph describe one version and are merged.

    Raises:
        CatalogContractError: a graph's header is incomplete.
    """
    fields: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        graph_uri = row.get("g")
        if not graph_uri:
            continue
        for name, value in row.items():
            if name == "g" or not value:
                continue
            bucket = fields[graph_uri][name]
            if value not in bucket:
                bucket.append(value)
    return [
        header_from_fields(graph_uri, fields[graph_uri]) for graph_uri in sorted(fields)
    ]


def head_headers(headers: Sequence[OntologyHeader]) -> list[OntologyHeader]:
    """One head per ontology; diverged ontologies are logged and left out.

    A diverged ontology (several heads) is listed by the catalog and its
    versions can be read, but nothing extracts against it or extends it
    until a merge makes one head.
    """
    single, diverged = catalog_heads(headers)
    for iri, heads in diverged.items():
        logger.error(
            "Ontology %s has %d heads (%s); it is excluded from extraction "
            "until merged",
            iri,
            len(heads),
            ", ".join(h.version_iri or h.graph_uri for h in heads),
        )
    return single


__all__ = [
    "CatalogContractError",
    "LIST_NAMED_GRAPHS_QUERY",
    "LineageT",
    "ONTOLOGY_HEADER_QUERY",
    "deterministic_turtle_serialization",
    "head_headers",
    "headers_from_select_rows",
    "ontology_from_named_graph",
]
