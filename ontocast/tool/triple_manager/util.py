"""Shared helpers for triple store backends."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import TypeVar

from rdflib import Graph
from rdflib.namespace import OWL, RDF

from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.ontology import Ontology, OntologyPropertiesWithLineage
from ontocast.onto.ontology_header import OntologyHeader
from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)

_HASH_IDENTIFIER_PREFIX = "hash:"
_PARENT_HASH_IRI_PREFIX = "urn:hash:"

#: One row per ``(named graph, ontology subject, parent hash)``; fold with
#: :func:`headers_from_select_rows`. Predicates mirror
#: :meth:`ontocast.onto.ontology.Ontology.sync_properties_to_graph`.
ONTOLOGY_HEADER_QUERY = """
PREFIX owl:     <http://www.w3.org/2002/07/owl#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX prov:    <http://www.w3.org/ns/prov#>
SELECT DISTINCT ?g ?onto ?version ?created ?identifier ?parent
WHERE {
  GRAPH ?g {
    ?onto a owl:Ontology .
    OPTIONAL { ?onto owl:versionInfo ?version }
    OPTIONAL { ?onto dcterms:created ?created }
    OPTIONAL {
      ?onto dcterms:identifier ?identifier
      FILTER(STRSTARTS(STR(?identifier), "hash:"))
    }
    OPTIONAL {
      ?onto prov:wasDerivedFrom ?parent
      FILTER(STRSTARTS(STR(?parent), "urn:hash:"))
    }
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


def compare_versions(ver1: str, ver2: str) -> int:
    """Compare two semantic version strings."""

    def _parse_version(v: str) -> tuple:
        parts = v.split(".")
        result = []
        for part in parts:
            numeric_part = re.sub(r"[^0-9].*$", "", part)
            result.append(int(numeric_part) if numeric_part else 0)
        while len(result) < 3:
            result.append(0)
        return tuple(result)

    try:
        v1_parts = _parse_version(ver1)
        v2_parts = _parse_version(ver2)
        if v1_parts < v2_parts:
            return -1
        if v1_parts > v2_parts:
            return 1
        return 0
    except Exception:
        return 1 if ver1 > ver2 else (-1 if ver1 < ver2 else 0)


def ontology_iri_for_named_graph(graph_uri: str, ontology_subject: str) -> str:
    """Resolve the catalog IRI for an ontology stored in ``graph_uri``.

    A versioned graph name (``<iri>#<hash>``) identifies the ontology by its base
    IRI; an unversioned graph name defers to the ``owl:Ontology`` subject.

    Args:
        graph_uri: Named graph URI the ontology was read from.
        ontology_subject: Subject typed ``owl:Ontology`` inside that graph.

    Returns:
        str: The ontology's catalog IRI.
    """
    if "#" in graph_uri:
        namespace, _ = split_namespace_local(graph_uri)
        if namespace is not None and namespace.endswith("#"):
            return namespace[:-1]
        return graph_uri
    return ontology_subject


def ontology_from_named_graph(graph_uri: str, graph: Graph) -> Ontology | None:
    """Build an :class:`Ontology` from a named-graph export."""
    try:
        deterministic_turtle = deterministic_turtle_serialization(graph)
        deterministic_graph = RDFGraph()
        deterministic_graph.parse(data=deterministic_turtle, format="turtle")
        for prefix, namespace in graph.namespaces():
            if prefix:
                deterministic_graph.bind(prefix, namespace)
        graph = deterministic_graph

        for onto_subj, _, _ in graph.triples((None, RDF.type, OWL.Ontology)):
            onto_iri = ontology_iri_for_named_graph(graph_uri, str(onto_subj))

            ontology = Ontology(graph=graph, iri=onto_iri)
            ontology.sync_properties_from_graph()
            # Named-graph exports carry triples only — author @prefix bindings are
            # serialization metadata the store never held. First rebind the
            # author's names from persisted sh:declare triples, then let implicit
            # binding cover any namespace that predates declaration support (it
            # skips already-declared stems), so prompt-context prefix advertising
            # survives a triple-store round trip with authorial names intact.
            ontology.graph.bind_declared_prefixes()
            ontology.graph.bind_implicit_namespaces(prefix_base=ontology.ontology_id)
            logger.debug(
                "Loaded ontology %s version %s from graph %s",
                onto_iri,
                ontology.version,
                graph_uri,
            )
            return ontology
    except Exception as exc:
        logger.warning("Error building ontology from %s: %s", graph_uri, exc)
    return None


LineageT = TypeVar("LineageT", bound=OntologyPropertiesWithLineage)


def dedupe_terminal_ontologies(all_ontologies: Sequence[LineageT]) -> list[LineageT]:
    """Keep the latest terminal version per ontology IRI.

    Works on anything carrying lineage metadata, so the same selection runs over
    materialized :class:`~ontocast.onto.ontology.Ontology` objects and over
    graph-less :class:`~ontocast.onto.ontology_header.OntologyHeader` records.

    Args:
        all_ontologies: Every stored version, across all IRIs. The parent-hash
            sweep needs the full set, not just candidate terminals.

    Returns:
        list: One entry per distinct IRI, of the same type that was passed in.
    """
    ontology_dict: dict[str, list[LineageT]] = defaultdict(list)
    for onto in all_ontologies:
        ontology_dict[onto.iri].append(onto)

    all_parent_hashes: set[str] = set()
    for onto in all_ontologies:
        for parent_hash in onto.parent_hashes:
            all_parent_hashes.add(parent_hash)

    ontologies: list[LineageT] = []
    for iri, versions in ontology_dict.items():
        if len(versions) == 1:
            ontologies.append(versions[0])
            continue

        terminal_versions = [
            v for v in versions if v.hash and v.hash not in all_parent_hashes
        ]
        if not terminal_versions:
            logger.warning(
                "No terminal ontologies found for %s, using all versions", iri
            )
            terminal_versions = versions

        try:
            versions_with_created = [
                (created_at, v)
                for v in terminal_versions
                if (created_at := v.created_at) is not None
            ]
            if versions_with_created:
                versions_with_created.sort(key=lambda pair: pair[0], reverse=True)
                ontologies.append(versions_with_created[0][1])
                continue

            versions_with_ver = [v for v in terminal_versions if v.version]
            if versions_with_ver:
                versions_with_ver.sort(key=lambda x: str(x.version), reverse=False)
                ontologies.append(versions_with_ver[-1])
            else:
                ontologies.append(terminal_versions[0])
        except Exception as exc:
            logger.warning("Could not select terminal ontology for %s: %s", iri, exc)
            ontologies.append(terminal_versions[0])

    return ontologies


def _parse_created_at(value: str) -> datetime | None:
    """Parse an xsd:dateTime lexical form, tolerating a trailing ``Z``."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("Unparseable dcterms:created value %r", value)
        return None


def headers_from_select_rows(
    rows: Iterable[dict[str, str]],
) -> list[OntologyHeader]:
    """Fold :data:`ONTOLOGY_HEADER_QUERY` rows into one header per named graph.

    ``?parent`` is multi-valued, so the query returns a small cross product; rows
    sharing a ``(?g, ?onto)`` pair describe one ontology version and are merged,
    accumulating ``parent_hashes``.

    Args:
        rows: SELECT bindings, each mapping variable name to lexical value.

    Returns:
        list[OntologyHeader]: One header per stored ontology version, ordered by
        graph URI for determinism.
    """
    by_graph: dict[tuple[str, str], OntologyHeader] = {}
    parents: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in rows:
        graph_uri = row.get("g")
        onto_subject = row.get("onto")
        if not graph_uri or not onto_subject:
            continue
        key = (graph_uri, onto_subject)

        parent = row.get("parent")
        if parent and parent.startswith(_PARENT_HASH_IRI_PREFIX):
            parent_hash = parent[len(_PARENT_HASH_IRI_PREFIX) :]
            if parent_hash and parent_hash not in parents[key]:
                parents[key].append(parent_hash)

        if key in by_graph:
            continue

        identifier = row.get("identifier") or ""
        onto_hash = (
            identifier[len(_HASH_IDENTIFIER_PREFIX) :]
            if identifier.startswith(_HASH_IDENTIFIER_PREFIX)
            else None
        )
        created = row.get("created")
        by_graph[key] = OntologyHeader(
            graph_uri=graph_uri,
            iri=ontology_iri_for_named_graph(graph_uri, onto_subject),
            version=row.get("version"),
            hash=onto_hash,
            created_at=_parse_created_at(created) if created else None,
        )

    for key, header in by_graph.items():
        header.parent_hashes = sorted(parents[key])

    return [by_graph[key] for key in sorted(by_graph)]
