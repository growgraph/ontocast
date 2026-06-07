"""Shared helpers for triple store backends."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from rdflib import Graph
from rdflib.namespace import OWL, RDF

from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)


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
            onto_iri = str(onto_subj)
            if "#" in graph_uri:
                namespace, _ = split_namespace_local(graph_uri)
                base_iri = graph_uri
                if namespace is not None and namespace.endswith("#"):
                    base_iri = namespace[:-1]
                onto_iri = base_iri

            ontology = Ontology(graph=graph, iri=onto_iri)
            ontology.sync_properties_from_graph()
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


def dedupe_terminal_ontologies(all_ontologies: list[Ontology]) -> list[Ontology]:
    """Keep the latest terminal version per ontology IRI."""
    ontology_dict: dict[str, list[Ontology]] = defaultdict(list)
    for onto in all_ontologies:
        ontology_dict[onto.iri].append(onto)

    all_parent_hashes: set[str] = set()
    for onto in all_ontologies:
        for parent_hash in onto.parent_hashes:
            all_parent_hashes.add(parent_hash)

    ontologies: list[Ontology] = []
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
                v for v in terminal_versions if v.created_at is not None
            ]
            if versions_with_created:
                versions_with_created.sort(key=lambda x: x.created_at, reverse=True)
                ontologies.append(versions_with_created[0])
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
