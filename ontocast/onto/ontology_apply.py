"""Apply ontology update deltas onto catalog ontologies by namespace ownership.

``U → O*``: complement inserts are partitioned by subject/predicate namespace onto
writable catalog IRIs, then merged onto each ontology's freshest terminal.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from rdflib import Literal, URIRef

from ontocast.onto.constants import COMMON_PREFIXES, DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.iri_policy import normalize_namespace_iri, split_namespace_local
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import (
    RDFLIB_DEFAULT_NAMESPACE_URIS,
    is_rdflib_default_namespace,
)
from ontocast.tool.ontology_manager import OntologyManager

logger = logging.getLogger(__name__)

_STANDARD_NAMESPACE_STEMS: frozenset[str] = frozenset(
    {
        normalize_namespace_iri(uri.strip("<>"), context="auto").rstrip("/#")
        for uri in (
            *RDFLIB_DEFAULT_NAMESPACE_URIS,
            *(v.strip("<>") for v in COMMON_PREFIXES.values()),
            "https://schema.org/",
            "http://schema.org/",
            DEFAULT_IRI,
        )
    }
)


def complement_inserts(inserts: RDFGraph, snapshot_graph: RDFGraph) -> RDFGraph:
    """Return insert triples not already present in the prompt snapshot."""
    result = RDFGraph()
    snapshot_set = set(snapshot_graph)
    for prefix, namespace in inserts.namespaces():
        if prefix:
            result.bind(prefix, namespace)
    for triple in inserts:
        if triple not in snapshot_set:
            result.add(triple)
    return result


def _namespace_stem(uri: str) -> str | None:
    ns, _ = split_namespace_local(uri)
    if not ns:
        return None
    return ns.rstrip("/#")


def _is_standard_stem(stem: str | None) -> bool:
    if not stem:
        return True
    if stem in _STANDARD_NAMESPACE_STEMS:
        return True
    # Also treat common RDFLIB defaults via helper on full IRI forms
    for suffix in ("/", "#", ""):
        candidate = f"{stem}{suffix}"
        if is_rdflib_default_namespace(candidate):
            return True
    return False


def build_namespace_owner_map(
    ontology_manager: OntologyManager,
    writable_iris: list[str],
) -> dict[str, str]:
    """Map namespace stem → catalog ontology IRI for writable sources."""
    owners: dict[str, str] = {}
    for iri in writable_iris:
        if not iri or iri == NULL_ONTOLOGY.iri:
            continue
        ontology = ontology_manager.get_freshest_terminal_ontology_by_iri(iri)
        if ontology is None or ontology.is_null():
            logger.warning(
                "No catalog ontology for writable IRI %s; skipping ownership", iri
            )
            continue
        stems: set[str] = set()
        if ontology.iri:
            stems.add(ontology.iri.rstrip("/#"))
            stems.add(
                normalize_namespace_iri(ontology.iri, context="ontology").rstrip("/#")
            )
        if ontology.namespace:
            stems.add(ontology.namespace.rstrip("/#"))
        for _prefix, namespace_uri in ontology.graph.namespaces():
            ns = str(namespace_uri)
            if is_rdflib_default_namespace(ns) or ns in {
                str(v).strip("<>") for v in COMMON_PREFIXES.values()
            }:
                continue
            stem = ns.rstrip("/#")
            if not _is_standard_stem(stem):
                stems.add(stem)
        for stem in stems:
            if stem in owners and owners[stem] != iri:
                logger.debug(
                    "Namespace stem %s claimed by both %s and %s; keeping %s",
                    stem,
                    owners[stem],
                    iri,
                    owners[stem],
                )
                continue
            owners[stem] = iri
    return owners


def _owner_for_uri(uri: str, owners: dict[str, str]) -> str | None:
    stem = _namespace_stem(uri)
    if stem is None or _is_standard_stem(stem):
        return None
    return owners.get(stem)


def partition_inserts_by_namespace(
    inserts: RDFGraph,
    *,
    writable_iris: list[str],
    ontology_manager: OntologyManager,
) -> tuple[dict[str, RDFGraph], int]:
    """Partition insert triples onto writable catalog IRIs by namespace ownership.

    Ownership prefers the subject URI namespace, then the predicate namespace.
    Returns ``(iri → graph, unattributed_triple_count)``.
    """
    owners = build_namespace_owner_map(ontology_manager, writable_iris)
    writable = set(writable_iris)
    buckets: dict[str, RDFGraph] = defaultdict(RDFGraph)
    unattributed = 0

    for prefix, namespace in inserts.namespaces():
        if prefix:
            for bucket in buckets.values():
                bucket.bind(prefix, namespace)

    for s, p, o in inserts:
        owner: str | None = None
        if isinstance(s, URIRef):
            owner = _owner_for_uri(str(s), owners)
        if owner is None and isinstance(p, URIRef):
            owner = _owner_for_uri(str(p), owners)
        if owner is None and isinstance(o, URIRef) and not isinstance(o, Literal):
            # Object-only domain IRIs (e.g. typing to a new class under a domain NS)
            # still need a home when subject is a blank/cd node — rare for ontology
            # schema inserts; prefer subject/predicate first.
            candidate = _owner_for_uri(str(o), owners)
            if candidate is not None:
                owner = candidate

        if owner is None or owner not in writable:
            unattributed += 1
            logger.debug(
                "Unattributed ontology insert triple: %s %s %s",
                s,
                p,
                o,
            )
            continue

        bucket = buckets[owner]
        for prefix, namespace in inserts.namespaces():
            if prefix:
                bucket.bind(prefix, namespace)
        bucket.add((s, p, o))

    return dict(buckets), unattributed


def apply_partitioned_inserts(
    partitioned: dict[str, RDFGraph],
    *,
    ontology_manager: OntologyManager,
    normalize_units_fn,
    tools,
) -> tuple[list[Ontology], dict[str, int]]:
    """Merge each per-IRI insert graph onto the freshest catalog base.

    ``normalize_units_fn`` is typically
    :func:`ontocast.agent.normalize_ontology.normalize_ontology_units`.
    """
    artifacts: list[Ontology] = []
    metrics: dict[str, int] = {
        "apply_touched_iris": 0,
        "apply_skipped_missing_base": 0,
        "apply_insert_triples": 0,
    }
    for iri, delta in sorted(partitioned.items(), key=lambda item: item[0]):
        if len(delta) == 0:
            continue
        metrics["apply_insert_triples"] += len(delta)
        base = ontology_manager.get_freshest_terminal_ontology_by_iri(iri)
        if base is None or base.is_null():
            logger.warning(
                "Cannot apply %s insert triples: no catalog base for %s",
                len(delta),
                iri,
            )
            metrics["apply_skipped_missing_base"] += 1
            continue
        unit = ContentUnit(
            text="",
            index=0,
            doc_iri="urn:ontocast:apply",
            graph=delta,
            type=OutputType.ONTOLOGIES,
        )
        result, _applied, _prov = normalize_units_fn(
            [unit],
            tools,
            base_ontology=base,
            require_base=True,
        )
        if not result.is_null() and len(result.graph) > 0:
            artifacts.append(result)
            metrics["apply_touched_iris"] += 1
    return artifacts, metrics
