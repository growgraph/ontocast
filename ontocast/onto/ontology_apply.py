"""Apply ontology update deltas onto catalog ontologies by namespace ownership.

``U → O*``: complement inserts and catalog deletes are partitioned by
subject/predicate namespace onto writable catalog IRIs, then applied
(delete-then-insert) onto each ontology's freshest terminal — or an explicit
in-run base override.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from pydantic import Field
from rdflib import Literal, URIRef

from ontocast.onto.constants import COMMON_PREFIXES, DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.iri_policy import normalize_namespace_iri, split_namespace_local
from ontocast.onto.model import BasePydanticModel
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.util import (
    RDFLIB_DEFAULT_NAMESPACE_URIS,
    is_rdflib_default_namespace,
)
from ontocast.tool.ontology_manager import OntologyManager

logger = logging.getLogger(__name__)


class OntologyDelta(BasePydanticModel):
    """Net ontology change relative to a prompt snapshot.

    ``inserts`` holds complement triples (``U \\ S``); ``deletes`` holds
    snapshot triples removed by GraphUpdate delete operations, destined for
    propagation onto catalog terminals.
    """

    inserts: RDFGraph = Field(default_factory=RDFGraph)
    deletes: RDFGraph = Field(default_factory=RDFGraph)

    def is_empty(self) -> bool:
        """True when the delta carries no insert and no delete triples."""
        return len(self.inserts) == 0 and len(self.deletes) == 0


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
    base_overrides: dict[str, Ontology] | None = None,
) -> dict[str, str]:
    """Map namespace stem → catalog ontology IRI for writable sources.

    ``base_overrides`` supplies in-run artifacts (not yet registered with the
    manager) whose namespaces must resolve, keyed by ontology IRI.
    """
    owners: dict[str, str] = {}
    for iri in writable_iris:
        if not iri or iri == NULL_ONTOLOGY.iri:
            continue
        ontology = (base_overrides or {}).get(iri)
        if ontology is None or ontology.is_null():
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


def partition_triples_by_namespace(
    triples: RDFGraph,
    *,
    writable_iris: list[str],
    ontology_manager: OntologyManager,
    base_overrides: dict[str, Ontology] | None = None,
) -> tuple[dict[str, RDFGraph], int]:
    """Partition delta triples onto writable catalog IRIs by namespace ownership.

    Works for both insert and delete deltas. Ownership prefers the subject URI
    namespace, then the predicate namespace.
    Returns ``(iri → graph, unattributed_triple_count)``.
    """
    owners = build_namespace_owner_map(
        ontology_manager, writable_iris, base_overrides=base_overrides
    )
    writable = set(writable_iris)
    buckets: dict[str, RDFGraph] = defaultdict(RDFGraph)
    unattributed = 0

    for prefix, namespace in triples.namespaces():
        if prefix:
            for bucket in buckets.values():
                bucket.bind(prefix, namespace)

    for s, p, o in triples:
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
                "Unattributed ontology delta triple: %s %s %s",
                s,
                p,
                o,
            )
            continue

        bucket = buckets[owner]
        for prefix, namespace in triples.namespaces():
            if prefix:
                bucket.bind(prefix, namespace)
        bucket.add((s, p, o))

    return dict(buckets), unattributed


def apply_partitioned_updates(
    partitioned_inserts: dict[str, RDFGraph],
    *,
    ontology_manager: OntologyManager,
    normalize_units_fn,
    tools,
    partitioned_deletes: dict[str, RDFGraph] | None = None,
    base_overrides: dict[str, Ontology] | None = None,
) -> tuple[list[Ontology], dict[str, int], list[GraphUpdate]]:
    """Apply each per-IRI delta (delete-then-insert) onto its catalog base.

    The base is the freshest catalog terminal for the IRI unless
    ``base_overrides`` supplies an in-run artifact (e.g. the map-stage output a
    consolidation delta must build on). ``normalize_units_fn`` is typically
    :func:`ontocast.agent.normalize_ontology.normalize_ontology_units`.

    Returns ``(artifacts, metrics, applied_updates)`` where ``applied_updates``
    are the GraphUpdates actually executed, for version-bump analysis.
    """
    deletes_by_iri = partitioned_deletes or {}
    artifacts: list[Ontology] = []
    applied_updates: list[GraphUpdate] = []
    metrics: dict[str, int] = {
        "apply_touched_iris": 0,
        "apply_skipped_missing_base": 0,
        "apply_insert_triples": 0,
        "apply_delete_triples": 0,
        "apply_deletes_no_match": 0,
    }
    for iri in sorted(set(partitioned_inserts) | set(deletes_by_iri)):
        insert_delta = partitioned_inserts.get(iri) or RDFGraph()
        delete_delta = deletes_by_iri.get(iri) or RDFGraph()
        if len(insert_delta) == 0 and len(delete_delta) == 0:
            continue
        metrics["apply_insert_triples"] += len(insert_delta)
        metrics["apply_delete_triples"] += len(delete_delta)
        base = (base_overrides or {}).get(iri)
        if base is None or base.is_null():
            base = ontology_manager.get_freshest_terminal_ontology_by_iri(iri)
        if base is None or base.is_null():
            logger.warning(
                "Cannot apply %s insert / %s delete triples: no catalog base for %s",
                len(insert_delta),
                len(delete_delta),
                iri,
            )
            metrics["apply_skipped_missing_base"] += 1
            continue
        if len(delete_delta) > 0:
            # A delete names a triple the unit's *snapshot* contained; the
            # apply target is the freshest *terminal*. When the two diverge
            # (a stale vector index, a terminal advanced by another run) the
            # DELETE DATA is a silent no-op — count it, or the divergence is
            # invisible in every artifact.
            no_match = sum(1 for triple in delete_delta if triple not in base.graph)
            if no_match:
                metrics["apply_deletes_no_match"] += no_match
                logger.warning(
                    "%d of %d delete triple(s) for %s are absent from the "
                    "catalog base — snapshot/terminal divergence (stale "
                    "vector index?); they will apply as no-ops",
                    no_match,
                    len(delete_delta),
                    iri,
                )
        units = []
        if len(insert_delta) > 0:
            units.append(
                ContentUnit(
                    text="",
                    index=0,
                    doc_iri="urn:ontocast:apply",
                    graph=insert_delta,
                    type=OutputType.ONTOLOGIES,
                )
            )
        result, applied, _prov = normalize_units_fn(
            units,
            tools,
            base_ontology=base,
            require_base=True,
            delete_graph=delete_delta if len(delete_delta) > 0 else None,
        )
        if not result.is_null() and len(result.graph) > 0:
            artifacts.append(result)
            applied_updates.extend(applied)
            metrics["apply_touched_iris"] += 1
    return artifacts, metrics, applied_updates
