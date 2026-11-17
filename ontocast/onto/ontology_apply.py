"""Apply ontology update deltas onto catalog ontologies by namespace ownership.

``U → O*``: complement inserts and catalog deletes are partitioned by
subject namespace onto writable catalog IRIs, then applied
(delete-then-insert) onto each ontology's freshest terminal — or an explicit
in-run base override.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from pydantic import Field
from rdflib import OWL, BNode, URIRef

from ontocast.onto.blank_structures import blank_structures, remove_with_structures
from ontocast.onto.constants import COMMON_PREFIXES
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.identity import is_owned_ontology
from ontocast.onto.iri_policy import normalize_namespace_iri, split_namespace_local
from ontocast.onto.model import BasePydanticModel
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.util import (
    is_rdflib_default_namespace,
    non_domain_namespaces,
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


# The same membership the prompt's "domain ontologies" clause uses, in stem
# form: this comparison is against normalized IRIs, so trailing / and # are
# stripped. Membership is defined once in ``non_domain_namespaces``.
_STANDARD_NAMESPACE_STEMS: frozenset[str] = frozenset(
    {
        normalize_namespace_iri(uri.strip("<>"), context="auto").rstrip("/#")
        for uri in non_domain_namespaces()
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

    An ontology owns the stems of its IRI and namespace. A namespace it only
    binds as a prefix is a weaker claim, honoured when no writable ontology
    owns that stem outright: an ontology binds the prefixes of the terms it
    refers to, and a statement about another ontology's term must not follow
    the prefix into the referring one. Between two claims of the same
    strength, the first writable IRI keeps the stem.

    ``base_overrides`` supplies in-run artifacts (not yet registered with the
    manager) whose namespaces must resolve, keyed by ontology IRI.
    """
    own: list[tuple[str, set[str]]] = []
    bound: list[tuple[str, set[str]]] = []
    for iri in writable_iris:
        if not iri or iri == NULL_ONTOLOGY.iri:
            continue
        ontology = (base_overrides or {}).get(iri)
        if ontology is None or ontology.is_null():
            ontology = ontology_manager.get_head_by_iri(iri)
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
        prefixes: set[str] = set()
        for _prefix, namespace_uri in ontology.graph.namespaces():
            ns = str(namespace_uri)
            if is_rdflib_default_namespace(ns) or ns in {
                str(v).strip("<>") for v in COMMON_PREFIXES.values()
            }:
                continue
            stem = ns.rstrip("/#")
            if not _is_standard_stem(stem):
                prefixes.add(stem)
        own.append((iri, stems))
        bound.append((iri, prefixes - stems))

    owners: dict[str, str] = {}
    for claims in (own, bound):
        for iri, stems in claims:
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
    current_domain: str,
    base_overrides: dict[str, Ontology] | None = None,
) -> tuple[dict[str, RDFGraph], int]:
    """Partition delta triples onto writable catalog IRIs by namespace ownership.

    Works for both insert and delete deltas. A triple belongs to the ontology
    that owns its subject. A named subject no writable ontology owns is never
    filed by its predicate or object: that would write an assertion about an
    individual, or an axiom about a term whose declaration went nowhere, into
    whichever ontology owns the property it uses.

    Blank nodes are filed as structures (:mod:`ontocast.onto.blank_structures`): a
    restriction or a list goes whole with the owner of the named subject that
    points at it, so ``_:r owl:onProperty ext:p`` never lands in ``ext``. A structure
    nothing points at (``[] a owl:AllDisjointClasses``) goes to the single
    owner of the named terms it uses; an ``owl:Axiom`` annotation goes with its
    ``owl:annotatedSource``. A structure whose anchors or terms have no single
    writable owner is unattributed, together with the statements pointing at it.

    An external ontology (outside ``current_domain``) among ``writable_iris``
    owns its namespaces, so no other ontology claims them through a bound
    prefix, but nothing is filed into it: what it would own is unattributed.
    Returns ``(iri → graph, unattributed_triple_count)``.
    """
    owners = build_namespace_owner_map(
        ontology_manager, writable_iris, base_overrides=base_overrides
    )
    writable = {iri for iri in writable_iris if is_owned_ontology(iri, current_domain)}
    buckets: dict[str, RDFGraph] = defaultdict(RDFGraph)
    unattributed = 0

    for prefix, namespace in triples.namespaces():
        if prefix:
            for bucket in buckets.values():
                bucket.bind(prefix, namespace)

    structure_owner: dict[tuple, str | None] = {}
    # Statements pointing at a structure no single owner takes: they go with it,
    # or each anchor's ontology would keep an edge to a node it does not hold.
    orphaned_inbound: set[tuple] = set()
    for structure in blank_structures(triples):
        sources = {
            o
            for _, p, o in structure.triples
            if p == OWL.annotatedSource and isinstance(o, URIRef)
        }
        named = structure.anchors or sources or structure.references
        candidates = {_owner_for_uri(str(node), owners) for node in named}
        if not structure.anchors:
            candidates.discard(None)
        owner = next(iter(candidates)) if len(candidates) == 1 else None
        for triple in structure.triples:
            structure_owner[triple] = owner
        if owner is None or owner not in writable:
            orphaned_inbound |= structure.inbound

    for s, p, o in triples:
        owner: str | None = None
        if (s, p, o) in orphaned_inbound:
            owner = None
        elif isinstance(s, URIRef):
            owner = _owner_for_uri(str(s), owners)
        else:
            owner = structure_owner.get((s, p, o))

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


def refile_fresh_statements(
    fresh: Ontology,
    *,
    writable_iris: list[str],
    ontology_manager: OntologyManager,
    current_domain: str,
    base_overrides: dict[str, Ontology] | None = None,
) -> tuple[RDFGraph, int]:
    """File a fresh ontology's statements by subject, as an update delta's are.

    A fresh render writes a whole new ontology, but a statement in it about
    another ontology's term belongs to that ontology. The header and every
    statement :func:`partition_triples_by_namespace` files to ``fresh`` stay in
    its graph; the rest leave it. Returns the statements another writable
    ontology owns, to be applied there, and how many statements no writable
    ontology owns (removed).
    """
    header = URIRef(fresh.iri)
    body = RDFGraph()
    for prefix, namespace in fresh.graph.namespaces():
        if prefix:
            body.bind(prefix, namespace)
    for triple in fresh.graph:
        if triple[0] != header:
            body.add(triple)
    writable = list(writable_iris)
    if fresh.iri not in writable:
        writable.append(fresh.iri)
    overrides = {**(base_overrides or {}), fresh.iri: fresh}
    buckets, unattributed = partition_triples_by_namespace(
        body,
        writable_iris=writable,
        ontology_manager=ontology_manager,
        current_domain=current_domain,
        base_overrides=overrides,
    )
    kept = buckets.pop(fresh.iri, RDFGraph())
    moved = RDFGraph()
    for prefix, namespace in fresh.graph.namespaces():
        if prefix:
            moved.bind(prefix, namespace)
    for graph in buckets.values():
        for triple in graph:
            moved.add(triple)
    for triple in list(body):
        if triple not in kept:
            fresh.graph.remove(triple)
    return moved, unattributed


def _keep_structures_whole(deletes: RDFGraph, base: RDFGraph) -> tuple[int, int]:
    """Make ``deletes`` remove a blank-node structure of ``base`` whole or not at all.

    A delete of every statement pointing at a structure takes the structure
    with it, or it would be left orphaned in the base. A delete of part of a
    structure that something still points at is dropped: half a restriction is
    not a restriction, and nothing in the delta says what should replace it.
    Edits ``deletes`` in place.

    Returns:
        ``(dropped, expanded)``: delete triples removed, and triples added.
    """
    if not any(isinstance(s, BNode) or isinstance(o, BNode) for s, _, o in deletes):
        return 0, 0
    dropped = expanded = 0
    for structure in blank_structures(base):
        internal = {triple for triple in structure.triples if triple in deletes}
        orphaned = bool(structure.inbound) and all(
            triple in deletes for triple in structure.inbound
        )
        if orphaned:
            for triple in structure.triples:
                if triple not in deletes:
                    deletes.add(triple)
                    expanded += 1
        elif internal:
            for triple in internal:
                deletes.remove(triple)
            dropped += len(internal)
    return dropped, expanded


def apply_partitioned_updates(
    partitioned_inserts: dict[str, RDFGraph],
    *,
    ontology_manager: OntologyManager,
    normalize_units_fn,
    tools,
    partitioned_deletes: dict[str, RDFGraph] | None = None,
    base_overrides: dict[str, Ontology] | None = None,
) -> tuple[list[Ontology], dict[str, int], dict[str, list[GraphUpdate]]]:
    """Apply each per-IRI delta (delete-then-insert) onto its catalog base.

    The base is the catalog head for the IRI unless ``base_overrides`` supplies
    an in-run artifact (e.g. the map-stage output a consolidation delta must
    build on). ``normalize_units_fn`` is typically
    :func:`ontocast.agent.normalize_ontology.normalize_ontology_units`.

    Triples whose subject is the ontology IRI itself are dropped before the
    apply, with the blank-node structures they point at, and counted as
    ``apply_header_triples_dropped``: the version header is written by code,
    never by a render. Deletes keep the base's blank-node structures whole
    (:func:`_keep_structures_whole`).

    Returns ``(artifacts, metrics, applied_updates_by_iri)`` where the updates
    are the GraphUpdates actually executed on each IRI, for its version label.
    """
    deletes_by_iri = partitioned_deletes or {}
    artifacts: list[Ontology] = []
    applied_updates: dict[str, list[GraphUpdate]] = {}
    metrics: dict[str, int] = {
        "apply_touched_iris": 0,
        "apply_skipped_missing_base": 0,
        "apply_insert_triples": 0,
        "apply_delete_triples": 0,
        "apply_deletes_no_match": 0,
        "apply_header_triples_dropped": 0,
        "apply_structure_deletes_dropped": 0,
        "apply_structure_deletes_expanded": 0,
    }
    for iri in sorted(set(partitioned_inserts) | set(deletes_by_iri)):
        insert_delta = partitioned_inserts.get(iri) or RDFGraph()
        delete_delta = deletes_by_iri.get(iri) or RDFGraph()
        for delta in (insert_delta, delete_delta):
            header_triples = list(delta.triples((URIRef(iri), None, None)))
            metrics["apply_header_triples_dropped"] += remove_with_structures(
                delta, header_triples
            )
        if len(insert_delta) == 0 and len(delete_delta) == 0:
            continue
        metrics["apply_insert_triples"] += len(insert_delta)
        metrics["apply_delete_triples"] += len(delete_delta)
        base = (base_overrides or {}).get(iri)
        if base is None or base.is_null():
            base = ontology_manager.get_head_by_iri(iri)
        if base is None or base.is_null():
            logger.warning(
                "Cannot apply %s insert / %s delete triples: no catalog base for %s",
                len(insert_delta),
                len(delete_delta),
                iri,
            )
            metrics["apply_skipped_missing_base"] += 1
            continue
        dropped, expanded = _keep_structures_whole(delete_delta, base.graph)
        metrics["apply_structure_deletes_dropped"] += dropped
        metrics["apply_structure_deletes_expanded"] += expanded
        if len(insert_delta) == 0 and len(delete_delta) == 0:
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
            if applied:
                applied_updates.setdefault(iri, []).extend(applied)
            metrics["apply_touched_iris"] += 1
    return artifacts, metrics, applied_updates
