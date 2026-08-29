"""Retrieves multi-ontology context patches from vector search."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from pydantic import Field, PrivateAttr
from rdflib import BNode, Namespace, URIRef
from rdflib.namespace import RDFS

from ontocast.config import (
    CrossQueryMergeMode,
    LexicalTriggerFusion,
    PatchRetrievalConfig,
    SymbolCaseMismatchPolicy,
    VectorStoreConfig,
)
from ontocast.onto.constants import COMMON_PREFIXES
from ontocast.onto.iri_policy import as_sparql_iriref
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_header import OntologyHeader
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import RDFLIB_DEFAULT_NAMESPACE_URIS
from ontocast.tool.onto import Tool
from ontocast.tool.representation_text import ROLE_PREDICATE
from ontocast.tool.sparql import (
    build_candidate_subgraph_query,
    filter_overbroad_namespace_map,
    select_relevant_ontologies,
)
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.tool.triple_manager.util import dedupe_terminal_ontologies
from ontocast.tool.vector_store.core import (
    GraphAtom,
    OntologySearchHit,
    OntologySearchHitsByChannel,
    VectorStoreManager,
)
from ontocast.tool.vector_store.diagnostics import (
    build_ontology_rank_diagnostics,
)
from ontocast.tool.vector_store.lexical_trigger import tokenize_for_lexical_match
from ontocast.tool.vector_store.query_signals import (
    CatalogSurfaceIndex,
    number_adjacent_tokens,
)
from ontocast.tool.vector_store.util import (
    normalized_core_neighborhood_weights,
    normalized_fusion_weights,
    rank_fuse_channel_hits,
)

logger = logging.getLogger(__name__)

_STRUCTURAL_REFERENCE_PREDICATES = frozenset({RDFS.subClassOf, RDFS.domain, RDFS.range})

# Cap on terms per SPARQL ``VALUES`` clause. Seed counts are already bounded by
# the retrieval config; this keeps a runaway list from becoming one huge query.
_MAX_VALUES_TERMS = 128


def _bind_common_vocab_prefixes(graph: RDFGraph) -> None:
    """Declare standard RDF/SKOS/DC prefixes when missing (better Turtle for entities)."""
    bound = {prefix for prefix, _ in graph.namespaces() if prefix}
    for prefix, uri_wrapped in COMMON_PREFIXES.items():
        if prefix in bound:
            continue
        graph.bind(prefix, Namespace(uri_wrapped.strip("<>")))


def _source_iris_from_atoms(atoms: Iterable[GraphAtom]) -> list[str]:
    return sorted({atom.ontology_iri for atom in atoms if atom.ontology_iri})


def _is_ontology_declaration_atom(atom: GraphAtom) -> bool:
    """True when the atom focal IRI is the ontology header node (not an expansion seed)."""
    return bool(atom.ontology_iri and atom.iri == atom.ontology_iri)


def _ranked_entity_weights(
    atoms: list[GraphAtom],
) -> tuple[list[str], dict[str, float], dict[str, str | None]]:
    """Collapse atom scores to entity-level ranking, relevance weights, and roles."""
    best_score_by_iri: dict[str, float] = {}
    entity_roles: dict[str, str | None] = {}
    for atom in atoms:
        iri = atom.iri
        if not iri or _is_ontology_declaration_atom(atom):
            continue
        score = float(atom.score or 0.0)
        previous = best_score_by_iri.get(iri)
        if previous is None or score > previous:
            best_score_by_iri[iri] = score
            entity_roles[iri] = atom.entity_role
    ranked = sorted(
        best_score_by_iri.keys(),
        key=lambda iri: (-best_score_by_iri[iri], iri),
    )
    return ranked, best_score_by_iri, entity_roles


def _class_ancestors(graph: RDFGraph, cls: URIRef, *, max_depth: int) -> set[URIRef]:
    """Walk ``rdfs:subClassOf`` upward from ``cls``, bounded and cycle-safe."""
    seen: set[URIRef] = set()
    frontier = {cls}
    for _ in range(max(max_depth, 0)):
        nxt: set[URIRef] = set()
        for node in frontier:
            for parent in graph.objects(node, RDFS.subClassOf):
                if isinstance(parent, URIRef) and parent not in seen:
                    seen.add(parent)
                    nxt.add(parent)
        if not nxt:
            break
        frontier = nxt
    return seen


def _closure_floor_score(seed_relevance: dict[str, float]) -> float:
    """Score for schema-closure additions: strictly below every real seed.

    The closure fills structural gaps and must not displace a term the text
    actually matched when the induced-subgraph triple budget binds.
    ``floor * 0.5`` only achieves that for positive floors — at ``0.0`` it
    ties with the weakest seed (leaving truncation order arbitrary) and a
    negative floor would *raise* closure above it — so non-positive floors
    get a strictly smaller sentinel instead.
    """
    floor_score = min(seed_relevance.values(), default=0.0)
    if floor_score > 0.0:
        return floor_score * 0.5
    return floor_score - 1.0


def _schema_closure_entities(
    graph: RDFGraph,
    seed_uris: Iterable[str],
    *,
    max_entities: int,
    ancestor_depth: int,
    seed_relevance: dict[str, float] | None = None,
) -> dict[str, str]:
    """Terms reachable from the seeds by ``rdfs:domain`` / ``rdfs:range``.

    Vector retrieval ranks terms by how a text chunk reads, which favours noun
    phrases: a class scores against prose far more readily than the property
    that connects it. Admitting a class without any property whose domain or
    range mentions it hands the renderer nouns it cannot link, and the renderer
    then improvises a predicate from outside the catalog. This closes that gap
    deterministically -- no embeddings, no domain knowledge, no LLM call:

    - for an admitted class, the properties declaring it (or one of its
      ancestors) as domain or range;
    - for an admitted property, the classes it declares as domain and range.

    Args:
        graph: Merged ontology context to read schema axioms from.
        seed_uris: IRIs already admitted by retrieval.
        max_entities: Hard cap on added IRIs (0 disables the closure).
        ancestor_depth: How far to walk ``rdfs:subClassOf`` when matching a
            property's declared domain/range against an admitted class.
        seed_relevance: Retrieval score per seed IRI, used to rank additions.
            Without it the cap truncates alphabetically, which is close to
            random: ``domain#A…`` displaces ``observation#hasQuantityResult``
            for no reason but the letter it starts with.

    Returns:
        dict: ``{iri: role}`` for terms to add, excluding the seeds themselves.
    """
    if max_entities <= 0:
        return {}
    seeds = {URIRef(uri) for uri in seed_uris if uri}
    if not seeds:
        return {}
    scores = seed_relevance or {}

    # One pass over the schema axioms; ontologies are small enough that
    # indexing them beats re-scanning per seed.
    domain_range: dict[URIRef, set[URIRef]] = defaultdict(set)
    for predicate in (RDFS.domain, RDFS.range):
        for prop, target in graph.subject_objects(predicate):
            if isinstance(prop, URIRef) and isinstance(target, URIRef):
                domain_range[prop].add(target)

    # Best score of any seed that justifies each addition, so the cap keeps
    # what the text actually pointed at.
    induced_by: dict[URIRef, float] = {}

    def _offer(term: URIRef, score: float) -> None:
        if term in seeds:
            return
        if score > induced_by.get(term, float("-inf")):
            induced_by[term] = score

    ancestors_of: dict[URIRef, set[URIRef]] = {}
    for seed in seeds:
        score = float(scores.get(str(seed), 0.0))
        # Property seed -> its declared domain and range classes.
        for target in domain_range.get(seed, ()):
            _offer(target, score)
        ancestors_of[seed] = {seed} | _class_ancestors(
            graph, seed, max_depth=ancestor_depth
        )

    # Class seed -> properties whose domain/range names it or an ancestor.
    for prop, targets in domain_range.items():
        for seed, family in ancestors_of.items():
            if targets & family:
                _offer(prop, float(scores.get(str(seed), 0.0)))

    property_iris = {str(prop) for prop in domain_range}
    ranked = sorted(
        induced_by.items(),
        # Properties first at equal score: they are the scarce half of the
        # snapshot and the half a class cannot be linked without.
        key=lambda item: (
            -item[1],
            0 if str(item[0]) in property_iris else 1,
            str(item[0]),
        ),
    )
    return {
        str(term): ("predicate" if str(term) in property_iris else "resource")
        for term, _ in ranked[:max_entities]
    }


def _drop_module_contribution(graph: RDFGraph, module_graph: RDFGraph) -> None:
    """Remove what the snapshot already holds about a module's own subjects.

    The induced subgraph normally took a slice of the module before the closure
    decided to include it whole. Merging the two would be harmless for IRI
    subjects — RDF is a set — but every blank node in the module (OWL
    restriction shells, SHACL prefix declarations) gets fresh identity on
    merge, so the shell would appear twice: once from the slice, once from the
    module. Dropping the slice first keeps the closure idempotent.
    """
    module_subjects = {
        subject for subject in module_graph.subjects() if isinstance(subject, URIRef)
    }
    if not module_subjects:
        return

    stale: set = set()
    frontier = [s for s in graph.subjects() if s in module_subjects]
    while frontier:
        subject = frontier.pop()
        if subject in stale:
            continue
        stale.add(subject)
        for _, _, obj in graph.triples((subject, None, None)):
            # Blank nodes have no identity of their own: they are only
            # reachable through the subject being replaced.
            if isinstance(obj, BNode) and obj not in stale:
                frontier.append(obj)

    for subject in stale:
        graph.remove((subject, None, None))


def _filter_hits_by_relative_floor(
    hits: list[OntologySearchHit],
    *,
    score_ratio: float,
    min_query_best_score: float,
) -> list[OntologySearchHit]:
    """Relative score gating within one channel/query hit list.

    The floor is expressed as a distance below the best score rather than as
    ``best * score_ratio``. The multiplicative form is equivalent for positive scores but
    inverts once ``best`` is negative, which Qdrant cosine can return: at ``best = -0.2``
    it puts the floor at ``-0.16``, *above* the best hit, so the channel returns nothing.
    A ratio of 0 also has to short-circuit — otherwise the documented "0 disables" default
    computes a floor of exactly 0.0 and silently drops every negative-scoring hit.
    """
    if score_ratio < 0.0 or score_ratio > 1.0:
        raise ValueError("score_ratio must be in [0, 1]")
    if not hits:
        return []
    best = max(h.score for h in hits)
    if min_query_best_score > 0.0 and best < min_query_best_score:
        return []
    if score_ratio <= 0.0:
        return list(hits)
    floor = best - ((1.0 - score_ratio) * abs(best))
    return [hit for hit in hits if hit.score >= floor]


def _normalize_relevance_scores(atoms: list[GraphAtom]) -> list[GraphAtom]:
    """Scale atom scores to [0, 1] for MMR relevance term."""
    if not atoms:
        return []
    max_score = max(float(atom.score or 0.0) for atom in atoms)
    if max_score <= 0.0:
        return atoms
    return [
        atom.model_copy(update={"score": float(atom.score or 0.0) / max_score})
        for atom in atoms
    ]


def _best_hit_by_entity_iri(
    hits: list[OntologySearchHit],
) -> dict[str, OntologySearchHit]:
    best: dict[str, OntologySearchHit] = {}
    for hit in hits:
        iri = hit.atom.iri
        if not iri:
            continue
        prev = best.get(iri)
        if prev is None or hit.score > prev.score:
            best[iri] = hit
    return best


def _merge_hits_across_queries_max_score(
    collected: list[OntologySearchHit],
) -> list[OntologySearchHit]:
    best_by_iri = _best_hit_by_entity_iri(collected)
    return sorted(
        best_by_iri.values(),
        key=lambda hit: (hit.score, hit.atom.iri or ""),
        reverse=True,
    )


def _merge_hits_across_queries_sum_score(
    collected: list[OntologySearchHit],
) -> list[OntologySearchHit]:
    """Sum per-window fused scores per entity IRI, so corroboration counts.

    Under max-score an entity ranked second in eight windows loses to one ranked first in
    a single window, discarding the agreement between windows. Summing rewards a term the
    whole passage keeps pointing at. Scores here are unbounded in the window count, so
    absolute thresholds must not be applied to them (see ``_filter_and_merge_patch_hits``,
    which gates on per-window scores instead).
    """
    total_by_iri: dict[str, float] = defaultdict(float)
    best_by_iri: dict[str, OntologySearchHit] = {}
    for hit in collected:
        iri = hit.atom.iri
        if not iri:
            continue
        total_by_iri[iri] += hit.score
        previous = best_by_iri.get(iri)
        if previous is None or hit.score > previous.score:
            best_by_iri[iri] = hit

    merged: list[OntologySearchHit] = []
    for iri, total in total_by_iri.items():
        atom = best_by_iri[iri].atom.model_copy(update={"score": total})
        merged.append(OntologySearchHit(atom=atom, score=total))
    return sorted(
        merged,
        key=lambda hit: (hit.score, hit.atom.iri or ""),
        reverse=True,
    )


def _select_hits_round_robin_by_ontology(
    ranked_hits: list[OntologySearchHit],
    *,
    per_ontology_seed_quota: int,
    max_atoms: int,
    per_ontology_atom_floor: int = 0,
    per_role_atom_floor: int = 0,
) -> list[OntologySearchHit]:
    """Fair multi-ontology fill from a score-ranked unique-entity list.

    Three optional mechanisms:

    - ``per_role_atom_floor`` is a *reserve* for predicate-role atoms, taken
      first in global score order. Prose reads as noun phrases, so a class
      out-scores the property that connects it in a shared ranking, and the
      predicates carrying the graph structure are the ones squeezed out.
    - ``per_ontology_atom_floor`` is a *reserve*: before any global fill,
      each contributing ontology is guaranteed min(floor, its candidates)
      slots, allocated round-robin so small modules cannot be starved by
      one dominant ontology's seed volume.
    - ``per_ontology_seed_quota`` is a *ceiling*: round-robin taking at most
      that many seeds per ontology before the leftover fill.

    Remaining slots under ``max_atoms`` fill from leftover hits in global
    score order. ``<= 0`` disables each mechanism; ``max_atoms <= 0`` means
    no total cap.

    Ontologies are visited best-scoring first. Visiting them in IRI order instead made
    allocation alphabetical whenever the cap bound before every ontology was served —
    which is the common case, since the cap does not grow with catalog size.
    """
    if not ranked_hits:
        return []
    limit = len(ranked_hits) if max_atoms <= 0 else min(max_atoms, len(ranked_hits))
    if (
        per_ontology_seed_quota <= 0
        and per_ontology_atom_floor <= 0
        and per_role_atom_floor <= 0
    ):
        return ranked_hits[:limit]

    by_ontology: dict[str, list[OntologySearchHit]] = defaultdict(list)
    for hit in ranked_hits:
        onto = hit.atom.ontology_iri or ""
        by_ontology[onto].append(hit)

    queues: dict[str, list[OntologySearchHit]] = {
        onto: list(hits) for onto, hits in by_ontology.items()
    }
    # ``ranked_hits`` is score-descending, so first-seen order is best-score-first.
    ontology_order: list[str] = list(by_ontology.keys())
    taken_count: dict[str, int] = defaultdict(int)
    selected: list[OntologySearchHit] = []
    selected_iris: set[str] = set()

    if per_role_atom_floor > 0:
        for hit in ranked_hits:
            if len(selected) >= min(limit, per_role_atom_floor):
                break
            iri = hit.atom.iri
            if not iri or iri in selected_iris:
                continue
            if hit.atom.entity_role != ROLE_PREDICATE:
                continue
            selected.append(hit)
            selected_iris.add(iri)
            taken_count[hit.atom.ontology_iri or ""] += 1
        # Reserved hits must not be handed out twice by the ontology passes.
        for onto, queue in queues.items():
            queues[onto] = [h for h in queue if h.atom.iri not in selected_iris]

    def _round_robin_fill(per_ontology_cap: int) -> None:
        progressed = True
        while len(selected) < limit and progressed:
            progressed = False
            for onto in ontology_order:
                if len(selected) >= limit:
                    break
                if taken_count[onto] >= per_ontology_cap:
                    continue
                queue = queues[onto]
                while queue:
                    hit = queue.pop(0)
                    iri = hit.atom.iri
                    if not iri or iri in selected_iris:
                        continue
                    selected.append(hit)
                    selected_iris.add(iri)
                    taken_count[onto] += 1
                    progressed = True
                    break

    if per_ontology_atom_floor > 0:
        _round_robin_fill(per_ontology_atom_floor)
    if per_ontology_seed_quota > 0:
        _round_robin_fill(max(per_ontology_seed_quota, per_ontology_atom_floor))

    if len(selected) < limit:
        for hit in ranked_hits:
            if len(selected) >= limit:
                break
            iri = hit.atom.iri
            if not iri or iri in selected_iris:
                continue
            selected.append(hit)
            selected_iris.add(iri)

    return selected


def _merge_hits_across_queries_hybrid(
    collected: list[OntologySearchHit],
    *,
    max_atoms_tier1: int,
    per_ontology_seed_quota: int,
    min_entity_score: float,
    max_atoms_total: int,
) -> list[OntologySearchHit]:
    """Tier-1 strong global seeds, tier-2 per-ontology coverage."""
    best_by_iri = _best_hit_by_entity_iri(collected)
    if not best_by_iri:
        return []

    tier1_candidates = sorted(
        best_by_iri.values(),
        key=lambda hit: (hit.score, hit.atom.iri or ""),
        reverse=True,
    )
    tier1_limit = len(tier1_candidates) if max_atoms_tier1 <= 0 else max_atoms_tier1
    tier1 = tier1_candidates[:tier1_limit]
    selected_iris = {hit.atom.iri for hit in tier1 if hit.atom.iri}

    by_ontology: dict[str, list[OntologySearchHit]] = defaultdict(list)
    for hit in best_by_iri.values():
        onto_iri = hit.atom.ontology_iri
        entity_iri = hit.atom.iri
        if not onto_iri or not entity_iri or entity_iri in selected_iris:
            continue
        if hit.score >= min_entity_score:
            by_ontology[onto_iri].append(hit)

    tier2: list[OntologySearchHit] = []
    quota = per_ontology_seed_quota if per_ontology_seed_quota > 0 else 9999
    for onto_iri in sorted(by_ontology.keys()):
        candidates = sorted(
            by_ontology[onto_iri],
            key=lambda hit: (hit.score, hit.atom.iri or ""),
            reverse=True,
        )
        added = 0
        for hit in candidates:
            if hit.atom.iri in selected_iris:
                continue
            tier2.append(hit)
            selected_iris.add(hit.atom.iri)
            added += 1
            if added >= quota:
                break

    merged = tier1 + tier2
    if max_atoms_total > 0:
        merged = merged[:max_atoms_total]
    return merged


def _merge_hits_across_queries(
    collected: list[OntologySearchHit],
    *,
    merge_mode: CrossQueryMergeMode,
    max_atoms_tier1: int,
    per_ontology_seed_quota: int,
    min_entity_score: float,
    max_atoms_total: int,
) -> list[OntologySearchHit]:
    if merge_mode in (CrossQueryMergeMode.MAX_SCORE, CrossQueryMergeMode.SUM_SCORE):
        merged = (
            _merge_hits_across_queries_max_score(collected)
            if merge_mode == CrossQueryMergeMode.MAX_SCORE
            else _merge_hits_across_queries_sum_score(collected)
        )
        if max_atoms_total > 0:
            return _select_hits_round_robin_by_ontology(
                merged,
                per_ontology_seed_quota=per_ontology_seed_quota,
                max_atoms=max_atoms_total,
            )
        return merged
    return _merge_hits_across_queries_hybrid(
        collected,
        max_atoms_tier1=max_atoms_tier1,
        per_ontology_seed_quota=per_ontology_seed_quota,
        min_entity_score=min_entity_score,
        max_atoms_total=max_atoms_total,
    )


def _select_atoms_round_robin_by_ontology(
    ranked_atoms: list[GraphAtom],
    *,
    per_ontology_seed_quota: int,
    max_atoms: int,
    per_ontology_atom_floor: int = 0,
    per_role_atom_floor: int = 0,
) -> list[GraphAtom]:
    """GraphAtom variant of :func:`_select_hits_round_robin_by_ontology`."""
    as_hits = [
        OntologySearchHit(atom=atom, score=float(atom.score or 0.0))
        for atom in ranked_atoms
    ]
    selected = _select_hits_round_robin_by_ontology(
        as_hits,
        per_ontology_seed_quota=per_ontology_seed_quota,
        max_atoms=max_atoms,
        per_ontology_atom_floor=per_ontology_atom_floor,
        per_role_atom_floor=per_role_atom_floor,
    )
    return [hit.atom for hit in selected]


def _filter_and_merge_patch_hits(
    hits_by_query: list[OntologySearchHitsByChannel],
    *,
    store_config: VectorStoreConfig,
    patch_config: PatchRetrievalConfig,
    per_query_core_score_ratio: float,
    per_query_neighborhood_score_ratio: float,
    per_query_bm25_score_ratio: float,
    min_core_query_best_score: float,
    min_neighborhood_query_best_score: float,
    min_bm25_query_best_score: float,
    min_merged_max_score: float,
    max_atoms_total: int = 0,
) -> list[GraphAtom]:
    """Filter each channel per query, then merge across queries."""
    cw, nw, bw = normalized_fusion_weights(store_config)
    collected: list[OntologySearchHit] = []
    for query_hits in hits_by_query:
        filtered_core = _filter_hits_by_relative_floor(
            query_hits.core_hits,
            score_ratio=per_query_core_score_ratio,
            min_query_best_score=min_core_query_best_score,
        )
        filtered_neighborhood = _filter_hits_by_relative_floor(
            query_hits.neighborhood_hits,
            score_ratio=per_query_neighborhood_score_ratio,
            min_query_best_score=min_neighborhood_query_best_score,
        )
        filtered_bm25 = _filter_hits_by_relative_floor(
            query_hits.bm25_hits,
            score_ratio=per_query_bm25_score_ratio,
            min_query_best_score=min_bm25_query_best_score,
        )
        collected.extend(
            rank_fuse_channel_hits(
                filtered_core,
                filtered_neighborhood,
                filtered_bm25,
                core_weight=cw,
                neighborhood_weight=nw,
                bm25_weight=bw,
                limit=max(
                    len(filtered_core)
                    + len(filtered_neighborhood)
                    + len(filtered_bm25),
                    1,
                ),
            )
        )

    if not collected:
        return []

    # Gate on the best *per-window* fused score, before the cross-window merge. Under
    # max-score merging the merged top score is that same number, so this is unchanged;
    # under sum-score merging the merged total grows with window count and an absolute
    # threshold against it would be meaningless.
    best_window_score = max(hit.score for hit in collected)
    if min_merged_max_score > 0.0 and best_window_score < min_merged_max_score:
        return []

    merged_hits = _merge_hits_across_queries(
        collected,
        merge_mode=patch_config.cross_query_merge_mode,
        max_atoms_tier1=patch_config.max_atoms_tier1,
        per_ontology_seed_quota=patch_config.per_ontology_seed_quota,
        min_entity_score=patch_config.min_entity_score,
        max_atoms_total=max_atoms_total,
    )
    if not merged_hits:
        return []

    out: list[GraphAtom] = []
    for hit in merged_hits:
        atom = hit.atom.model_copy(update={"score": hit.score})
        out.append(atom)
    return out


def _ontology_iri_for_entity(
    entity_iri: str,
    ontologies: list[Ontology],
) -> str | None:
    """Resolve which catalog ontology document owns ``entity_iri``."""
    ref = URIRef(entity_iri)
    for ontology in ontologies:
        namespace = (ontology.namespace or ontology.iri or "").rstrip("#/")
        if not namespace:
            continue
        if (
            entity_iri == namespace
            or entity_iri.startswith(f"{namespace}#")
            or entity_iri.startswith(f"{namespace}/")
        ):
            return ontology.iri
    for ontology in ontologies:
        graph = ontology.graph
        if any(graph.triples((ref, None, None))):
            return ontology.iri
    return None


def _expand_ontology_iris_by_reference(
    entity_uris: list[str],
    hit_ontology_iris: list[str],
    ontologies: list[Ontology],
) -> list[str]:
    """Include ontologies referenced by seed subClassOf/domain/range axioms."""
    expanded = set(hit_ontology_iris)
    seed_refs = {URIRef(uri) for uri in entity_uris if uri}
    referenced_iris: set[str] = set()

    for ontology in ontologies:
        graph = ontology.graph
        for seed in seed_refs:
            for _, pred, obj in graph.triples((seed, None, None)):
                if pred in _STRUCTURAL_REFERENCE_PREDICATES and isinstance(obj, URIRef):
                    referenced_iris.add(str(obj))
            for subj, pred, _ in graph.triples((None, None, seed)):
                if pred in _STRUCTURAL_REFERENCE_PREDICATES and isinstance(
                    subj, URIRef
                ):
                    referenced_iris.add(str(subj))

    for ref_iri in referenced_iris:
        owner = _ontology_iri_for_entity(ref_iri, ontologies)
        if owner:
            expanded.add(owner)

    return sorted(expanded)


def _namespace_candidates(headers: Iterable[OntologyHeader]) -> list[tuple[str, str]]:
    """Return ``(namespace, ontology_iri)`` pairs, longest namespace first.

    Sorting by descending length makes nested namespaces (``…/domain`` versus
    ``…/domain/sub``) resolve to the most specific owner deterministically.
    """
    candidates = [
        (namespace.rstrip("#/"), header.iri)
        for header in headers
        if (namespace := header.namespace or header.iri)
    ]
    return sorted(
        (pair for pair in candidates if pair[0]),
        key=lambda pair: (-len(pair[0]), pair[0]),
    )


def _owner_by_namespace(ref_iri: str, candidates: list[tuple[str, str]]) -> str | None:
    """Resolve the ontology whose namespace encloses ``ref_iri``, if any."""
    for namespace, ontology_iri in candidates:
        if (
            ref_iri == namespace
            or ref_iri.startswith(f"{namespace}#")
            or ref_iri.startswith(f"{namespace}/")
        ):
            return ontology_iri
    return None


def _unresolved_by_namespace(
    referenced_iris: Iterable[str],
    headers: list[OntologyHeader],
) -> list[str]:
    """IRIs that no catalog namespace encloses, needing a declaring-graph lookup."""
    candidates = _namespace_candidates(headers)
    return sorted(
        ref for ref in referenced_iris if _owner_by_namespace(ref, candidates) is None
    )


def _resolve_reference_owners(
    referenced_iris: Iterable[str],
    headers: list[OntologyHeader],
    owners_by_ref: dict[str, list[str]],
) -> set[str]:
    """Map referenced IRIs to the ontologies that own them.

    Namespace containment is tried first and is what catches *dangling*
    references -- an IRI inside ontology X's namespace that X declares no triples
    about. Only IRIs no namespace encloses fall back to the ontology whose named
    graph declares them.

    Args:
        referenced_iris: IRIs reachable from the seeds via structural predicates.
        headers: Catalog headers supplying namespaces and IRIs.
        owners_by_ref: Declaring ontology IRIs per reference, from the second hop.

    Returns:
        set[str]: Owning ontology IRIs.
    """
    candidates = _namespace_candidates(headers)
    owners: set[str] = set()
    for ref_iri in referenced_iris:
        owner = _owner_by_namespace(ref_iri, candidates)
        if owner is None:
            declaring = owners_by_ref.get(ref_iri)
            # Parity with the in-Python path, which attributes one owner per
            # reference; smallest IRI keeps that choice deterministic.
            owner = min(declaring) if declaring else None
        if owner:
            owners.add(owner)
    return owners


def _chunked(values: list[str], size: int) -> list[list[str]]:
    """Split ``values`` into chunks of at most ``size`` items."""
    return [values[start : start + size] for start in range(0, len(values), size)]


def _seed_reference_query(seed_irefs: list[str]) -> str:
    """Build the hop-1 SELECT: IRIs the seeds reach via structural predicates."""
    predicates = " ".join(
        f"<{predicate}>" for predicate in sorted(_STRUCTURAL_REFERENCE_PREDICATES)
    )
    seeds = " ".join(seed_irefs)
    return f"""
SELECT DISTINCT ?g ?ref WHERE {{
  VALUES ?seed {{ {seeds} }}
  VALUES ?p {{ {predicates} }}
  GRAPH ?g {{ {{ ?seed ?p ?ref }} UNION {{ ?ref ?p ?seed }} }}
  FILTER(isIRI(?ref))
}}
"""


def _declaring_graph_query(ref_irefs: list[str]) -> str:
    """Build the hop-2 SELECT: which named graph declares each reference.

    ``?ref`` appears as subject only, matching the in-Python fallback's
    ``graph.triples((ref, None, None))``; matching both directions would silently
    widen expansion.
    """
    refs = " ".join(ref_irefs)
    return f"""
SELECT DISTINCT ?g ?ref WHERE {{
  VALUES ?ref {{ {refs} }}
  GRAPH ?g {{ ?ref ?p ?o }}
}}
"""


def _sparql_irefs(iris: Iterable[str]) -> list[str]:
    """Render IRIs for SPARQL ``VALUES``, dropping (and logging) unusable ones."""
    irefs: list[str] = []
    for iri in iris:
        iref = as_sparql_iriref(iri)
        if iref is None:
            logger.warning("Skipping IRI that cannot be a SPARQL IRIREF: %r", iri)
            continue
        irefs.append(iref)
    return irefs


async def _aselect_union(
    triple_store_manager: TripleStoreManager,
    queries: list[str],
) -> list[dict[str, str]]:
    """Run SELECTs concurrently and concatenate their rows."""
    if not queries:
        return []
    results = await asyncio.gather(
        *(triple_store_manager.aselect(query) for query in queries)
    )
    return [row for rows in results for row in rows]


async def _aexpand_ontology_iris_by_reference(
    triple_store_manager: TripleStoreManager,
    entity_uris: list[str],
    hit_ontology_iris: list[str],
) -> tuple[list[str], dict[str, int | str]]:
    """Expand the ontology filter using targeted SELECTs instead of the catalog.

    Answers the same question as :func:`_expand_ontology_iris_by_reference` --
    which ontologies do the seeds reach through ``rdfs:subClassOf`` / ``domain`` /
    ``range`` -- without materializing any ontology graph.

    Args:
        triple_store_manager: A backend whose ``supports_sparql_select()`` is true.
        entity_uris: Seed entity IRIs from the merged vector hits.
        hit_ontology_iris: Ontology IRIs the hits themselves came from.

    Returns:
        tuple: ``(ontology_iris, metrics)``, where ``ontology_iris`` is the sorted
        union of hit and referenced ontologies.
    """
    headers = await triple_store_manager.afetch_ontology_catalog()
    terminal_headers = dedupe_terminal_ontologies(headers)
    # Both hops see every stored version; restricting to terminal graphs keeps
    # parity with the fallback, which iterates the deduped catalog.
    terminal_graph_uris = {header.graph_uri for header in terminal_headers}

    seed_irefs = _sparql_irefs(entity_uris)
    seed_queries = [
        _seed_reference_query(chunk)
        for chunk in _chunked(seed_irefs, _MAX_VALUES_TERMS)
    ]
    referenced_iris = {
        ref
        for row in await _aselect_union(triple_store_manager, seed_queries)
        if row.get("g") in terminal_graph_uris and (ref := row.get("ref"))
    }

    owners_by_ref: dict[str, list[str]] = defaultdict(list)
    unresolved = _unresolved_by_namespace(referenced_iris, terminal_headers)
    declaring_queries: list[str] = []
    if unresolved:
        graph_uri_to_iri = {h.graph_uri: h.iri for h in terminal_headers}
        declaring_queries = [
            _declaring_graph_query(chunk)
            for chunk in _chunked(_sparql_irefs(unresolved), _MAX_VALUES_TERMS)
        ]
        for row in await _aselect_union(triple_store_manager, declaring_queries):
            owner = graph_uri_to_iri.get(row.get("g", ""))
            ref = row.get("ref")
            if owner and ref and owner not in owners_by_ref[ref]:
                owners_by_ref[ref].append(owner)

    expanded = set(hit_ontology_iris)
    expanded |= _resolve_reference_owners(
        referenced_iris, terminal_headers, owners_by_ref
    )
    metrics: dict[str, int | str] = {
        "catalog_access_mode": "sparql",
        "catalog_select_queries": 1 + len(seed_queries) + len(declaring_queries),
        "catalog_graphs_fetched": 0,
    }
    return sorted(expanded), metrics


async def _aexpand_ontology_iris(
    triple_store_manager: TripleStoreManager,
    entity_uris: list[str],
    hit_ontology_iris: list[str],
) -> tuple[list[str], list[Ontology] | None, dict[str, int | str]]:
    """Expand the ontology filter, preferring SPARQL over a full catalog fetch.

    Returns:
        tuple: ``(ontology_iris, catalog, metrics)``. ``catalog`` is the
        materialized catalog when one had to be fetched, so the caller can hand it
        to the induced-subgraph step instead of paying for it twice; it is ``None``
        on the SPARQL path, where only the needed graphs are fetched later.
    """
    if triple_store_manager.supports_sparql_select():
        try:
            ontology_iris, metrics = await _aexpand_ontology_iris_by_reference(
                triple_store_manager, entity_uris, hit_ontology_iris
            )
            return ontology_iris, None, metrics
        except Exception as exc:
            # Slow retrieval beats a failed document; the metric makes a
            # persistently degraded backend visible rather than merely sluggish.
            logger.warning(
                "SPARQL ontology expansion failed (%s); falling back to full catalog",
                exc,
            )

    catalog = await triple_store_manager.afetch_ontologies()
    ontology_iris = _expand_ontology_iris_by_reference(
        entity_uris, hit_ontology_iris, catalog
    )
    metrics: dict[str, int | str] = {
        "catalog_access_mode": "full_fetch_fallback",
        "catalog_select_queries": 0,
        "catalog_graphs_fetched": len(catalog),
    }
    return ontology_iris, catalog, metrics


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for non-empty equal-length vectors, else 0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_fused(
    a_core: list[float],
    a_neighborhood: list[float],
    b_core: list[float],
    b_neighborhood: list[float],
    *,
    core_weight: float,
    neighborhood_weight: float,
) -> float:
    """Weighted cosine similarity using both core and neighborhood vectors."""
    core_sim = _cosine_similarity(a_core, b_core)
    neighborhood_sim = _cosine_similarity(a_neighborhood, b_neighborhood)
    return (core_weight * core_sim) + (neighborhood_weight * neighborhood_sim)


def _mmr_rerank(
    atoms: list[GraphAtom],
    vectors: dict[str, tuple[list[float], list[float]]],
    *,
    mmr_lambda: float,
    max_atoms: int,
    core_weight: float,
    neighborhood_weight: float,
) -> list[GraphAtom]:
    """Greedy MMR reranking over merged atoms."""
    if not atoms:
        return []
    if mmr_lambda < 0.0 or mmr_lambda > 1.0:
        raise ValueError("mmr_lambda must be in [0, 1]")
    if max_atoms < 0:
        raise ValueError("max_atoms must be >= 0")

    limit = len(atoms) if max_atoms == 0 else min(max_atoms, len(atoms))
    ranked = sorted(atoms, key=lambda atom: float(atom.score or 0.0), reverse=True)
    selected: list[GraphAtom] = []
    remaining = ranked.copy()

    while remaining and len(selected) < limit:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        best_idx = 0
        best_value = float("-inf")
        for idx, candidate in enumerate(remaining):
            relevance = float(candidate.score or 0.0)
            candidate_vecs = vectors.get(candidate.atom_id)
            max_similarity = 0.0
            if candidate_vecs is not None:
                for chosen in selected:
                    chosen_vecs = vectors.get(chosen.atom_id)
                    if chosen_vecs is None:
                        continue
                    sim = _cosine_fused(
                        candidate_vecs[0],
                        candidate_vecs[1],
                        chosen_vecs[0],
                        chosen_vecs[1],
                        core_weight=core_weight,
                        neighborhood_weight=neighborhood_weight,
                    )
                    if sim > max_similarity:
                        max_similarity = sim
            mmr_score = (mmr_lambda * relevance) - ((1.0 - mmr_lambda) * max_similarity)
            if mmr_score > best_value:
                best_value = mmr_score
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected


def _demote_case_mismatched_symbol_atoms(
    atoms: list[GraphAtom],
    query_text: str,
    *,
    policy: SymbolCaseMismatchPolicy,
    demote_factor: float,
) -> tuple[list[GraphAtom], int]:
    """Demote or drop atoms whose only symbol evidence is case-mismatched.

    The BM25/dense document text is case-folded before indexing, so prose
    ``meV`` also retrieves ``unit:MegaEV`` (symbol ``MeV``) — one token away
    from a 10^9 unit error. Symbol surfaces are case-significant by
    definition (the lexical-trigger lane matches them exact-case); when a
    query token equals one of an atom's symbol surfaces only after case
    folding, and *no* query token matches any surface exactly, the symbol
    evidence is counterfeit. Domain-agnostic: surfaces come from declared
    ``skos:notation`` / ``qudt:symbol`` / ``qudt:ucumCode`` literals, never
    from a vocabulary-specific list.

    Atoms without symbol surfaces, exact-case matches, and label-only matches
    are never touched; ``demote`` softens rather than removes, so an atom
    with genuine semantic evidence survives at a lower rank.

    Returns:
        Tuple of (atoms after policy application, penalized count).
    """
    if policy is SymbolCaseMismatchPolicy.OFF or not query_text or not atoms:
        return atoms, 0
    tokens = set(tokenize_for_lexical_match(query_text))
    if not tokens:
        return atoms, 0
    folded_tokens = {token.lower() for token in tokens}
    out: list[GraphAtom] = []
    penalized = 0
    for atom in atoms:
        surfaces = [s for s in atom.symbol_surfaces if s.strip()]
        if not surfaces:
            out.append(atom)
            continue
        if any(surface in tokens for surface in surfaces):
            out.append(atom)
            continue
        if not any(surface.lower() in folded_tokens for surface in surfaces):
            out.append(atom)
            continue
        penalized += 1
        if policy is SymbolCaseMismatchPolicy.DROP:
            continue
        out.append(
            atom.model_copy(update={"score": float(atom.score or 0.0) * demote_factor})
        )
    return out, penalized


def _merge_lexical_trigger_atoms(
    merged: list[GraphAtom],
    trigger_atoms: list[GraphAtom],
    fusion: LexicalTriggerFusion = LexicalTriggerFusion.MAX_MERGE,
) -> tuple[list[GraphAtom], int, int]:
    """Fuse lexical-trigger hits with semantic seeds.

    ``max_merge`` promotes an atom retrieval already found to
    ``max(semantic score, trigger score)`` — a case-sensitive exact notation match
    is evidence, not a duplicate — and appends unseen atoms. ``append`` (legacy)
    only appends unseen atoms, discarding trigger evidence for known IRIs.

    A fourth RRF fusion channel was considered and rejected: a trigger match is
    binary per chunk (no ranked list per query window), so RRF over it degenerates
    to a constant rank-1 bonus — which max_merge implements directly.

    Returns:
        Tuple of (fused atoms, promoted count, appended count).
    """
    if not trigger_atoms:
        return merged, 0, 0
    trigger_score_by_iri: dict[str, float] = {}
    for atom in trigger_atoms:
        if atom.iri:
            trigger_score_by_iri[atom.iri] = max(
                trigger_score_by_iri.get(atom.iri, 0.0), float(atom.score or 0.0)
            )
    promoted = 0
    out: list[GraphAtom] = []
    existing_iris: set[str] = set()
    for atom in merged:
        iri = atom.iri
        if iri:
            existing_iris.add(iri)
        trigger_score = trigger_score_by_iri.get(iri or "")
        if (
            fusion is LexicalTriggerFusion.MAX_MERGE
            and trigger_score is not None
            and trigger_score > float(atom.score or 0.0)
        ):
            out.append(atom.model_copy(update={"score": trigger_score}))
            promoted += 1
        else:
            out.append(atom)
    appended = 0
    for atom in trigger_atoms:
        iri = atom.iri
        if iri and iri in existing_iris:
            continue
        out.append(atom)
        appended += 1
        if iri:
            existing_iris.add(iri)
    return out, promoted, appended


class OntologyPatchRetriever(Tool):
    """Combines vector retrieval into one composite ontology graph."""

    vector_store: VectorStoreManager = Field(exclude=True)
    sparql_tool: Any | None = Field(default=None, exclude=True)
    # Typed ``Any`` for the same reason as ``sparql_tool``: OntologyManager holds a
    # back-reference to this class, so a concrete annotation would be a cycle.
    ontology_manager: Any | None = Field(default=None, exclude=True)
    patch: PatchRetrievalConfig = Field(
        default_factory=PatchRetrievalConfig,
        exclude=True,
    )
    _last_retrieval_metrics: dict[str, Any] = PrivateAttr(default_factory=dict)
    _surface_index: CatalogSurfaceIndex | None = PrivateAttr(default=None)
    # Whole-module graphs for the small-module closure, keyed by ontology IRI.
    # The catalog is stable for the life of a run, and every content unit hits
    # the same handful of modules — refetching per unit multiplies catalog
    # reads by the unit count for no new information. ``None`` caches a miss.
    _small_module_cache: dict[str, Ontology | None] = PrivateAttr(default_factory=dict)
    # Tenancy the cache was filled under. The retriever outlives a tenancy
    # switch, and serving one tenant's modules to another would be a leak.
    _small_module_cache_scope: str = PrivateAttr(default="")

    @property
    def last_retrieval_metrics(self) -> dict[str, Any]:
        return self._last_retrieval_metrics

    def _match_query_unit_signals(self, trigger_source: str) -> dict[str, str]:
        """Match number-adjacent unit tokens against catalog surface forms.

        Additive, outside the semantic atom budget (precedent: the lexical
        trigger lane). Returns ``{entity_iri: ontology_iri}``; empty when the
        lane is disabled or nothing matches.
        """
        if not self.vector_store.store_config.query_unit_signals_enabled:
            return {}
        manager = self.ontology_manager
        if manager is None or not trigger_source:
            return {}
        tokens = number_adjacent_tokens(trigger_source)
        if not tokens:
            return {}
        if self._surface_index is None:
            # Symbol/notation predicates come from configuration rather than
            # being compiled into query_signals; built lazily because the
            # store config is not available at PrivateAttr default time.
            self._surface_index = CatalogSurfaceIndex(
                symbol_predicates=[
                    URIRef(iri)
                    for iri in (
                        self.vector_store.store_config.induced_subgraph_symbol_predicates
                    )
                ]
            )
        matched = self._surface_index.match(tokens, manager.ontologies)
        if matched:
            logger.info(
                "Query unit signals matched %d entity(ies) from tokens %s",
                len(matched),
                sorted(tokens),
            )
        return matched

    @staticmethod
    def _schema_axiom_graph(
        merged_context: tuple[RDFGraph, dict[str, str]] | None,
        catalog: list[Ontology] | None,
    ) -> RDFGraph | None:
        """Pick the graph to read ``rdfs:domain``/``rdfs:range`` axioms from.

        Whichever of the two expansion paths materialized the ontologies wins;
        neither being available means the induced-subgraph call is fetching on
        its own and there is nothing local to close over.
        """
        if merged_context is not None:
            return merged_context[0]
        if catalog:
            combined = RDFGraph()
            for ontology in catalog:
                combined += ontology.graph
            return combined
        return None

    async def _apply_small_module_closure(
        self, graph: RDFGraph, hit_ontology_iris: list[str]
    ) -> None:
        """Merge whole small modules into the snapshot (header-stripped).

        A vocabulary small enough to fit entirely (e.g. a qualified-quantity
        module of ~20 terms) is included wholesale once any of its atoms is
        admitted: partial inclusion of a tiny module is what pushes the
        renderer to improvise near-miss property names.
        """
        closure_max = self.patch.small_module_closure_max_triples
        if closure_max <= 0:
            return
        modules = await self._asmall_module_candidates(hit_ontology_iris)
        closed: list[str] = []
        for onto_iri, ontology in modules:
            if len(ontology.graph) > closure_max:
                continue
            module_graph = Ontology.strip_ontology_header_triples(ontology.graph.copy())
            _drop_module_contribution(graph, module_graph)
            graph += module_graph
            for prefix, namespace_uri in ontology.graph.namespaces():
                graph.bind(prefix, namespace_uri)
            closed.append(onto_iri)
        if closed:
            self._last_retrieval_metrics["module_closure_iris"] = closed

    async def _asmall_module_candidates(
        self, hit_ontology_iris: list[str]
    ) -> list[tuple[str, Ontology]]:
        """Resolve hit ontologies to full graphs, manager first, store second.

        The in-memory manager is empty in every deployment that keeps its
        catalog in a triple store and fetches per query — which is the normal
        server configuration, and where this closure silently did nothing.
        """
        store_config = getattr(self.vector_store, "store_config", None)
        scope = str(getattr(store_config, "ontology_table", "") or "")
        if scope != self._small_module_cache_scope:
            self._small_module_cache.clear()
            self._small_module_cache_scope = scope

        wanted = sorted(set(hit_ontology_iris))
        resolved: list[tuple[str, Ontology]] = []
        missing: list[str] = []
        manager = self.ontology_manager
        for onto_iri in wanted:
            if onto_iri in self._small_module_cache:
                cached = self._small_module_cache[onto_iri]
                if cached is not None:
                    resolved.append((onto_iri, cached))
                continue
            ontology = (
                manager.get_freshest_terminal_ontology_by_iri(onto_iri)
                if manager is not None
                else None
            )
            if ontology is None or ontology.is_null():
                missing.append(onto_iri)
            else:
                self._small_module_cache[onto_iri] = ontology
                resolved.append((onto_iri, ontology))

        store = self.sparql_tool.triple_store_manager if self.sparql_tool else None
        if missing and store is not None:
            try:
                fetched = await store.afetch_ontologies_by_iri(missing)
            except Exception as exc:
                # Do NOT cache on this path. A None entry is a permanent
                # negative (see the miss-caching note above), so memoizing a
                # transient store error would silently strip the small-module
                # closure from every later unit in the process.
                logger.warning(
                    "Small-module closure catalog fetch failed (not cached, "
                    "will retry on the next unit): %s",
                    exc,
                )
                return sorted(resolved, key=lambda item: item[0])
            by_iri = {o.iri: o for o in fetched if o.iri and not o.is_null()}
            for onto_iri in missing:
                found = by_iri.get(onto_iri)
                self._small_module_cache[onto_iri] = found
                if found is not None:
                    resolved.append((onto_iri, found))
        return sorted(resolved, key=lambda item: item[0])

    async def _acandidate_context(
        self,
        *,
        entity_uris: list[str],
        ontology_iris: list[str],
        ontology_version_filters: dict[str, set[str]] | None,
        ontology_hash_filters: dict[str, set[str]] | None,
        depth: int,
    ) -> tuple[RDFGraph, dict[str, str]]:
        """Build the working graph from a CONSTRUCT instead of merging catalogs.

        Version and hash filters are applied to the *headers*, so the CONSTRUCT is
        restricted to exactly the named graphs the merge path would have selected.

        Prefix bindings cannot come from a CONSTRUCT, so they are rebuilt from the
        catalog's author-prefix table; standard vocabulary prefixes are bound
        downstream by :func:`_bind_common_vocab_prefixes` as on the merge path.

        Returns:
            tuple: ``(candidate_graph, prefix_map)``.
        """
        manager = self.ontology_manager
        assert manager is not None and self.sparql_tool is not None
        store = self.sparql_tool.triple_store_manager
        headers = select_relevant_ontologies(
            dedupe_terminal_ontologies(await manager.aget_catalog_headers()),
            ontology_iris,
            ontology_version_filters,
            ontology_hash_filters,
        )
        if not headers:
            return RDFGraph(), {}

        graph_irefs = _sparql_irefs([header.graph_uri for header in headers])
        seed_irefs = _sparql_irefs(entity_uris)
        if not graph_irefs or not seed_irefs:
            return RDFGraph(), {}

        candidate = RDFGraph()
        for chunk in _chunked(seed_irefs, _MAX_VALUES_TERMS):
            partial = await store.aconstruct(
                build_candidate_subgraph_query(chunk, graph_irefs, depth=depth)
            )
            candidate += partial

        prefix_map: dict[str, str] = {}
        for header in headers:
            namespace = str(header.namespace)
            prefix = manager.author_prefix_for_namespace(namespace)
            if prefix:
                prefix_map[prefix] = namespace
        prefix_map = filter_overbroad_namespace_map(prefix_map)
        for prefix, namespace in prefix_map.items():
            candidate.bind(prefix, Namespace(namespace))
        # Mirror the merge path: author @prefix names persisted as sh:declare
        # triples (pulled by the candidate CONSTRUCT's header branch) win over
        # stem-derived recovery, exactly as ontology_from_named_graph binds them
        # for merged catalog graphs.
        declared = candidate.bind_declared_prefixes()
        known_before_declared = set(prefix_map.values())
        for namespace, prefix in declared.items():
            if namespace not in known_before_declared:
                prefix_map[prefix] = namespace
        # Graphs served from a triple store carry no author @prefix bindings, so
        # stem-derived prefixes fill any remaining gap (see
        # ontology_from_named_graph). Recover the same implicit stems here so
        # both context paths advertise identical namespaces.
        candidate.bind_implicit_namespaces()
        known_namespaces = set(prefix_map.values())
        for prefix, namespace_uri in candidate.namespaces():
            ns = str(namespace_uri)
            if (
                not prefix
                or ns in known_namespaces
                or ns in RDFLIB_DEFAULT_NAMESPACE_URIS
            ):
                continue
            prefix_map[prefix] = ns
        return candidate, prefix_map

    async def _aresolve_merged_context(
        self,
        *,
        entity_uris: list[str],
        ontology_iris: list[str],
        catalog: list[Ontology] | None,
        ontology_version_filters: dict[str, set[str]] | None,
        ontology_hash_filters: dict[str, set[str]] | None,
        depth: int,
        candidate_pushdown: bool,
    ) -> tuple[RDFGraph, dict[str, str]] | None:
        """Resolve the merged ontology context through the catalog, or ``None``.

        Returning ``None`` leaves the induced-subgraph call on its own fetch path,
        which is what happens when no catalog is registered or a read fails.
        ``catalog`` being set means the reference-expansion fallback already
        materialized everything, so there is nothing left to save here.

        Args:
            ontology_iris: Ontology IRIs surviving reference expansion.
            catalog: Ontologies already materialized by the fallback path, if any.
            ontology_version_filters: Allowed versions per ontology IRI.
            ontology_hash_filters: Allowed hashes per ontology IRI.

        Returns:
            tuple | None: ``(merged_graph, prefix_map)``, or ``None`` to fall back.
        """
        manager = self.ontology_manager
        if manager is None or catalog is not None:
            return None
        store = self.sparql_tool.triple_store_manager if self.sparql_tool else None
        use_pushdown = (
            candidate_pushdown
            and store is not None
            and store.supports_sparql_construct()
        )
        try:
            if use_pushdown:
                merged = await self._acandidate_context(
                    entity_uris=entity_uris,
                    ontology_iris=ontology_iris,
                    ontology_version_filters=ontology_version_filters,
                    ontology_hash_filters=ontology_hash_filters,
                    depth=depth,
                )
                mode = "sparql_candidate"
            else:
                selected = select_relevant_ontologies(
                    await manager.aget_ontologies_by_iri(ontology_iris),
                    ontology_iris,
                    ontology_version_filters,
                    ontology_hash_filters,
                )
                merged = await manager.aget_merged_graph(selected)
                mode = "merged_catalog"
        except Exception as exc:
            logger.warning(
                "Catalog context via OntologyManager failed (%s); "
                "falling back to a direct triple-store read",
                exc,
            )
            return None
        self._last_retrieval_metrics.update(manager.catalog_cache_stats())
        self._last_retrieval_metrics["catalog_context_mode"] = mode
        self._last_retrieval_metrics["catalog_context_triples"] = len(merged[0])
        return merged

    def _effective_top_k(self, top_k: int | None) -> int:
        if top_k is not None:
            return top_k
        return self.vector_store.store_config.top_k

    def _resolve_subgraph_budget(
        self,
        subgraph_depth: int | None,
        max_total_triples: int | None,
        estimated_triples_per_query: int | None,
    ) -> tuple[int, int, int]:
        """Fill unset induced-subgraph budget arguments from configuration."""
        sc = self.vector_store.store_config
        return (
            sc.induced_subgraph_depth if subgraph_depth is None else subgraph_depth,
            (
                sc.induced_subgraph_max_total_triples
                if max_total_triples is None
                else max_total_triples
            ),
            (
                sc.induced_subgraph_estimated_triples_per_query
                if estimated_triples_per_query is None
                else estimated_triples_per_query
            ),
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        expand_sparql: bool = True,
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
        estimated_triples_per_query: int | None = None,
    ) -> tuple[RDFGraph, list[str]]:
        """Retrieve top-k hits for one query and optional induced subgraph; returns source ontology IRIs."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aretrieve(
                    query=query,
                    top_k=top_k,
                    expand_sparql=expand_sparql,
                    subgraph_depth=subgraph_depth,
                    max_total_triples=max_total_triples,
                    estimated_triples_per_query=estimated_triples_per_query,
                )
            )
        raise RuntimeError(
            "retrieve() cannot be called from async code; use await aretrieve()"
        )

    def retrieve_ensemble(
        self,
        queries: list[str],
        top_k: int | None = None,
        expand_sparql: bool = True,
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
        estimated_triples_per_query: int | None = None,
        trigger_text: str | None = None,
    ) -> tuple[RDFGraph, list[str]]:
        """Sync: one induced graph and source IRIs for the union of vector hits over ``queries``."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aretrieve_ensemble(
                    queries=queries,
                    top_k=top_k,
                    expand_sparql=expand_sparql,
                    subgraph_depth=subgraph_depth,
                    max_total_triples=max_total_triples,
                    estimated_triples_per_query=estimated_triples_per_query,
                    trigger_text=trigger_text,
                )
            )
        raise RuntimeError(
            "retrieve_ensemble() is not allowed inside async code; use aretrieve_ensemble()"
        )

    async def aretrieve(
        self,
        query: str,
        top_k: int | None = None,
        expand_sparql: bool = True,
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
        estimated_triples_per_query: int | None = None,
        trigger_text: str | None = None,
    ) -> tuple[RDFGraph, list[str]]:
        """Async single-query variant of :meth:`aretrieve_ensemble`."""
        return await self.aretrieve_ensemble(
            queries=[query],
            top_k=top_k,
            expand_sparql=expand_sparql,
            subgraph_depth=subgraph_depth,
            max_total_triples=max_total_triples,
            estimated_triples_per_query=estimated_triples_per_query,
            trigger_text=trigger_text,
        )

    async def aretrieve_ensemble(
        self,
        queries: list[str],
        top_k: int | None = None,
        expand_sparql: bool = True,
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
        estimated_triples_per_query: int | None = None,
        trigger_text: str | None = None,
    ) -> tuple[RDFGraph, list[str]]:
        """Vector search over all ``queries`` once, score-filter, dedupe, single subgraph expansion.

        ``subgraph_depth`` / ``max_total_triples`` / ``estimated_triples_per_query``
        default to the configured values (``ONTOLOGY_PATCH_INDUCED_SUBGRAPH_*``).
        They previously carried literal defaults of 1 / 300 / 24, which
        contradicted the config defaults of 2 / 1200 / 24: the pipeline passed
        config explicitly and was unaffected, but any other caller of this
        public API silently got a 4x smaller snapshot than the deployment was
        configured for.
        """
        self._last_retrieval_metrics = {}
        subgraph_depth, max_total_triples, estimated_triples_per_query = (
            self._resolve_subgraph_budget(
                subgraph_depth, max_total_triples, estimated_triples_per_query
            )
        )
        trigger_source = (trigger_text or "").strip()
        if not queries and not trigger_source:
            return RDFGraph(), []

        eff_top_k = self._effective_top_k(top_k)
        hits_by_query: list[OntologySearchHitsByChannel] = []
        if queries:
            hits_by_query = await self.vector_store.asearch_patch_hits_many(
                queries=queries,
                top_k=eff_top_k,
            )
        sc = self.vector_store.store_config
        pc = self.patch
        eff_max_atoms = pc.effective_max_atoms(len(queries))
        merged = _filter_and_merge_patch_hits(
            hits_by_query,
            store_config=sc,
            patch_config=pc,
            per_query_core_score_ratio=pc.per_query_core_score_ratio,
            per_query_neighborhood_score_ratio=pc.per_query_neighborhood_score_ratio,
            per_query_bm25_score_ratio=pc.per_query_bm25_score_ratio,
            min_core_query_best_score=pc.min_core_query_best_score,
            min_neighborhood_query_best_score=pc.min_neighborhood_query_best_score,
            min_bm25_query_best_score=pc.min_bm25_query_best_score,
            min_merged_max_score=pc.min_merged_max_score,
            max_atoms_total=0,
        )
        atoms_after_dedupe = len(merged)
        merged = [atom for atom in merged if not _is_ontology_declaration_atom(atom)]

        if merged and pc.merged_score_ratio > 0.0:
            merged_top = float(merged[0].score or 0.0)
            merged_floor = merged_top * pc.merged_score_ratio
            merged = [
                atom for atom in merged if float(atom.score or 0.0) >= merged_floor
            ]

        ranked_before_cut = list(merged)

        if merged and pc.mmr_lambda < 1.0:
            merged = _normalize_relevance_scores(merged)
            vectors = await self.vector_store.afetch_vectors(
                [atom.atom_id for atom in merged]
            )
            core_w, neigh_w = normalized_core_neighborhood_weights(sc)
            merged = _mmr_rerank(
                merged,
                vectors,
                mmr_lambda=pc.mmr_lambda,
                max_atoms=eff_max_atoms,
                core_weight=core_w,
                neighborhood_weight=neigh_w,
            )
        elif pc.cross_query_merge_mode in (
            CrossQueryMergeMode.MAX_SCORE,
            CrossQueryMergeMode.SUM_SCORE,
        ):
            merged = _select_atoms_round_robin_by_ontology(
                merged,
                per_ontology_seed_quota=pc.per_ontology_seed_quota,
                max_atoms=eff_max_atoms,
                per_ontology_atom_floor=pc.per_ontology_atom_floor,
                per_role_atom_floor=pc.per_role_atom_floor,
            )
        elif eff_max_atoms > 0:
            merged = merged[:eff_max_atoms]

        trigger_source = trigger_source or " ".join(queries)
        trigger_atoms = await asyncio.to_thread(
            self.vector_store.match_lexical_triggers, trigger_source
        )
        merged, trigger_promoted, trigger_appended = _merge_lexical_trigger_atoms(
            merged, trigger_atoms, fusion=sc.lexical_trigger_fusion
        )
        # After the trigger merge: an exact-case trigger hit is positive
        # evidence and exempts the atom; what remains penalizable is the
        # case-folded BM25/dense residue.
        merged, symbol_case_penalized = _demote_case_mismatched_symbol_atoms(
            merged,
            trigger_source,
            policy=sc.symbol_case_mismatch_policy,
            demote_factor=sc.symbol_case_mismatch_demote_factor,
        )

        if not merged:
            self._last_retrieval_metrics = {
                "query_count": len(queries),
                "top_k": eff_top_k,
                "effective_max_atoms": eff_max_atoms,
                "atoms_after_dedupe": atoms_after_dedupe,
                "atoms_final": 0,
                "seed_iris": [],
                "lexical_trigger_hits": len(trigger_atoms),
                "lexical_trigger_atom_ids": [a.atom_id for a in trigger_atoms],
                "lexical_trigger_promoted": trigger_promoted,
                "lexical_trigger_appended": trigger_appended,
                "symbol_case_penalized": symbol_case_penalized,
            }
            if pc.dump_ontology_ranks:
                self._last_retrieval_metrics["ontology_rank_diagnostics"] = (
                    build_ontology_rank_diagnostics(
                        hits_by_query, ranked_before_cut, []
                    )
                )
            return RDFGraph(), []

        source_iris = _source_iris_from_atoms(merged)
        seeds_by_ontology: dict[str, int] = defaultdict(int)
        for atom in merged:
            if atom.ontology_iri:
                seeds_by_ontology[atom.ontology_iri] += 1

        self._last_retrieval_metrics = {
            "query_count": len(queries),
            "top_k": eff_top_k,
            "effective_max_atoms": eff_max_atoms,
            "merge_mode": pc.cross_query_merge_mode.value,
            "atoms_after_dedupe": atoms_after_dedupe,
            "atoms_final": len(merged),
            "seed_iris": [atom.iri for atom in merged if atom.iri],
            "source_ontology_iris": source_iris,
            "seeds_by_ontology": dict(seeds_by_ontology),
            "lexical_trigger_hits": len(trigger_atoms),
            "lexical_trigger_atom_ids": [a.atom_id for a in trigger_atoms],
            "lexical_trigger_iris": [a.iri for a in trigger_atoms if a.iri],
            "lexical_trigger_promoted": trigger_promoted,
            "lexical_trigger_appended": trigger_appended,
            "symbol_case_penalized": symbol_case_penalized,
        }
        if pc.dump_ontology_ranks:
            self._last_retrieval_metrics["ontology_rank_diagnostics"] = (
                build_ontology_rank_diagnostics(
                    hits_by_query, ranked_before_cut, merged
                )
            )

        if not expand_sparql or self.sparql_tool is None:
            return RDFGraph(), source_iris

        entity_uris, entity_relevance, entity_roles = _ranked_entity_weights(merged)
        signal_entities = self._match_query_unit_signals(trigger_source)
        for signal_iri, signal_onto_iri in sorted(signal_entities.items()):
            if signal_iri in entity_relevance:
                continue
            entity_uris.append(signal_iri)
            entity_relevance[signal_iri] = sc.lexical_trigger_score
            entity_roles[signal_iri] = "resource"
        if signal_entities:
            self._last_retrieval_metrics["query_signal_iris"] = sorted(
                signal_entities.keys()
            )
        hit_ontology_iris = sorted(
            {atom.ontology_iri for atom in merged if atom.ontology_iri}
            | set(signal_entities.values())
        )
        ontology_version_filters: dict[str, set[str]] = {}
        ontology_hash_filters: dict[str, set[str]] = {}
        for atom in merged:
            if atom.ontology_iri and atom.ontology_version:
                ontology_version_filters.setdefault(atom.ontology_iri, set()).add(
                    str(atom.ontology_version)
                )
            if atom.ontology_iri and atom.ontology_hash:
                ontology_hash_filters.setdefault(atom.ontology_iri, set()).add(
                    atom.ontology_hash
                )

        ontology_iris = hit_ontology_iris
        catalog: list[Ontology] | None = None
        triple_store_manager = self.sparql_tool.triple_store_manager
        if triple_store_manager is not None:
            ontology_iris, catalog, expansion_metrics = await _aexpand_ontology_iris(
                triple_store_manager, entity_uris, hit_ontology_iris
            )
            expanded = sorted(set(ontology_iris) - set(hit_ontology_iris))
            if expanded:
                self._last_retrieval_metrics["expanded_ontology_iris"] = expanded
            self._last_retrieval_metrics.update(expansion_metrics)

        merged_context = await self._aresolve_merged_context(
            entity_uris=entity_uris,
            ontology_iris=ontology_iris,
            catalog=catalog,
            ontology_version_filters=ontology_version_filters or None,
            ontology_hash_filters=ontology_hash_filters or None,
            depth=subgraph_depth,
            candidate_pushdown=sc.induced_subgraph_candidate_pushdown,
        )

        schema_graph = self._schema_axiom_graph(merged_context, catalog)
        if schema_graph is not None:
            closure = _schema_closure_entities(
                schema_graph,
                entity_uris,
                max_entities=pc.schema_closure_max_entities,
                ancestor_depth=pc.schema_closure_ancestor_depth,
                seed_relevance=entity_relevance,
            )
            if closure:
                closure_score = _closure_floor_score(entity_relevance)
                for closure_iri, closure_role in closure.items():
                    entity_uris.append(closure_iri)
                    entity_relevance[closure_iri] = closure_score
                    entity_roles[closure_iri] = closure_role
                self._last_retrieval_metrics["schema_closure_iris"] = sorted(closure)

        hub_seed_count = sc.induced_subgraph_hub_seed_count
        ancestor_depth = sc.induced_subgraph_ancestor_closure_depth
        entity_groups: dict[str, str] = {
            atom.iri: atom.ontology_iri
            for atom in merged
            if atom.iri and atom.ontology_iri
        }
        for signal_iri, signal_onto_iri in signal_entities.items():
            entity_groups.setdefault(signal_iri, signal_onto_iri)
        symbol_predicates = tuple(
            URIRef(iri) for iri in sc.induced_subgraph_symbol_predicates
        )

        graph = await self.sparql_tool.aget_induced_subgraph(
            ontologies=catalog,
            merged=merged_context,
            entity_uris=entity_uris,
            entity_relevance=entity_relevance,
            entity_roles=entity_roles,
            ontology_iris=ontology_iris,
            depth=subgraph_depth,
            max_total_triples=max_total_triples,
            estimated_triples_per_query=estimated_triples_per_query,
            ontology_version_filters=ontology_version_filters or None,
            ontology_hash_filters=ontology_hash_filters or None,
            hub_seed_count=hub_seed_count,
            ancestor_closure_depth=ancestor_depth,
            type_promotion_score_factor=(
                sc.induced_subgraph_type_promotion_score_factor
            ),
            seed_order=sc.induced_subgraph_seed_order.value,
            entity_groups=entity_groups,
            extra_description_predicates=symbol_predicates,
        )
        await self._apply_small_module_closure(graph, hit_ontology_iris)

        self._last_retrieval_metrics["snapshot_triple_count"] = len(graph)
        self._last_retrieval_metrics["ontology_iris_for_expansion"] = ontology_iris
        self._last_retrieval_metrics.update(self.sparql_tool.last_finalize_metrics)

        _bind_common_vocab_prefixes(graph)
        return graph, source_iris
