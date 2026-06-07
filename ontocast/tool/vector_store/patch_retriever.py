"""Retrieves multi-ontology context patches from vector search."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from pydantic import Field, PrivateAttr
from rdflib import Namespace, URIRef
from rdflib.namespace import RDFS

from ontocast.config import CrossQueryMergeMode, PatchRetrievalConfig, VectorStoreConfig
from ontocast.onto.constants import COMMON_PREFIXES
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.onto import Tool
from ontocast.tool.vector_store.core import (
    GraphAtom,
    OntologySearchHit,
    OntologySearchHitsByChannel,
    VectorStoreManager,
)
from ontocast.tool.vector_store.util import (
    normalized_core_neighborhood_weights,
    normalized_fusion_weights,
    rank_fuse_channel_hits,
)

logger = logging.getLogger(__name__)

_STRUCTURAL_REFERENCE_PREDICATES = frozenset({RDFS.subClassOf, RDFS.domain, RDFS.range})


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


def _filter_hits_by_relative_floor(
    hits: list[OntologySearchHit],
    *,
    score_ratio: float,
    min_query_best_score: float,
) -> list[OntologySearchHit]:
    """Relative score gating within one channel/query hit list."""
    if score_ratio < 0.0 or score_ratio > 1.0:
        raise ValueError("score_ratio must be in [0, 1]")
    if not hits:
        return []
    best = max(h.score for h in hits)
    if min_query_best_score > 0.0 and best < min_query_best_score:
        return []
    floor = best * score_ratio
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
    if merge_mode == CrossQueryMergeMode.RRF:
        return rank_fuse_channel_hits(
            collected,
            [],
            [],
            core_weight=1.0,
            neighborhood_weight=0.0,
            bm25_weight=0.0,
            limit=max(len(collected), 1),
        )
    if merge_mode == CrossQueryMergeMode.MAX_SCORE:
        merged = _merge_hits_across_queries_max_score(collected)
        if max_atoms_total > 0:
            merged = merged[:max_atoms_total]
        return merged
    return _merge_hits_across_queries_hybrid(
        collected,
        max_atoms_tier1=max_atoms_tier1,
        per_ontology_seed_quota=per_ontology_seed_quota,
        min_entity_score=min_entity_score,
        max_atoms_total=max_atoms_total,
    )


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

    merged_hits = _merge_hits_across_queries(
        collected,
        merge_mode=patch_config.cross_query_merge_mode,
        max_atoms_tier1=patch_config.max_atoms_tier1,
        per_ontology_seed_quota=patch_config.per_ontology_seed_quota,
        min_entity_score=patch_config.min_entity_score,
        max_atoms_total=0,
    )
    if not merged_hits:
        return []

    merged_max = merged_hits[0].score
    if min_merged_max_score > 0.0 and merged_max < min_merged_max_score:
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


class OntologyPatchRetriever(Tool):
    """Combines vector retrieval into one composite ontology graph."""

    vector_store: VectorStoreManager = Field(exclude=True)
    sparql_tool: Any | None = Field(default=None, exclude=True)
    patch: PatchRetrievalConfig = Field(
        default_factory=PatchRetrievalConfig,
        exclude=True,
    )
    _last_retrieval_metrics: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def last_retrieval_metrics(self) -> dict[str, Any]:
        return self._last_retrieval_metrics

    def _effective_top_k(self, top_k: int | None) -> int:
        if top_k is not None:
            return top_k
        return self.vector_store.store_config.top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        expand_sparql: bool = True,
        subgraph_depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
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
        subgraph_depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
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
        subgraph_depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
    ) -> tuple[RDFGraph, list[str]]:
        """Async single-query variant of :meth:`aretrieve_ensemble`."""
        return await self.aretrieve_ensemble(
            queries=[query],
            top_k=top_k,
            expand_sparql=expand_sparql,
            subgraph_depth=subgraph_depth,
            max_total_triples=max_total_triples,
            estimated_triples_per_query=estimated_triples_per_query,
        )

    async def aretrieve_ensemble(
        self,
        queries: list[str],
        top_k: int | None = None,
        expand_sparql: bool = True,
        subgraph_depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
    ) -> tuple[RDFGraph, list[str]]:
        """Vector search over all ``queries`` once, score-filter, dedupe, single subgraph expansion."""
        self._last_retrieval_metrics = {}
        if not queries:
            return RDFGraph(), []

        eff_top_k = self._effective_top_k(top_k)
        hits_by_query = await self.vector_store.asearch_patch_hits_many(
            queries=queries,
            top_k=eff_top_k,
        )
        sc = self.vector_store.store_config
        pc = self.patch
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
        )
        atoms_after_merge = len(merged)
        merged = [atom for atom in merged if not _is_ontology_declaration_atom(atom)]

        if merged and pc.merged_score_ratio > 0.0:
            merged_top = float(merged[0].score or 0.0)
            merged_floor = merged_top * pc.merged_score_ratio
            merged = [
                atom for atom in merged if float(atom.score or 0.0) >= merged_floor
            ]

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
                max_atoms=pc.max_atoms,
                core_weight=core_w,
                neighborhood_weight=neigh_w,
            )
        elif pc.max_atoms > 0:
            merged = merged[: pc.max_atoms]

        if not merged:
            self._last_retrieval_metrics = {
                "query_count": len(queries),
                "top_k": eff_top_k,
                "atoms_after_merge": atoms_after_merge,
                "atoms_final": 0,
            }
            return RDFGraph(), []

        source_iris = _source_iris_from_atoms(merged)
        seeds_by_ontology: dict[str, int] = defaultdict(int)
        for atom in merged:
            if atom.ontology_iri:
                seeds_by_ontology[atom.ontology_iri] += 1

        self._last_retrieval_metrics = {
            "query_count": len(queries),
            "top_k": eff_top_k,
            "merge_mode": pc.cross_query_merge_mode.value,
            "atoms_after_merge": atoms_after_merge,
            "atoms_final": len(merged),
            "source_ontology_iris": source_iris,
            "seeds_by_ontology": dict(seeds_by_ontology),
        }

        if not expand_sparql or self.sparql_tool is None:
            return RDFGraph(), source_iris

        entity_uris, entity_relevance, entity_roles = _ranked_entity_weights(merged)
        hit_ontology_iris = sorted(
            {atom.ontology_iri for atom in merged if atom.ontology_iri}
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
        if self.sparql_tool.triple_store_manager is not None:
            catalog = await self.sparql_tool.triple_store_manager.afetch_ontologies()
            ontology_iris = _expand_ontology_iris_by_reference(
                entity_uris,
                hit_ontology_iris,
                catalog,
            )
            expanded = sorted(set(ontology_iris) - set(hit_ontology_iris))
            if expanded:
                self._last_retrieval_metrics["expanded_ontology_iris"] = expanded

        hub_seed_count = sc.induced_subgraph_hub_seed_count
        ancestor_depth = sc.induced_subgraph_ancestor_closure_depth

        graph = await self.sparql_tool.aget_induced_subgraph(
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
        )
        self._last_retrieval_metrics["snapshot_triple_count"] = len(graph)
        self._last_retrieval_metrics["ontology_iris_for_expansion"] = ontology_iris
        self._last_retrieval_metrics.update(self.sparql_tool.last_finalize_metrics)

        _bind_common_vocab_prefixes(graph)
        return graph, source_iris
