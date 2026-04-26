"""Retrieves multi-ontology context patches from vector search."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable
from typing import Any

from pydantic import Field
from rdflib import Namespace

from ontocast.config import PatchRetrievalConfig, QdrantConfig
from ontocast.onto.constants import COMMON_PREFIXES
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.onto import Tool
from ontocast.tool.vector_store.core import (
    GraphAtom,
    OntologySearchHit,
    OntologySearchHitsByChannel,
)
from ontocast.tool.vector_store.qdrant import QdrantVectorStore


def _bind_common_vocab_prefixes(graph: RDFGraph) -> None:
    """Declare standard RDF/SKOS/DC prefixes when missing (better Turtle for entities)."""
    bound = {prefix for prefix, _ in graph.namespaces() if prefix}
    for prefix, uri_wrapped in COMMON_PREFIXES.items():
        if prefix in bound:
            continue
        graph.bind(prefix, Namespace(uri_wrapped.strip("<>")))


def _source_iris_from_atoms(atoms: Iterable[GraphAtom]) -> list[str]:
    return sorted({atom.ontology_iri for atom in atoms if atom.ontology_iri})


def _ranked_entity_weights(
    atoms: list[GraphAtom],
) -> tuple[list[str], dict[str, float]]:
    """Collapse atom scores to entity-level ranking and relevance weights."""
    best_score_by_iri: dict[str, float] = {}
    for atom in atoms:
        iri = atom.iri
        if not iri:
            continue
        score = float(atom.score or 0.0)
        previous = best_score_by_iri.get(iri)
        if previous is None or score > previous:
            best_score_by_iri[iri] = score
    ranked = sorted(
        best_score_by_iri.keys(),
        key=lambda iri: (-best_score_by_iri[iri], iri),
    )
    return ranked, best_score_by_iri


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


def _normalized_fusion_weights_triple(
    qc: QdrantConfig,
) -> tuple[float, float, float]:
    cw, nw, bw = (
        qc.fusion_core_weight,
        qc.fusion_neighborhood_weight,
        qc.fusion_bm25_weight,
    )
    total = cw + nw + bw
    if total <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (cw / total, nw / total, bw / total)


def _rank_fuse_hits(
    core_hits: list[OntologySearchHit],
    neighborhood_hits: list[OntologySearchHit],
    bm25_hits: list[OntologySearchHit],
    *,
    core_weight: float,
    neighborhood_weight: float,
    bm25_weight: float,
) -> list[OntologySearchHit]:
    """Rank-fuse core, neighborhood, and optional BM25 hit lists by weighted reciprocal rank."""
    best_by_id: dict[str, OntologySearchHit] = {}
    rank_score_by_id: dict[str, float] = {}

    for rank, hit in enumerate(core_hits, start=1):
        aid = hit.atom.atom_id
        rank_score_by_id[aid] = rank_score_by_id.get(aid, 0.0) + (core_weight / rank)
        prev = best_by_id.get(aid)
        if prev is None or hit.score > prev.score:
            best_by_id[aid] = hit

    for rank, hit in enumerate(neighborhood_hits, start=1):
        aid = hit.atom.atom_id
        rank_score_by_id[aid] = rank_score_by_id.get(aid, 0.0) + (
            neighborhood_weight / rank
        )
        prev = best_by_id.get(aid)
        if prev is None or hit.score > prev.score:
            best_by_id[aid] = hit

    for rank, hit in enumerate(bm25_hits, start=1):
        aid = hit.atom.atom_id
        rank_score_by_id[aid] = rank_score_by_id.get(aid, 0.0) + (bm25_weight / rank)
        prev = best_by_id.get(aid)
        if prev is None or hit.score > prev.score:
            best_by_id[aid] = hit

    ranked_ids = sorted(
        rank_score_by_id.keys(),
        key=lambda aid: (rank_score_by_id[aid], best_by_id[aid].score, aid),
        reverse=True,
    )
    out: list[OntologySearchHit] = []
    for aid in ranked_ids:
        source_hit = best_by_id[aid]
        atom = source_hit.atom.model_copy(update={"score": rank_score_by_id[aid]})
        out.append(OntologySearchHit(atom=atom, score=rank_score_by_id[aid]))
    return out


def _filter_and_merge_patch_hits(
    hits_by_query: list[OntologySearchHitsByChannel],
    *,
    qdrant_config: QdrantConfig,
    per_query_core_score_ratio: float,
    per_query_neighborhood_score_ratio: float,
    per_query_bm25_score_ratio: float,
    min_core_query_best_score: float,
    min_neighborhood_query_best_score: float,
    min_bm25_query_best_score: float,
    min_merged_max_score: float,
) -> list[GraphAtom]:
    """Filter each channel per query, then rank-fuse per query and across queries."""
    cw, nw, bw = _normalized_fusion_weights_triple(qdrant_config)
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
            _rank_fuse_hits(
                filtered_core,
                filtered_neighborhood,
                filtered_bm25,
                core_weight=cw,
                neighborhood_weight=nw,
                bm25_weight=bw,
            )
        )

    if not collected:
        return []

    merged_hits = _rank_fuse_hits(
        collected,
        [],
        [],
        core_weight=1.0,
        neighborhood_weight=0.0,
        bm25_weight=0.0,
    )
    merged_max = merged_hits[0].score
    if min_merged_max_score > 0.0 and merged_max < min_merged_max_score:
        return []

    out: list[GraphAtom] = []
    for hit in merged_hits:
        atom = hit.atom.model_copy(update={"score": hit.score})
        out.append(atom)
    return out


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

    vector_store: QdrantVectorStore = Field(exclude=True)
    sparql_tool: Any | None = Field(default=None, exclude=True)
    patch: PatchRetrievalConfig = Field(
        default_factory=PatchRetrievalConfig,
        exclude=True,
    )

    def _effective_top_k(self, top_k: int | None) -> int:
        if top_k is not None:
            return top_k
        return self.vector_store.config.top_k

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
        """Vector search over all ``queries`` once, score-filter, dedupe, single subgraph expansion.

        Hits are filtered per query and per channel relative to each channel's best
        score (see ``PatchRetrievalConfig`` per-query ratio fields for core,
        neighborhood, and BM25), then merged by rank fusion so channels with
        different score distributions all contribute. Optional per-channel
        min-best filters and ``min_merged_max_score`` reject weak or irrelevant
        candidates.

        Returns the merged RDF graph (possibly disconnected across ontologies) and sorted
        distinct ontology IRIs that contributed vector hits.
        """
        if not queries:
            return RDFGraph(), []
        eff_top_k = self._effective_top_k(top_k)
        hits_by_query = await self.vector_store.asearch_patch_hits_many(
            queries=queries,
            top_k=eff_top_k,
        )
        qc = self.vector_store.config
        pc = self.patch
        merged = _filter_and_merge_patch_hits(
            hits_by_query,
            qdrant_config=qc,
            per_query_core_score_ratio=pc.per_query_core_score_ratio,
            per_query_neighborhood_score_ratio=pc.per_query_neighborhood_score_ratio,
            per_query_bm25_score_ratio=pc.per_query_bm25_score_ratio,
            min_core_query_best_score=pc.min_core_query_best_score,
            min_neighborhood_query_best_score=pc.min_neighborhood_query_best_score,
            min_bm25_query_best_score=pc.min_bm25_query_best_score,
            min_merged_max_score=pc.min_merged_max_score,
        )
        if merged and pc.merged_score_ratio > 0.0:
            merged_top = float(merged[0].score or 0.0)
            merged_floor = merged_top * pc.merged_score_ratio
            merged = [
                atom for atom in merged if float(atom.score or 0.0) >= merged_floor
            ]
        if merged and pc.mmr_lambda < 1.0:
            vectors = await self.vector_store.afetch_vectors(
                [atom.atom_id for atom in merged]
            )
            merged = _mmr_rerank(
                merged,
                vectors,
                mmr_lambda=pc.mmr_lambda,
                max_atoms=pc.max_atoms,
                core_weight=qc.fusion_core_weight,
                neighborhood_weight=qc.fusion_neighborhood_weight,
            )
        elif pc.max_atoms > 0:
            merged = merged[: pc.max_atoms]
        source_iris = _source_iris_from_atoms(merged)

        if not expand_sparql or self.sparql_tool is None:
            return RDFGraph(), source_iris

        if not merged:
            return RDFGraph(), []

        entity_uris, entity_relevance = _ranked_entity_weights(merged)
        ontology_iris = sorted(
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

        graph = await self.sparql_tool.aget_induced_subgraph(
            entity_uris=entity_uris,
            entity_relevance=entity_relevance,
            ontology_iris=ontology_iris,
            depth=subgraph_depth,
            max_total_triples=max_total_triples,
            estimated_triples_per_query=estimated_triples_per_query,
            ontology_version_filters=ontology_version_filters or None,
            ontology_hash_filters=ontology_hash_filters or None,
        )
        _bind_common_vocab_prefixes(graph)
        return graph, source_iris
