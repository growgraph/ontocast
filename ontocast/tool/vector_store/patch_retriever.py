"""Retrieves multi-ontology context patches from vector search."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from pydantic import Field
from rdflib import Namespace

from ontocast.onto.constants import COMMON_PREFIXES
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.onto import Tool
from ontocast.tool.vector_store.core import GraphAtom, OntologySearchHit
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


def _filter_and_merge_patch_hits(
    hits_by_query: list[list[OntologySearchHit]],
    *,
    per_query_score_ratio: float,
    min_query_best_score: float,
    min_merged_max_score: float,
) -> list[GraphAtom]:
    """Per-query relative cutoff, optional per-query floor, dedupe by atom_id with max score.

    ``per_query_score_ratio`` applies within each query relative to that query's best score
    so weak queries still contribute their comparatively strong hits. When
    ``min_merged_max_score`` > 0 and the best retained score is below it, returns [].
    """
    if per_query_score_ratio < 0.0 or per_query_score_ratio > 1.0:
        raise ValueError("per_query_score_ratio must be in [0, 1]")
    collected: list[OntologySearchHit] = []
    for hits in hits_by_query:
        if not hits:
            continue
        best = max(h.score for h in hits)
        if min_query_best_score > 0.0 and best < min_query_best_score:
            continue
        floor = best * per_query_score_ratio
        for hit in hits:
            if hit.score >= floor:
                collected.append(hit)

    if not collected:
        return []

    best_by_id: dict[str, OntologySearchHit] = {}
    for hit in collected:
        aid = hit.atom.atom_id
        prev = best_by_id.get(aid)
        if prev is None or hit.score > prev.score:
            best_by_id[aid] = hit

    merged_hits = sorted(best_by_id.values(), key=lambda h: h.score, reverse=True)
    merged_max = merged_hits[0].score
    if min_merged_max_score > 0.0 and merged_max < min_merged_max_score:
        return []

    out: list[GraphAtom] = []
    for hit in merged_hits:
        atom = hit.atom.model_copy(update={"score": hit.score})
        out.append(atom)
    return out


class OntologyPatchRetriever(Tool):
    """Combines vector retrieval into one composite ontology graph."""

    vector_store: QdrantVectorStore = Field(exclude=True)
    sparql_tool: Any | None = Field(default=None, exclude=True)

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
        max_triples: int = 2000,
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
                    max_triples=max_triples,
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
        max_triples: int = 2000,
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
                    max_triples=max_triples,
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
        max_triples: int = 2000,
    ) -> tuple[RDFGraph, list[str]]:
        """Async single-query variant of :meth:`aretrieve_ensemble`."""
        return await self.aretrieve_ensemble(
            queries=[query],
            top_k=top_k,
            expand_sparql=expand_sparql,
            subgraph_depth=subgraph_depth,
            max_triples=max_triples,
        )

    async def aretrieve_ensemble(
        self,
        queries: list[str],
        top_k: int | None = None,
        expand_sparql: bool = True,
        subgraph_depth: int = 1,
        max_triples: int = 2000,
    ) -> tuple[RDFGraph, list[str]]:
        """Vector search over all ``queries`` once, score-filter, dedupe, single subgraph expansion.

        Hits are filtered per query relative to that query's best score (see
        ``QdrantConfig.patch_per_query_score_ratio``) so a strong query does not
        suppress weaker queries' comparatively strong hits. Optional
        ``patch_min_merged_max_score`` / ``patch_min_query_best_score`` reject
        globally weak or irrelevant sub-queries.

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
        merged = _filter_and_merge_patch_hits(
            hits_by_query,
            per_query_score_ratio=qc.patch_per_query_score_ratio,
            min_query_best_score=qc.patch_min_query_best_score,
            min_merged_max_score=qc.patch_min_merged_max_score,
        )
        source_iris = _source_iris_from_atoms(merged)

        if not expand_sparql or self.sparql_tool is None:
            return RDFGraph(), source_iris

        if not merged:
            return RDFGraph(), []

        entity_uris = sorted({atom.iri for atom in merged if atom.iri})
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
            ontology_iris=ontology_iris,
            depth=subgraph_depth,
            max_triples=max_triples,
            ontology_version_filters=ontology_version_filters or None,
            ontology_hash_filters=ontology_hash_filters or None,
        )
        _bind_common_vocab_prefixes(graph)
        return graph, source_iris
