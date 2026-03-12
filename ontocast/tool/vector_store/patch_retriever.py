"""Retrieves multi-ontology context patches from vector search."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.onto import Tool
from ontocast.tool.vector_store.core import OntologyAtom
from ontocast.tool.vector_store.qdrant import QdrantVectorStore


class OntologyPatchRetriever(Tool):
    """Combines vector retrieval into one composite ontology graph."""

    vector_store: QdrantVectorStore = Field(exclude=True)
    sparql_tool: Any | None = Field(default=None, exclude=True)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        expand_sparql: bool = True,
        subgraph_depth: int = 1,
        max_triples: int = 2000,
    ) -> tuple[RDFGraph, list[OntologyAtom]]:
        """Retrieve top-k atoms and expand graph via triple-store/SPARQL lookup."""
        batched_results = self.retrieve_many(
            queries=[query],
            top_k=top_k,
            expand_sparql=expand_sparql,
            subgraph_depth=subgraph_depth,
            max_triples=max_triples,
        )
        if not batched_results:
            return RDFGraph(), []
        return batched_results[0]

    def retrieve_many(
        self,
        queries: list[str],
        top_k: int = 10,
        expand_sparql: bool = True,
        subgraph_depth: int = 1,
        max_triples: int = 2000,
    ) -> list[tuple[RDFGraph, list[OntologyAtom]]]:
        """Batch retrieve graph patches for many proposition queries."""
        if not queries:
            return []
        if not expand_sparql or self.sparql_tool is None:
            hits_by_query = self.vector_store.search_patch_hits_many(
                queries=queries,
                top_k=top_k,
            )
            return [(RDFGraph(), [hit.atom for hit in hits]) for hits in hits_by_query]

        hits_by_query = self.vector_store.search_patch_hits_many(
            queries=queries,
            top_k=top_k,
        )
        results: list[tuple[RDFGraph, list[OntologyAtom]]] = []
        for hits in hits_by_query:
            atoms = [hit.atom for hit in hits]
            if not atoms:
                results.append((RDFGraph(), []))
                continue
            entity_uris = sorted({atom.iri for atom in atoms if atom.iri})
            ontology_iris = sorted(
                {atom.ontology_iri for atom in atoms if atom.ontology_iri}
            )
            ontology_version_filters: dict[str, set[str]] = {}
            ontology_hash_filters: dict[str, set[str]] = {}
            for atom in atoms:
                if atom.ontology_iri and atom.ontology_version:
                    ontology_version_filters.setdefault(atom.ontology_iri, set()).add(
                        str(atom.ontology_version)
                    )
                if atom.ontology_iri and atom.ontology_hash:
                    ontology_hash_filters.setdefault(atom.ontology_iri, set()).add(
                        atom.ontology_hash
                    )

            graph = self.sparql_tool.get_induced_subgraph(
                entity_uris=entity_uris,
                ontology_iris=ontology_iris,
                depth=subgraph_depth,
                max_triples=max_triples,
                ontology_version_filters=ontology_version_filters or None,
                ontology_hash_filters=ontology_hash_filters or None,
            )
            results.append((graph, atoms))
        return results
