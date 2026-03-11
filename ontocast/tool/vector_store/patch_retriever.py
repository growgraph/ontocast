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
        self, query: str, top_k: int = 10, expand_sparql: bool = True
    ) -> tuple[RDFGraph, list[OntologyAtom]]:
        """Retrieve top-k atoms and expand graph via triple-store/SPARQL lookup."""
        atoms = self.vector_store.search_patches(query=query, top_k=top_k)
        if not expand_sparql or self.sparql_tool is None or not atoms:
            return RDFGraph(), atoms

        entity_uris = sorted({atom.iri for atom in atoms if atom.iri})
        ontology_iris = sorted(
            {atom.ontology_iri for atom in atoms if atom.ontology_iri}
        )
        expanded = self.sparql_tool.get_induced_subgraph(
            entity_uris=entity_uris,
            ontology_iris=ontology_iris,
            depth=1,
        )
        return expanded, atoms
