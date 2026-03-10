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
        self, query: str, top_k: int = 10, expand_sparql: bool = False
    ) -> tuple[RDFGraph, list[OntologyAtom]]:
        """Retrieve top-k atoms and merge them into a context graph."""
        atoms = self.vector_store.search_patches(query=query, top_k=top_k)
        merged = RDFGraph()
        for atom in atoms:
            merged.parse(data=atom.turtle, format="turtle")
        if expand_sparql and self.sparql_tool is not None:
            # Placeholder for deterministic neighborhood expansion from triple store.
            # Current implementation returns merged atom neighborhoods only.
            return merged, atoms
        return merged, atoms
