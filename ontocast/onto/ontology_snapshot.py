"""Prompt-facing ontology snapshot: graph view with provenance, no catalog identity.

Assembled from catalog ontologies (``O* → S``). Not a versioned catalog subject;
writeback (``U → O*``) targets real :class:`Ontology` instances by namespace ownership.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from ontocast.onto.enum import OntologyAssemblyMode
from ontocast.onto.llm_graph_payload import LLMGraphWire
from ontocast.onto.model import BasePydanticModel
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph


class OntologySnapshot(BasePydanticModel):
    """Ephemeral multi-source ontology context for LLM prompts.

    Holds triples + prefix bindings and assembly provenance. Does **not** carry
    catalog ``iri`` / ``ontology_id`` / lineage — those belong only on
    :class:`~ontocast.onto.ontology.Ontology`.
    """

    graph: LLMGraphWire = Field(
        default_factory=RDFGraph,
        description="Prompt ontology triples and namespace bindings.",
    )
    source_iris: list[str] = Field(
        default_factory=list,
        description="Catalog ontology IRIs that contributed to this snapshot.",
    )
    assembly_mode: OntologyAssemblyMode = Field(
        default=OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
        description="How this snapshot was assembled.",
    )
    title: str | None = Field(default=None, description="Optional prompt title.")
    description: str | None = Field(
        default=None, description="Optional prompt description."
    )
    content_hash: str = Field(
        default="",
        description="Content hash of graph (set on construction / refresh).",
    )

    @model_validator(mode="after")
    def _ensure_content_hash(self) -> OntologySnapshot:
        if not self.content_hash and isinstance(self.graph, RDFGraph):
            self.content_hash = self.graph.hash() if len(self.graph) > 0 else ""
        return self

    def is_empty(self) -> bool:
        """True when the snapshot graph has no triples."""
        return len(self.graph) == 0

    def refresh_content_hash(self) -> None:
        """Recompute ``content_hash`` from the current graph."""
        self.content_hash = self.graph.hash() if len(self.graph) > 0 else ""

    def domain_prefix_pairs(self) -> list[tuple[str, str]]:
        """Domain prefix/namespace pairs from graph bindings (prompt hygiene)."""
        from ontocast.prompt.ontology_context import (
            extract_domain_prefix_pairs_from_graph,
        )

        return extract_domain_prefix_pairs_from_graph(self.graph)

    def describe_for_prompt(self) -> str:
        """Human-readable multi-source description for ontology-update prompts."""
        pairs = self.domain_prefix_pairs()
        prefix_lines = (
            "\n".join(f"  - `{p}:` <{ns}>" for p, ns in pairs)
            if pairs
            else "  (none declared beyond standard vocabularies)"
        )
        sources = (
            "\n".join(f"  - <{iri}>" for iri in self.source_iris)
            if self.source_iris
            else "  (none)"
        )
        title = self.title or "(untitled snapshot)"
        desc = self.description or ""
        return (
            f"Ontology context: {title}\n"
            f"Assembly mode: {self.assembly_mode.value}\n"
            f"Description: {desc}\n"
            f"Source catalog IRIs:\n{sources}\n"
            f"Domain prefixes:\n{prefix_lines}\n"
        )

    @classmethod
    def empty(
        cls,
        *,
        assembly_mode: OntologyAssemblyMode = OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE,
        title: str | None = None,
        description: str | None = None,
    ) -> OntologySnapshot:
        """Build an empty snapshot (no triples, no sources)."""
        return cls(
            graph=RDFGraph(),
            source_iris=[],
            assembly_mode=assembly_mode,
            title=title,
            description=description,
            content_hash="",
        )

    @classmethod
    def from_ontology(
        cls,
        ontology: Ontology,
        *,
        assembly_mode: OntologyAssemblyMode,
        title: str | None = None,
        description: str | None = None,
    ) -> OntologySnapshot:
        """Assemble a snapshot from a single catalog ontology (graph copy)."""
        if ontology.is_null():
            return cls.empty(
                assembly_mode=assembly_mode,
                title=title or "Null ontology",
                description=description or "No catalog ontology selected.",
            )
        graph = ontology.graph.copy()
        source = [ontology.iri] if ontology.iri else []
        return cls(
            graph=graph,
            source_iris=source,
            assembly_mode=assembly_mode,
            title=title or ontology.title,
            description=description or ontology.description,
            content_hash=graph.hash() if len(graph) > 0 else "",
        )

    @classmethod
    def from_graph(
        cls,
        graph: RDFGraph,
        *,
        source_iris: list[str],
        assembly_mode: OntologyAssemblyMode,
        title: str | None = None,
        description: str | None = None,
        strip_headers: bool = True,
    ) -> OntologySnapshot:
        """Assemble a snapshot from a (possibly multi-source) graph."""
        working = graph.copy()
        if strip_headers:
            Ontology.strip_ontology_header_triples(working)
        return cls(
            graph=working,
            source_iris=list(source_iris),
            assembly_mode=assembly_mode,
            title=title,
            description=description,
            content_hash=working.hash() if len(working) > 0 else "",
        )
