"""Prompt-facing ontology snapshot: graph view with provenance, no catalog identity.

Assembled from catalog ontologies (``O* → S``). Not a versioned catalog subject;
writeback (``U → O*``) targets real :class:`Ontology` instances by namespace ownership.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, PrivateAttr

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
        description=(
            "Content hash of graph. Lazy: empty until refresh_content_hash() "
            "is called — canonical hashing is too expensive for the per-unit "
            "hot path."
        ),
    )

    #: Derived prompt text, keyed by graph identity so a reassigned graph misses.
    #: Populated only through :meth:`prompt_chapter`.
    _prompt_cache: dict[tuple[int, int, str], str] = PrivateAttr(default_factory=dict)

    def is_empty(self) -> bool:
        """True when the snapshot graph has no triples."""
        return len(self.graph) == 0

    def refresh_content_hash(self) -> None:
        """Recompute ``content_hash`` from the current graph."""
        self.content_hash = self.graph.hash() if len(self.graph) > 0 else ""

    def invalidate_prompt_cache(self) -> None:
        """Drop memoised prompt text.

        Call this after mutating :attr:`graph` in place. Reassigning ``graph``
        needs no call -- the cache key includes the graph's identity.
        """
        self._prompt_cache.clear()

    def prompt_chapter(self, profile: Any) -> str:
        """Serialised ontology chapter for prompts, memoised per graph.

        Serialising the ontology is the single most expensive step in building a
        facts prompt, and under a shared document snapshot every unit -- and
        every render attempt within a unit -- would otherwise redo it on an
        identical graph, synchronously, on the event loop.

        The memo is keyed on the graph's identity, length and the profile's wire
        format. It is therefore correct for a snapshot whose graph is replaced,
        and *assumes* the graph is not mutated in place, which is the contract
        this class already documents ("ephemeral" context, read-only in the unit
        loops). Any code that does mutate it must call
        :meth:`invalidate_prompt_cache`.

        Args:
            profile: Graph format profile supplying the serialisation.

        Returns:
            str: The ``# ONTOLOGY`` chapter, including the index appendix.
        """
        from ontocast.prompt.ontology_context import build_ontology_index

        key = (id(self.graph), len(self.graph), str(profile.format))
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached
        chapter = profile.format_ontology_chapter(
            self.graph, suffix=build_ontology_index(self.graph)
        )
        # Bound the memo: a snapshot only ever holds one live graph, so stale
        # entries are strictly dead weight after a reassignment.
        self._prompt_cache.clear()
        self._prompt_cache[key] = chapter
        return chapter

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
        )
