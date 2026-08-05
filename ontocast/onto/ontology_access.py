"""Read-only accessors for ontology context on document vs unit workflow state.

Centralizes prompt-effective ontology resolution so agents and stategraph code
use :class:`~ontocast.onto.ontology_snapshot.OntologySnapshot` views (assemble
product) rather than treating snapshots as catalog :class:`Ontology` instances.
"""

from collections.abc import Iterable
from typing import Protocol

from ontocast.onto.constants import prefix_lookup_for_ingest
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph, extract_known_prefixes
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.prompt.ontology_context import (
    extract_domain_prefix_pairs_from_graph,
)


class OntologyPromptSource(Protocol):
    """Ontology material used to build LLM prompts (TTL, prefixes, seed checks)."""

    def effective_graph_for_prompt(self) -> RDFGraph:
        """Graph whose triples should appear in the main ontology chapter."""
        ...

    def ontology_graph_for_prefixes(self) -> RDFGraph:
        """Graph used to collect namespace prefixes for TTL repair."""
        ...

    def has_non_empty_seed(self) -> bool:
        """Whether the seed snapshot has triples (vs empty / null context)."""
        ...

    def domain_prefix_pairs(self) -> list[tuple[str, str]]:
        """Domain ontology prefix/namespace pairs used for prompt instructions."""
        ...

    def prompt_ontology_description(self) -> str:
        """Human-readable description for ontology-update intros."""
        ...

    def writable_iris(self) -> list[str]:
        """Catalog IRIs that may receive writeback from this unit."""
        ...


def _merge_prefix_bindings_from_graph(
    merged: dict[str, str],
    graph: RDFGraph,
    extra_prefix: str | None = None,
    extra_namespace: str | None = None,
) -> None:
    """Add explicit and implicit namespace bindings from *graph* into *merged*."""
    for prefix, namespace_uri in extract_known_prefixes(
        graph,
        extra_prefix=extra_prefix,
        extra_namespace=extra_namespace,
    ).items():
        if prefix not in merged:
            merged[prefix] = namespace_uri

    scratch = graph.copy()
    scratch.bind_implicit_namespaces()
    for prefix, namespace_uri in scratch.namespaces():
        if prefix and prefix not in merged:
            merged[prefix] = str(namespace_uri)


_SEMANTIC_SUFFIXES = frozenset(
    {
        "relations",
        "concepts",
        "properties",
        "individuals",
        "classes",
        "roles",
        "attributes",
        "instances",
    }
)


def _add_semantic_aliases(prefix_map: dict[str, str]) -> dict[str, str]:
    """Extend *prefix_map* with URI-tail aliases for common semantic namespace segments."""
    new = {
        uri.rstrip("#/").rsplit("/", 1)[-1].lower(): uri
        for uri in prefix_map.values()
        if uri.rstrip("#/").rsplit("/", 1)[-1].lower() in _SEMANTIC_SUFFIXES
    }
    new = {k: v for k, v in new.items() if k not in prefix_map}
    return {**prefix_map, **new} if new else prefix_map


def build_llm_prefix_map_from_graphs(
    primary: RDFGraph,
    supplemental: Iterable[RDFGraph | Ontology] = (),
) -> dict[str, str]:
    """Collect namespace prefixes for LLM Turtle/JSON-LD ingest repair from graphs."""
    merged = prefix_lookup_for_ingest()
    _merge_prefix_bindings_from_graph(merged, primary)
    for item in supplemental:
        if isinstance(item, Ontology):
            if item.is_null():
                continue
            graph = item.graph
            if not isinstance(graph, RDFGraph):
                normalized = RDFGraph()
                for triple in graph:
                    normalized.add(triple)
                for prefix, namespace_uri in graph.namespaces():
                    normalized.bind(prefix, namespace_uri)
                graph = normalized
            _merge_prefix_bindings_from_graph(
                merged,
                graph,
                extra_prefix=item.prefix or None,
                extra_namespace=item.namespace or None,
            )
        else:
            _merge_prefix_bindings_from_graph(merged, item)
    return _add_semantic_aliases(merged)


def build_llm_prefix_map(
    primary: Ontology | OntologySnapshot | RDFGraph,
    supplemental: Iterable[Ontology] = (),
) -> dict[str, str]:
    """Collect namespace prefixes for LLM Turtle/JSON-LD ingest repair.

    Accepts a catalog :class:`Ontology`, an :class:`OntologySnapshot`, or a raw
    :class:`RDFGraph` as primary.
    """
    if isinstance(primary, OntologySnapshot):
        primary_graph = primary.graph
    elif isinstance(primary, Ontology):
        if primary.is_null():
            return _add_semantic_aliases(prefix_lookup_for_ingest())
        primary_graph = primary.graph
        if not isinstance(primary_graph, RDFGraph):
            normalized = RDFGraph()
            for triple in primary_graph:
                normalized.add(triple)
            for prefix, namespace_uri in primary_graph.namespaces():
                normalized.bind(prefix, namespace_uri)
            primary_graph = normalized
    else:
        primary_graph = primary
    return build_llm_prefix_map_from_graphs(primary_graph, supplemental)


def known_prefixes_for_llm_parse(
    source: OntologyPromptSource,
    supplemental: Iterable[Ontology] = (),
) -> dict[str, str]:
    """Collect namespace prefixes for TTL/JSON-LD repair during LLM output parsing."""
    return build_llm_prefix_map_from_graphs(
        source.ontology_graph_for_prefixes(), supplemental
    )


class UnitOntologyAccess:
    """Accessor for :class:`UnitOntologyState` (ontology map loop)."""

    __slots__ = ("_state",)

    def __init__(self, state: UnitOntologyState) -> None:
        self._state = state

    def effective_graph_for_prompt(self) -> RDFGraph:
        if len(self._state.working_graph) > 0:
            return self._state.working_graph
        return self._state.ontology_snapshot.graph

    def ontology_graph_for_prefixes(self) -> RDFGraph:
        return self.effective_graph_for_prompt()

    def has_non_empty_seed(self) -> bool:
        return not self._state.ontology_snapshot.is_empty()

    def domain_prefix_pairs(self) -> list[tuple[str, str]]:
        return extract_domain_prefix_pairs_from_graph(self.effective_graph_for_prompt())

    def prompt_ontology_description(self) -> str:
        return self._state.ontology_snapshot.describe_for_prompt()

    def writable_iris(self) -> list[str]:
        return list(self._state.writable_iris)

    # ---- compatibility shims during migration ----
    def effective_ontology_for_prompt(self) -> OntologySnapshot:
        """Return a transient snapshot view of the effective prompt graph."""
        snap = self._state.ontology_snapshot
        return OntologySnapshot(
            graph=self.effective_graph_for_prompt().copy(),
            source_iris=list(snap.source_iris),
            assembly_mode=snap.assembly_mode,
            title=snap.title,
            description=snap.description,
        )

    def ontology_for_prefixes(self) -> OntologySnapshot:
        return self.effective_ontology_for_prompt()

    def has_non_null_seed_snapshot(self) -> bool:
        return self.has_non_empty_seed()


class UnitFactsOntologyAccess:
    """Accessor for :class:`UnitFactsState`; facts prompts use snapshot context only."""

    __slots__ = ("_state",)

    def __init__(self, state: UnitFactsState) -> None:
        self._state = state

    def effective_graph_for_prompt(self) -> RDFGraph:
        return self._state.ontology_snapshot.graph

    def ontology_graph_for_prefixes(self) -> RDFGraph:
        return self._state.ontology_snapshot.graph

    def has_non_empty_seed(self) -> bool:
        return not self._state.ontology_snapshot.is_empty()

    def domain_prefix_pairs(self) -> list[tuple[str, str]]:
        return self._state.ontology_snapshot.domain_prefix_pairs()

    def prompt_ontology_description(self) -> str:
        return self._state.ontology_snapshot.describe_for_prompt()

    def writable_iris(self) -> list[str]:
        return list(self._state.writable_iris)

    def effective_ontology_for_prompt(self) -> OntologySnapshot:
        return self._state.ontology_snapshot

    def ontology_for_prefixes(self) -> OntologySnapshot:
        return self._state.ontology_snapshot

    def has_non_null_seed_snapshot(self) -> bool:
        return self.has_non_empty_seed()


class DocumentOntologyAccess:
    """Accessor for :class:`AgentState` (document-level reduce / serialize)."""

    __slots__ = ("_state",)

    def __init__(self, state: AgentState) -> None:
        self._state = state

    def reduced_artifacts(self) -> list[Ontology]:
        if self._state.reduced_ontology_artifacts:
            return list(self._state.reduced_ontology_artifacts)
        return list(self._state.ontology_artifacts)

    def has_any_artifacts(self) -> bool:
        return bool(
            self._state.reduced_ontology_artifacts or self._state.ontology_artifacts
        )

    def has_non_null_artifacts(self) -> bool:
        return any(not ontology.is_null() for ontology in self.reduced_artifacts())

    def ontology_by_anchor(self, anchor_iri: str) -> Ontology | None:
        if anchor_iri in self._state.reduced_ontology_by_anchor:
            return self._state.reduced_ontology_by_anchor[anchor_iri]
        for ontology in self.reduced_artifacts():
            if ontology.iri == anchor_iri:
                return ontology
        return None

    def serialization_targets(self) -> list[Ontology]:
        """Ontologies to version and persist (per-catalog-IRI artifacts after apply)."""
        artifacts = self.reduced_artifacts()
        if artifacts:
            return artifacts
        return []


def ontology_access_for_unit_ontology(state: UnitOntologyState) -> UnitOntologyAccess:
    return UnitOntologyAccess(state)


def ontology_access_for_unit_facts(state: UnitFactsState) -> UnitFactsOntologyAccess:
    return UnitFactsOntologyAccess(state)


def document_ontology_access(state: AgentState) -> DocumentOntologyAccess:
    return DocumentOntologyAccess(state)
