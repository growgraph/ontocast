"""Read-only accessors for ontology context on document vs unit workflow state.

Centralizes prompt-effective ontology resolution and serialization target lists
so agents and stategraph code do not duplicate ``ontology_snapshot`` /
``ontology_artifacts`` branching.
"""

from collections.abc import Iterable
from typing import Protocol

from ontocast.onto.constants import prefix_lookup_for_ingest
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph, extract_known_prefixes
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.prompt.ontology_context import extract_domain_prefix_pairs


class OntologyPromptSource(Protocol):
    """Ontology material used to build LLM prompts (TTL, prefixes, seed checks)."""

    def effective_ontology_for_prompt(self) -> Ontology:
        """Ontology whose graph and metadata should appear in the main prompt."""
        ...

    def ontology_for_prefixes(self) -> Ontology:
        """Ontology used to collect namespace prefixes for TTL repair."""
        ...

    def has_non_null_seed_snapshot(self) -> bool:
        """Whether the immutable snapshot anchor is a real ontology (vs null IRI)."""
        ...

    def domain_prefix_pairs(self) -> list[tuple[str, str]]:
        """Domain ontology prefix/namespace pairs used for prompt instructions."""
        ...


def _merge_prefix_bindings_from_graph(
    merged: dict[str, str],
    graph: RDFGraph,
    extra_prefix: str | None,
    extra_namespace: str | None,
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
    """Extend *prefix_map* with URI-tail aliases for common semantic namespace segments.

    When an ontology registers ``ns2`` → ``…/relations#``, this adds the alias
    ``relations`` → ``…/relations#`` so that LLM-invented prefixes like
    ``relations:P57`` can be repaired without knowing the opaque prefix name.
    """
    new = {
        uri.rstrip("#/").rsplit("/", 1)[-1].lower(): uri
        for uri in prefix_map.values()
        if uri.rstrip("#/").rsplit("/", 1)[-1].lower() in _SEMANTIC_SUFFIXES
    }
    new = {k: v for k, v in new.items() if k not in prefix_map}
    return {**prefix_map, **new} if new else prefix_map


def build_llm_prefix_map(
    primary: Ontology,
    supplemental: Iterable[Ontology] = (),
) -> dict[str, str]:
    """Collect namespace prefixes for LLM Turtle/JSON-LD ingest repair.

    Layers (first wins on prefix name conflicts after ingest vocabulary):
    1. ``prefix_lookup_for_ingest()`` (COMMON + WELL_KNOWN)
    2. Primary ontology graph bindings + implicit stems
    3. Supplemental ontology graphs (same extraction)
    4. Semantic URI-tail aliases (``relations``, ``concepts``, etc.)
    """
    merged = prefix_lookup_for_ingest()
    ontologies: list[Ontology] = [primary, *supplemental]
    for ontology in ontologies:
        if ontology.is_null():
            continue
        graph = ontology.graph
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
            extra_prefix=ontology.prefix or None,
            extra_namespace=ontology.namespace or None,
        )
    return _add_semantic_aliases(merged)


def known_prefixes_for_llm_parse(source: OntologyPromptSource) -> dict[str, str]:
    """Collect namespace prefixes for TTL/JSON-LD repair during LLM output parsing."""
    return build_llm_prefix_map(source.ontology_for_prefixes())


class UnitOntologyAccess:
    """Accessor for :class:`UnitOntologyState` (ontology map loop)."""

    __slots__ = ("_state",)

    def __init__(self, state: UnitOntologyState) -> None:
        self._state = state

    def effective_ontology_for_prompt(self) -> Ontology:
        return self._state.current_ontology or self._state.ontology_snapshot

    def ontology_for_prefixes(self) -> Ontology:
        return self.effective_ontology_for_prompt()

    def has_non_null_seed_snapshot(self) -> bool:
        return not self._state.ontology_snapshot.is_null()

    def domain_prefix_pairs(self) -> list[tuple[str, str]]:
        return extract_domain_prefix_pairs(self.effective_ontology_for_prompt())


class UnitFactsOntologyAccess:
    """Accessor for :class:`UnitFactsState`; facts prompts use snapshot context only."""

    __slots__ = ("_state",)

    def __init__(self, state: UnitFactsState) -> None:
        self._state = state

    def effective_ontology_for_prompt(self) -> Ontology:
        return self._state.ontology_snapshot

    def ontology_for_prefixes(self) -> Ontology:
        return self._state.ontology_snapshot

    def has_non_null_seed_snapshot(self) -> bool:
        return not self._state.ontology_snapshot.is_null()

    def domain_prefix_pairs(self) -> list[tuple[str, str]]:
        return extract_domain_prefix_pairs(self.effective_ontology_for_prompt())


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
        """Ontologies to version and persist (per-anchor artifacts)."""
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
