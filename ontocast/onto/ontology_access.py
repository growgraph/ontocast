"""Read-only accessors for ontology context on document vs unit workflow state.

Centralizes prompt-effective ontology resolution and serialization target lists
so agents and stategraph code do not duplicate ``current_ontology`` /
``ontology_snapshot`` / ``ontology_artifacts`` branching.
"""

from typing import Protocol

from ontocast.onto.ontology import Ontology
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState


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


class DocumentOntologyAccess:
    """Accessor for :class:`AgentState` (document-level reduce / serialize)."""

    __slots__ = ("_state",)

    def __init__(self, state: AgentState) -> None:
        self._state = state

    def primary_ontology(self) -> Ontology:
        """Working ontology for merge, consolidation, and single-graph consumers."""
        return self._state.current_ontology

    def is_primary_null(self) -> bool:
        return self._state.current_ontology.is_null()

    def serialization_targets(self) -> list[Ontology]:
        """Ontologies to version and persist (per-anchor artifacts or primary)."""
        if self._state.ontology_artifacts:
            return list(self._state.ontology_artifacts)
        return [self._state.current_ontology]


def ontology_access_for_unit_ontology(state: UnitOntologyState) -> UnitOntologyAccess:
    return UnitOntologyAccess(state)


def ontology_access_for_unit_facts(state: UnitFactsState) -> UnitFactsOntologyAccess:
    return UnitFactsOntologyAccess(state)


def document_ontology_access(state: AgentState) -> DocumentOntologyAccess:
    return DocumentOntologyAccess(state)
