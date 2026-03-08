"""Dedicated state models for parallel unit loops."""

from collections import defaultdict
from copy import deepcopy

from pydantic import Field

from ontocast.onto.constants import DEFAULT_DOMAIN
from ontocast.onto.content_unit import ContentUnit, SourceUnit
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import BasePydanticModel, Suggestions
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState, BudgetTracker


def _render_updated_graph(
    graph: RDFGraph, updates: list[GraphUpdate], max_triples: int | None = None
) -> tuple[RDFGraph, bool]:
    """Apply GraphUpdate objects to a graph. Delegates to AgentState implementation."""
    return AgentState.render_updated_graph(graph, updates, max_triples=max_triples)


class UnitState(BasePydanticModel):
    """Common per-unit workflow state."""

    ontology_snapshot: Ontology = Field(description="Immutable ontology snapshot")
    suggestions: Suggestions = Field(default_factory=Suggestions)
    budget_tracker: BudgetTracker = Field(default_factory=BudgetTracker)
    max_retries: int = Field(default=3, ge=1)

    status: Status = Field(default=Status.NOT_VISITED)
    failure_stage: FailureStage | None = Field(default=None)
    failure_reason: str | None = Field(default=None)
    node_visits: dict[WorkflowNode, int] = Field(
        default_factory=lambda: defaultdict(int),
    )

    def get_content_unit_progress_string(self) -> str:
        """Progress string for logging (single unit context)."""
        return "content unit 1/1"

    def set_node_status(self, node: WorkflowNode, status: Status) -> None:
        """Set workflow node status (for logging)."""
        self.status = status

    def set_failure(self, stage: FailureStage, reason: str) -> None:
        """Record failure stage and reason."""
        self.failure_stage = stage
        self.failure_reason = reason
        self.status = Status.FAILED

    def clear_failure(self) -> None:
        """Clear failure state."""
        self.failure_stage = None
        self.failure_reason = None


class UnitFactsState(UnitState):
    """Independent per-unit state for facts extraction and critique."""

    content_unit: ContentUnit = Field(description="Unit under processing (mutable)")
    facts_user_instruction: str = Field(default="")
    facts_updates: list[GraphUpdate] = Field(default_factory=list)

    def update_facts(self) -> None:
        """Apply facts_updates to content_unit.graph and clear the list."""
        if not self.facts_updates:
            return
        updated_graph, _ = _render_updated_graph(
            self.content_unit.graph, self.facts_updates, max_triples=None
        )
        self.content_unit.graph = updated_graph
        self.facts_updates = []


class UnitOntologyState(UnitState):
    """Independent per-unit state for ontology improvement loop."""

    content_unit: SourceUnit = Field(description="Unit under processing")
    ontology_user_instruction: str = Field(default="")
    current_ontology: Ontology = Field(
        default_factory=Ontology, description="Current ontology under refinement"
    )
    ontology_updates: list[GraphUpdate] = Field(default_factory=list)
    ontology_updates_applied: list[GraphUpdate] = Field(default_factory=list)
    current_domain: str = Field(default=DEFAULT_DOMAIN)
    ontology_max_triples: int | None = Field(default=None)

    def model_post_init(self, __context) -> None:
        """Initialize mutable ontology state from immutable snapshot."""
        self.current_ontology = deepcopy(self.ontology_snapshot)

    @property
    def all_updates(self) -> list[GraphUpdate]:
        """All ontology updates produced by this unit (applied and pending)."""
        return [*self.ontology_updates_applied, *self.ontology_updates]

    def update_ontology(self) -> None:
        """Apply ontology_updates to current_ontology and clear the list."""
        if not self.ontology_updates:
            return
        updated_graph, was_applied = _render_updated_graph(
            self.current_ontology.graph,
            self.ontology_updates,
            max_triples=self.ontology_max_triples,
        )
        if not was_applied:
            return

        updated_ontology = deepcopy(self.current_ontology)
        updated_ontology.graph = updated_graph
        if self.current_ontology.hash:
            updated_ontology.parent_hashes = [self.current_ontology.hash]
        else:
            updated_ontology.parent_hashes = []
        if not updated_ontology.created_at:
            from datetime import datetime, timezone

            updated_ontology.created_at = datetime.now(timezone.utc)
        updated_ontology.hash = None
        updated_ontology._compute_and_set_hash()
        if not updated_ontology.hash and updated_ontology.parent_hashes:
            updated_ontology.hash = updated_ontology.parent_hashes[0]
        updated_ontology.sync_properties_to_graph()
        self.ontology_updates_applied += self.ontology_updates
        self.current_ontology = updated_ontology
        self.ontology_updates = []
