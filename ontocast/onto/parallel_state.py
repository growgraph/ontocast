"""Dedicated state models for parallel unit loops."""

from copy import deepcopy

from pydantic import Field

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import FailureStage, Status
from ontocast.onto.model import BasePydanticModel, Suggestions
from ontocast.onto.ontology import Ontology
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState, BudgetTracker


class UnitFactsState(BasePydanticModel):
    """Independent per-unit state for facts extraction and critique."""

    content_unit: ContentUnit = Field(description="Unit under processing")
    ontology_snapshot: Ontology = Field(description="Immutable ontology snapshot")
    facts_user_instruction: str = Field(default="")
    suggestions: Suggestions = Field(default_factory=Suggestions)
    budget_tracker: BudgetTracker = Field(default_factory=BudgetTracker)
    max_retries: int = Field(default=3, ge=1)

    status: Status = Field(default=Status.NOT_VISITED)
    failure_stage: FailureStage | None = Field(default=None)
    failure_reason: str | None = Field(default=None)
    output_unit: ContentUnit | None = Field(default=None)

    def to_agent_state(self) -> AgentState:
        """Create isolated agent state for the unit loop execution."""
        return AgentState(
            current_chunk=deepcopy(self.content_unit),
            current_ontology=deepcopy(self.ontology_snapshot),
            facts_user_instruction=self.facts_user_instruction,
            suggestions=deepcopy(self.suggestions),
            budget_tracker=self.budget_tracker,
            max_visits=self.max_retries,
            chunks=[],
            chunks_processed=[],
            files={},
        )

    def apply_agent_result(self, result: AgentState) -> None:
        """Transfer execution result back to unit state."""
        self.status = result.status
        self.failure_stage = result.failure_stage
        self.failure_reason = result.failure_reason
        self.suggestions = result.suggestions
        if result.status == Status.SUCCESS:
            self.output_unit = result.current_chunk


class UnitOntologyState(BasePydanticModel):
    """Independent per-unit state for ontology improvement loop."""

    content_unit: ContentUnit = Field(description="Unit under processing")
    ontology_snapshot: Ontology = Field(description="Immutable ontology snapshot")
    ontology_user_instruction: str = Field(default="")
    suggestions: Suggestions = Field(default_factory=Suggestions)
    budget_tracker: BudgetTracker = Field(default_factory=BudgetTracker)
    max_retries: int = Field(default=3, ge=1)

    status: Status = Field(default=Status.NOT_VISITED)
    failure_stage: FailureStage | None = Field(default=None)
    failure_reason: str | None = Field(default=None)
    output_updates: list[GraphUpdate] = Field(default_factory=list)
    output_ontology: Ontology | None = Field(default=None)

    def to_agent_state(self) -> AgentState:
        """Create isolated agent state for the unit loop execution."""
        return AgentState(
            current_chunk=deepcopy(self.content_unit),
            current_ontology=deepcopy(self.ontology_snapshot),
            ontology_user_instruction=self.ontology_user_instruction,
            suggestions=deepcopy(self.suggestions),
            budget_tracker=self.budget_tracker,
            max_visits=self.max_retries,
            chunks=[],
            chunks_processed=[],
            files={},
        )

    def apply_agent_result(self, result: AgentState) -> None:
        """Transfer execution result back to unit state."""
        self.status = result.status
        self.failure_stage = result.failure_stage
        self.failure_reason = result.failure_reason
        self.suggestions = result.suggestions
        self.output_updates = [
            *result.ontology_updates_applied,
            *result.ontology_updates,
        ]
        if result.status == Status.SUCCESS:
            self.output_ontology = result.current_ontology
