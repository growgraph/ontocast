"""Slim per-unit view of document state for the map-stage loops.

The unit render/critic loops only read a handful of document-level fields.
Passing the full :class:`~ontocast.onto.state.AgentState` required a deep copy
per unit (raw input bytes, docling document, every content unit, every graph),
which dominated fan-out cost. ``UnitLoopContext`` carries exactly what the
loops consume; large artifacts are shared by reference (read-only in the
loops).
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.model import BasePydanticModel
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.state import AgentState, BudgetTracker


class UnitLoopContext(BasePydanticModel):
    """Document-level inputs for one unit's ontology/facts loop.

    ``reduced_ontology_artifacts`` is shared by reference with the document
    state — the loops only read it. ``retrieval_metrics`` is per-unit and is
    folded back into the document state after the fan-out gathers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ontology_context_mode: OntologyContextMode
    ontology_selection_user_instruction: str = ""
    ontology_context_fixed_ontology_id: str = ""
    reduced_ontology_artifacts: list[Ontology] = Field(default_factory=list)
    budget_tracker: BudgetTracker = Field(default_factory=BudgetTracker)
    retrieval_metrics: dict[str, int | float | str | dict[str, Any]] = Field(
        default_factory=dict
    )

    @classmethod
    def from_agent_state(
        cls,
        state: AgentState,
        budget_tracker: BudgetTracker | None = None,
    ) -> UnitLoopContext:
        """Project the loop-relevant fields off a document state.

        Args:
            state: Document-level agent state.
            budget_tracker: Per-unit tracker LLM calls should be charged to;
                defaults to the document tracker (sequential pipelines).
        """
        return cls(
            ontology_context_mode=state.ontology_context_mode,
            ontology_selection_user_instruction=(
                state.ontology_selection_user_instruction
            ),
            ontology_context_fixed_ontology_id=(
                state.ontology_context_fixed_ontology_id
            ),
            reduced_ontology_artifacts=document_ontology_access(
                state
            ).reduced_artifacts(),
            budget_tracker=budget_tracker
            if budget_tracker is not None
            else state.budget_tracker,
        )

    def reduced_artifacts(self) -> list[Ontology]:
        """Reduced document ontology artifacts (shared, read-only)."""
        return list(self.reduced_ontology_artifacts)
