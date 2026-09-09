"""Tests for the slim per-unit loop context (replaces per-unit AgentState deep copies)."""

import pytest
from rdflib import OWL, RDF, URIRef

from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState, BudgetTracker
from ontocast.stategraph.unit_context import UnitLoopContext

pytestmark = pytest.mark.unit


def _ontology(iri: str) -> Ontology:
    graph = RDFGraph()
    graph.add((URIRef(iri), RDF.type, OWL.Ontology))
    return Ontology(graph=graph, iri=iri)


def test_from_agent_state_projects_loop_fields() -> None:
    state = AgentState(
        ontology_context_mode=OntologyContextMode.FIXED_SINGLE_ONTOLOGY,
        ontology_selection_user_instruction="prefer chemistry",
        ontology_context_fixed_ontology_id="matsci",
    )
    unit_budget = BudgetTracker()
    context = UnitLoopContext.from_agent_state(state, unit_budget)

    assert context.ontology_context_mode == OntologyContextMode.FIXED_SINGLE_ONTOLOGY
    assert context.ontology_selection_user_instruction == "prefer chemistry"
    assert context.ontology_context_fixed_ontology_id == "matsci"
    assert context.budget_tracker is unit_budget
    assert context.retrieval_metrics == {}


def test_artifacts_are_shared_by_reference_not_copied() -> None:
    ontology = _ontology("https://example.com/onto")
    state = AgentState(reduced_ontology_artifacts=[ontology])
    context = UnitLoopContext.from_agent_state(state)

    artifacts = context.reduced_artifacts()
    assert len(artifacts) == 1
    # Identity, not equality: the whole point is avoiding the deep copy.
    assert artifacts[0] is ontology


def test_reduced_artifacts_fall_back_to_ontology_artifacts() -> None:
    ontology = _ontology("https://example.com/onto")
    state = AgentState(ontology_artifacts=[ontology])
    context = UnitLoopContext.from_agent_state(state)
    assert [o.iri for o in context.reduced_artifacts()] == ["https://example.com/onto"]


def test_default_budget_tracker_is_document_tracker() -> None:
    state = AgentState()
    context = UnitLoopContext.from_agent_state(state)
    assert context.budget_tracker is state.budget_tracker


def test_per_unit_metrics_fold_back_and_do_not_leak_between_units() -> None:
    state = AgentState()
    context_a = UnitLoopContext.from_agent_state(state, BudgetTracker())
    context_b = UnitLoopContext.from_agent_state(state, BudgetTracker())

    context_a.retrieval_metrics["empty_snapshot_reason"] = "unit a reason"
    context_b.retrieval_metrics["ontology_context_mode"] = "selected_single_ontology"

    # Isolation: unit A's writes are invisible to unit B's context.
    assert "empty_snapshot_reason" not in context_b.retrieval_metrics
    assert context_a.retrieval_metrics != context_b.retrieval_metrics

    # Fold-back mirrors the fan-out reduce step.
    state.retrieval_metrics.update(context_a.retrieval_metrics)
    state.retrieval_metrics.update(context_b.retrieval_metrics)
    assert state.retrieval_metrics["empty_snapshot_reason"] == "unit a reason"
    assert (
        state.retrieval_metrics["ontology_context_mode"] == "selected_single_ontology"
    )


def test_budget_rebind_shares_metrics_with_original_context() -> None:
    """The in-loop shallow rebind must keep metrics visible to the caller."""
    state = AgentState()
    context = UnitLoopContext.from_agent_state(state, BudgetTracker())
    loop_tracker = BudgetTracker()
    rebound = context.model_copy(update={"budget_tracker": loop_tracker})

    rebound.retrieval_metrics["patch_retrieval"] = {"atoms_final": 3}
    assert context.retrieval_metrics["patch_retrieval"] == {"atoms_final": 3}
    assert rebound.budget_tracker is loop_tracker
