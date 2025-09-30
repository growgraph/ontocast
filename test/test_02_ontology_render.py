import pytest
from packaging.version import Version

from ontocast.agent import criticise_ontology, render_onto_triples
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.state import AgentState


def test_agent_text_to_ontology_fresh(
    agent_state_select_ontology_null: AgentState, apple_report: dict, tools
):
    """here no relevant ontology is present, we are trying to create a new one"""
    state = render_onto_triples(state=agent_state_select_ontology_null, tools=tools)

    assert state.ontology_addendum.iri is not None
    assert state.ontology_addendum.title is not None
    assert state.ontology_addendum.ontology_id is not None
    assert state.ontology_addendum.description is not None
    assert state.ontology_addendum.iri.startswith(state.current_domain)
    assert len(state.ontology_addendum.graph) > 0
    assert Version(state.ontology_addendum.version) >= Version("0.0.0")
    state.serialize("test/data/state_onto_addendum.json")


def test_agent_render_ontology(
    state_ontology_selected: AgentState, tools, state_ontology_rendered_filename
):
    state = state_ontology_selected
    state.status = Status.FAILED

    state = render_onto_triples(state=state, tools=tools)
    assert state.ontology_addendum.iri is not None
    assert state.ontology_addendum.iri.startswith(state.current_domain)
    assert len(state.ontology_addendum.graph) > 0
    state.serialize(state_ontology_rendered_filename)


@pytest.mark.order(after="test_agent_render_ontology")
def test_state_onto_criticized(
    state_ontology_rendered: AgentState, tools, state_ontology_criticized_filename
):
    state = criticise_ontology(state_ontology_rendered, tools=tools)
    assert state.status == Status.FAILED
    assert state.failure_stage == FailureStages.ONTOLOGY_CRITIQUE
    assert len(state.ontology_addendum.graph) > 0
    assert len(state.current_ontology.graph) > 0
    state.clear_failure()
    state.serialize(state_ontology_criticized_filename)
