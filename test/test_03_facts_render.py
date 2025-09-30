import pytest

from ontocast.agent import criticise_facts, render_facts, sublimate_ontology
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState


def test_render_facts(
    state_ontology_criticized: AgentState, tools, state_rendered_facts_filename
):
    state = state_ontology_criticized
    state = render_facts(state=state, tools=tools)

    assert len(state.current_chunk.graph) > 0
    assert state.status == Status.SUCCESS

    state.serialize(state_rendered_facts_filename)


@pytest.mark.order(after="test_render_facts")
def test_sublimate_ontology(
    state_rendered_facts: AgentState, tools, state_sublimated_filename
):
    state = state_rendered_facts
    state = sublimate_ontology(state=state, tools=tools)

    assert state.status == Status.SUCCESS
    assert state.failure_stage is None
    state.serialize(state_sublimated_filename)


@pytest.mark.order(after="test_sublimate")
def test_criticise_facts(
    state_sublimated: AgentState,
    tools,
    state_facts_failed_filename,
    state_facts_success_filename,
):
    state = state_sublimated
    state = criticise_facts(state=state, tools=tools)

    assert state.success_score > 0
    assert state.status == Status.FAILED

    state.serialize(state_facts_failed_filename)
    state.status = Status.SUCCESS
    state.clear_failure()
    state.serialize(state_facts_success_filename)


def test_render_facts_after_fail(state_facts_failed: AgentState, tools):
    state = state_facts_failed
    state = render_facts(state=state, tools=tools)

    assert len(state.current_chunk.graph) > 0
