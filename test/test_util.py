from ontocast.onto.enum import WorkflowNode
from ontocast.onto.state import AgentState
from ontocast.stategraph.util import count_visits_conditional_success, wrap_with


def test_wrap_with_basic_functionality():
    """Test basic wrapping functionality with a simple function."""

    def simple_func(state: AgentState) -> AgentState:
        state.set_text("test")
        return state

    node_name = WorkflowNode.TEXT_TO_ONTOLOGY
    node_name, wrapped_func = wrap_with(
        simple_func, node_name, count_visits_conditional_success
    )

    # Initialize state
    state = AgentState()

    # Execute wrapped function
    result = wrapped_func(state)

    # Check results
    assert result.input_text == "test"
    assert result.node_visits[node_name] == 1
    assert node_name == WorkflowNode.TEXT_TO_ONTOLOGY


def test_wrap_with_multiple_calls():
    """Test that visit counter increments correctly with multiple calls."""

    def increment_func(state: AgentState) -> AgentState:
        state.set_text(str(state.node_visits.get(WorkflowNode.TEXT_TO_ONTOLOGY, 0)))
        return state

    node_name = WorkflowNode.TEXT_TO_ONTOLOGY
    node_name, wrapped_func = wrap_with(
        increment_func, node_name, count_visits_conditional_success
    )

    # Initialize state
    state = AgentState()

    # Execute wrapped function multiple times
    for i in range(3):
        result = wrapped_func(state)
        state = result

    # Check results
    assert (
        result.input_text == "2"
    )  # Because we start at 0 and increment after each call
    assert result.node_visits[node_name] == 3


def test_wrap_with_different_node_types():
    """Test wrapping with different workflow node types."""

    def test_func(state: AgentState) -> AgentState:
        return state

    # Test with different node types
    test_nodes = [
        WorkflowNode.TEXT_TO_ONTOLOGY,
        WorkflowNode.TEXT_TO_FACTS,
        WorkflowNode.CRITICISE_ONTOLOGY,
    ]

    for node in test_nodes:
        node_name, wrapped_func = wrap_with(
            test_func, node, count_visits_conditional_success
        )

        state = AgentState()

        result = wrapped_func(state)
        assert result.node_visits[node] == 1
        assert node_name == node
