from functools import wraps
from typing import Callable

from ontocast.onto.enum import Status, WorkflowNode
from ontocast.onto.state import AgentState
from ontocast.util import logger


def count_visits_conditional_success(state: AgentState, current_node) -> AgentState:
    """Track node visits and handle success/failure conditions.

    This function increments the visit counter for a node and manages the state
    based on success/failure conditions and maximum visit limits.

    Args:
        state: The current agent state.
        current_node: The node being visited.

    Returns:
        AgentState: Updated agent state after processing visit conditions.
    """
    state.node_visits[current_node] += 1
    if state.status == Status.SUCCESS:
        logger.info(f"For {current_node}: status is SUCCESS, proceeding to next node")
        state.clear_failure()
    elif state.node_visits[current_node] >= state.max_visits:
        logger.error(f"For {current_node}: maximum visits exceeded")
        state.set_failure(current_node, reason="Maximum visits exceeded")
        state.status = Status.SUCCESS
    return state


def wrap_with(func, node_name, post_func) -> tuple[WorkflowNode, Callable]:
    """Add a visit counter to a function.

    This function wraps a given function with logging and post-processing
    functionality, typically used for workflow node execution.

    Args:
        func: The function to wrap.
        node_name: The name of the node.
        post_func: Function to execute after the main function.

    Returns:
        tuple[WorkflowNode, Callable]: A tuple containing the node name and
            the wrapped function.
    """

    @wraps(func)
    def wrapper(state: AgentState):
        logger.info(f"Starting to execute {node_name}")
        state = func(state)
        state = post_func(state, node_name)
        return state

    return node_name, wrapper
