import asyncio
import logging
from functools import wraps
from typing import Callable

from ontocast.onto.enum import Status, WorkflowNode
from ontocast.onto.state import AgentState

logger = logging.getLogger(__name__)


def count_visits_conditional_success(
    state: AgentState, current_node: WorkflowNode
) -> AgentState:
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
        logger.info(f"For {current_node}: maximum visits exceeded")
        # Don't set failure stage since we're continuing with SUCCESS status
        # Just log the reason and continue
        state.failure_reason = f"Maximum visits exceeded for {current_node}"
        state.status = Status.SUCCESS
    return state


def wrap_with(func, node_name, post_func) -> tuple[WorkflowNode, Callable]:
    """Add a visit counter to a function.

    This function wraps a given function with logging and post-processing
    functionality, typically used for workflow node execution.

    Args:
        func: The function to wrap (can be sync or async).
        node_name: The name of the node.
        post_func: Function to execute after the main function.

    Returns:
        tuple[WorkflowNode, Callable]: A tuple containing the node name and
            the wrapped function.
    """
    # Check if the function is async
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(state: AgentState):
            logger.info(f"Starting to execute {node_name}")
            state = await func(state)
            state = post_func(state, node_name)
            return state

        return node_name, async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(state: AgentState):
            logger.info(f"Starting to execute {node_name}")
            state = func(state)
            state = post_func(state, node_name)
            return state

        return node_name, sync_wrapper
