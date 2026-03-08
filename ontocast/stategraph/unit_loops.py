"""Reusable per-unit loops for parallel map/reduce execution."""

import logging

from ontocast.agent.criticise_facts import criticise_facts
from ontocast.agent.criticise_ontology import criticise_ontology
from ontocast.agent.render_facts import render_facts
from ontocast.agent.render_ontology import render_ontology
from ontocast.onto.enum import Status, WorkflowNode
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


async def unit_facts_loop(
    state: UnitFactsState, tools: ToolBox, max_retries: int | None = None
) -> UnitFactsState:
    """Run facts render/critic loop for one content unit.

    Ontology is selected once per document in the main workflow; ontology_snapshot
    is always provided by the caller.
    """
    unit_state = state.model_copy(deep=True)
    retries = max(1, max_retries or unit_state.max_retries)

    for attempt in range(1, retries + 1):
        unit_state.node_visits[WorkflowNode.TEXT_TO_FACTS] = attempt - 1
        unit_state.node_visits[WorkflowNode.CRITICISE_FACTS] = attempt - 1

        unit_state = await render_facts(unit_state, tools)
        if unit_state.status != Status.SUCCESS:
            logger.info(f"Unit facts render failed at attempt {attempt}/{retries}")
            continue

        unit_state = await criticise_facts(unit_state, tools)
        if unit_state.status == Status.SUCCESS:
            logger.info(f"Unit facts loop converged at attempt {attempt}/{retries}")
            return unit_state

    logger.info("Unit facts loop exhausted retries")
    return unit_state


async def unit_ontology_loop(
    state: UnitOntologyState, tools: ToolBox, max_retries: int | None = None
) -> UnitOntologyState:
    """Run ontology render/critic loop for one content unit.

    Ontology is selected once per document in the main workflow; ontology_snapshot
    is always provided by the caller (may be null for fresh-ontology builds).
    """
    unit_state = state.model_copy(deep=True)
    retries = max(1, max_retries or unit_state.max_retries)

    for attempt in range(1, retries + 1):
        unit_state.node_visits[WorkflowNode.TEXT_TO_ONTOLOGY] = attempt - 1
        unit_state.node_visits[WorkflowNode.CRITICISE_ONTOLOGY] = attempt - 1

        unit_state = await render_ontology(unit_state, tools)
        if unit_state.status != Status.SUCCESS:
            logger.info(f"Unit ontology render failed at attempt {attempt}/{retries}")
            continue

        unit_state = await criticise_ontology(unit_state, tools)
        if unit_state.status == Status.SUCCESS:
            logger.info(f"Unit ontology loop converged at attempt {attempt}/{retries}")
            return unit_state

    logger.info("Unit ontology loop exhausted retries")
    return unit_state
