"""Reusable per-unit render/critic retry loops.

These loops are designed for map/reduce execution where each content unit
is processed independently. They deep-copy the incoming unit state, then run
render -> critic until success or retry exhaustion.

Minimal tools contract:
- A ``ToolBox`` instance is still expected by type, but the loop itself only
  relies on downstream agent calls that resolve an LLM via
  ``tools.get_llm_tool(state.budget_tracker)``.
- No triple-store, chunker, converter, or aggregator capabilities are required
  for these atomic loops.
"""

import logging

from ontocast.agent.criticise_facts import criticise_facts
from ontocast.agent.criticise_ontology import criticise_ontology
from ontocast.agent.external_evidence import (
    fetch_external_evidence_for_node,
    plan_external_evidence_for_node,
)
from ontocast.agent.render_facts import render_facts
from ontocast.agent.render_ontology import render_ontology
from ontocast.onto.enum import Status, WorkflowNode
from ontocast.onto.model import ExternalEvidenceCacheEntry, ExternalEvidenceRequest
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.tool.atomic import AtomicToolBox

logger = logging.getLogger(__name__)


def _resolve_max_visits_limit(state_visits: int, override: int | None) -> int:
    """Return a safe visit limit while respecting explicit overrides."""
    visits = state_visits if override is None else override
    return max(1, visits)


def _reset_node_evidence_context(
    state: UnitFactsState | UnitOntologyState, node: WorkflowNode
) -> None:
    """Start node execution in no-search mode with empty evidence context."""
    state.set_external_evidence_request(node, ExternalEvidenceRequest())
    state.set_external_evidence_cache_entry(node, ExternalEvidenceCacheEntry())
    state.load_external_evidence_for_node(node)


async def facts_loop(
    state: UnitFactsState, tools: AtomicToolBox, max_visits_per_node: int | None = None
) -> UnitFactsState:
    """Run facts render/critic loop for one content unit.

    Ontology is selected once per document in the main workflow; ontology_snapshot
    is always provided by the caller.
    """
    unit_state = state.model_copy(deep=True)
    max_visits = _resolve_max_visits_limit(
        unit_state.max_visits_per_node, max_visits_per_node
    )
    unit_state.max_visits_per_node = max_visits

    for render_attempt in range(1, max_visits + 1):
        unit_state.node_visits[WorkflowNode.TEXT_TO_FACTS] += 1
        _reset_node_evidence_context(unit_state, WorkflowNode.TEXT_TO_FACTS)
        unit_state = await render_facts(unit_state, tools)
        if unit_state.status != Status.SUCCESS:
            render_request = unit_state.get_external_evidence_request(
                WorkflowNode.TEXT_TO_FACTS
            )
            if render_request.initiate_search:
                unit_state = await plan_external_evidence_for_node(
                    unit_state, tools, WorkflowNode.TEXT_TO_FACTS
                )
                unit_state = await fetch_external_evidence_for_node(
                    unit_state, tools, WorkflowNode.TEXT_TO_FACTS
                )
                unit_state = await render_facts(unit_state, tools)
                if unit_state.status == Status.SUCCESS:
                    logger.info(
                        "Unit facts render recovered with search at attempt %s/%s",
                        render_attempt,
                        max_visits,
                    )
                    # Continue to critic tier below.
                else:
                    logger.info(
                        "Unit facts render failed at attempt %s/%s (with search)",
                        render_attempt,
                        max_visits,
                    )
                    continue
            else:
                logger.info(
                    "Unit facts render failed at attempt %s/%s (no search request)",
                    render_attempt,
                    max_visits,
                )
                continue

        for critic_attempt in range(1, max_visits + 1):
            unit_state.node_visits[WorkflowNode.CRITICISE_FACTS] += 1
            _reset_node_evidence_context(unit_state, WorkflowNode.CRITICISE_FACTS)
            unit_state = await criticise_facts(unit_state, tools)
            if unit_state.status == Status.SUCCESS:
                logger.info(
                    "Unit facts loop converged at render %s/%s critic %s/%s",
                    render_attempt,
                    max_visits,
                    critic_attempt,
                    max_visits,
                )
                return unit_state

            critic_request = unit_state.get_external_evidence_request(
                WorkflowNode.CRITICISE_FACTS
            )
            if not critic_request.initiate_search:
                logger.info(
                    "Unit facts critic failed at render %s/%s critic %s/%s without search request",
                    render_attempt,
                    max_visits,
                    critic_attempt,
                    max_visits,
                )
                break

            unit_state = await plan_external_evidence_for_node(
                unit_state, tools, WorkflowNode.CRITICISE_FACTS
            )
            unit_state = await fetch_external_evidence_for_node(
                unit_state, tools, WorkflowNode.CRITICISE_FACTS
            )
            unit_state = await criticise_facts(unit_state, tools)
            if unit_state.status == Status.SUCCESS:
                logger.info(
                    "Unit facts loop converged with critic search at render %s/%s critic %s/%s",
                    render_attempt,
                    max_visits,
                    critic_attempt,
                    max_visits,
                )
                return unit_state

            continue

    logger.info("Unit facts loop exhausted retries")
    return unit_state


async def ontology_loop(
    state: UnitOntologyState,
    tools: AtomicToolBox,
    max_visits_per_node: int | None = None,
) -> UnitOntologyState:
    """Run ontology render/critic loop for one content unit.

    Ontology is selected once per document in the main workflow; ontology_snapshot
    is always provided by the caller (may be null for fresh-ontology builds).
    """
    unit_state = state.model_copy(deep=True)
    max_visits = _resolve_max_visits_limit(
        unit_state.max_visits_per_node, max_visits_per_node
    )
    unit_state.max_visits_per_node = max_visits

    for render_attempt in range(1, max_visits + 1):
        unit_state.node_visits[WorkflowNode.TEXT_TO_ONTOLOGY] += 1
        _reset_node_evidence_context(unit_state, WorkflowNode.TEXT_TO_ONTOLOGY)
        unit_state = await render_ontology(unit_state, tools)
        if unit_state.status != Status.SUCCESS:
            render_request = unit_state.get_external_evidence_request(
                WorkflowNode.TEXT_TO_ONTOLOGY
            )
            if render_request.initiate_search:
                unit_state = await plan_external_evidence_for_node(
                    unit_state, tools, WorkflowNode.TEXT_TO_ONTOLOGY
                )
                unit_state = await fetch_external_evidence_for_node(
                    unit_state, tools, WorkflowNode.TEXT_TO_ONTOLOGY
                )
                unit_state = await render_ontology(unit_state, tools)
                if unit_state.status == Status.SUCCESS:
                    logger.info(
                        "Unit ontology render recovered with search at attempt %s/%s",
                        render_attempt,
                        max_visits,
                    )
                else:
                    logger.info(
                        "Unit ontology render failed at attempt %s/%s (with search)",
                        render_attempt,
                        max_visits,
                    )
                    continue
            else:
                logger.info(
                    "Unit ontology render failed at attempt %s/%s (no search request)",
                    render_attempt,
                    max_visits,
                )
                continue

        for critic_attempt in range(1, max_visits + 1):
            unit_state.node_visits[WorkflowNode.CRITICISE_ONTOLOGY] += 1
            _reset_node_evidence_context(unit_state, WorkflowNode.CRITICISE_ONTOLOGY)
            unit_state = await criticise_ontology(unit_state, tools)
            if unit_state.status == Status.SUCCESS:
                logger.info(
                    "Unit ontology loop converged at render %s/%s critic %s/%s",
                    render_attempt,
                    max_visits,
                    critic_attempt,
                    max_visits,
                )
                return unit_state

            critic_request = unit_state.get_external_evidence_request(
                WorkflowNode.CRITICISE_ONTOLOGY
            )
            if not critic_request.initiate_search:
                logger.info(
                    "Unit ontology critic failed at render %s/%s critic %s/%s without search request",
                    render_attempt,
                    max_visits,
                    critic_attempt,
                    max_visits,
                )
                break

            unit_state = await plan_external_evidence_for_node(
                unit_state, tools, WorkflowNode.CRITICISE_ONTOLOGY
            )
            unit_state = await fetch_external_evidence_for_node(
                unit_state, tools, WorkflowNode.CRITICISE_ONTOLOGY
            )
            unit_state = await criticise_ontology(unit_state, tools)
            if unit_state.status == Status.SUCCESS:
                logger.info(
                    "Unit ontology loop converged with critic search at render %s/%s critic %s/%s",
                    render_attempt,
                    max_visits,
                    critic_attempt,
                    max_visits,
                )
                return unit_state

    logger.info("Unit ontology loop exhausted retries")
    return unit_state
