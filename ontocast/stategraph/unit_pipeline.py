"""Simplified single-unit agentic pipeline.

This module provides :func:`run_unit_pipeline`, a lightweight wrapper around
:func:`~ontocast.stategraph.atomic.ontology_loop` and
:func:`~ontocast.stategraph.atomic.facts_loop` that processes the entire input
as **one** content unit without chunking, normalization, or the full LangGraph
workflow.

The loops run sequentially:

1. **Ontology loop** (if ``render_mode`` includes ontology): extracts / improves
   ontology from the input text.  The initial ontology context is guided by
   ``agent_state.ontology_context_mode`` via the standard
   :func:`~ontocast.stategraph.context_resolver.resolve_unit_ontology_context`
   call inside the loop.
2. **Facts loop** (if ``render_mode`` includes facts): extracts facts from the
   input text.  When the ontology loop ran first, its output
   (``onto_result.current_ontology``) is injected directly as the facts
   ontology snapshot so that fact extraction immediately benefits from the
   freshly-generated ontology without a store round-trip.  When the ontology
   loop is skipped, ``ontology_context_mode`` guides the normal context
   resolution path inside :func:`~ontocast.stategraph.atomic.facts_loop`.
"""

import logging
from copy import deepcopy

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.atomic import facts_loop, ontology_loop
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


async def run_unit_pipeline(
    agent_state: AgentState,
    tools: ToolBox,
) -> tuple[UnitOntologyState | None, UnitFactsState | None]:
    """Run ontology and facts loops for a single content unit.

    The caller must have already set ``agent_state.input_text`` (e.g. by
    calling :func:`~ontocast.agent.convert_document.convert_document`).

    Args:
        agent_state: Fully configured agent state with ``input_text`` set.
            ``render_mode``, ``ontology_context_mode``,
            ``ontology_user_instruction``, ``facts_user_instruction``, and
            budget/visit settings are all read from this state.
        tools: Configured tool-box.

    Returns:
        A ``(onto_result, facts_result)`` tuple.  Either element is ``None``
        when the corresponding loop was skipped based on ``render_mode``.
    """
    if not agent_state.input_text:
        raise ValueError(
            "agent_state.input_text must be set before calling run_unit_pipeline"
        )

    unit = ContentUnit(
        text=agent_state.input_text,
        index=0,
        doc_iri=agent_state.doc_iri,
    )
    agent_state.content_units = [unit]

    onto_result: UnitOntologyState | None = None
    facts_result: UnitFactsState | None = None

    max_visits = tools.config.server.max_visits_per_node

    if agent_state.render_ontology:
        ontology_state = UnitOntologyState(
            content_unit=unit,
            ontology_snapshot=NULL_ONTOLOGY,
            ontology_patch_sources=[],
            ontology_user_instruction=agent_state.ontology_user_instruction,
            budget_tracker=deepcopy(agent_state.budget_tracker),
            max_visits_per_node=max_visits,
            current_domain=agent_state.current_domain,
            ontology_max_triples=tools.config.server.ontology_max_triples,
        )
        logger.info("run_unit_pipeline: starting ontology loop")
        onto_result = await ontology_loop(ontology_state, tools, agent_state)
        logger.info(
            "run_unit_pipeline: ontology loop finished (status=%s)", onto_result.status
        )
        agent_state.budget_tracker = onto_result.budget_tracker

    if agent_state.render_facts:
        facts_state = UnitFactsState(
            content_unit=unit,
            ontology_snapshot=NULL_ONTOLOGY,
            ontology_patch_sources=[],
            facts_user_instruction=agent_state.facts_user_instruction,
            budget_tracker=deepcopy(agent_state.budget_tracker),
            max_visits_per_node=max_visits,
        )
        logger.info("run_unit_pipeline: starting facts loop")
        if onto_result is not None:
            facts_result = await facts_loop(
                facts_state,
                tools,
                agent_state,
                pre_resolved_ontology=onto_result.current_ontology,
            )
        else:
            facts_result = await facts_loop(facts_state, tools, agent_state)
        logger.info(
            "run_unit_pipeline: facts loop finished (status=%s)", facts_result.status
        )
        agent_state.budget_tracker = facts_result.budget_tracker

    return onto_result, facts_result
