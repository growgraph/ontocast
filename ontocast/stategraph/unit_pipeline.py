"""Simplified single-unit agentic pipeline.

This module provides :func:`run_unit_pipeline`, a lightweight wrapper around
:func:`~ontocast.stategraph.atomic.ontology_loop` and
:func:`~ontocast.stategraph.atomic.facts_loop` that processes the entire input
as **one** content unit without chunking, normalization, or the full LangGraph
workflow.

The pipeline is self-contained: it accepts a raw ``AgentState`` (with
``raw_input`` populated) and handles document conversion internally, mirroring
the ``CONVERT_TO_MD`` first-node contract of the full
:func:`~ontocast.stategraph.create.create_agent_graph` workflow.

The loops run sequentially:

1. **Conversion**: raw bytes in ``raw_input`` are converted to text via
   :func:`~ontocast.agent.convert_document.convert_document`.
2. **Ontology loop** (if ``render_mode`` includes ontology): extracts / improves
   ontology from the input text.  The initial ontology context is guided by
   ``agent_state.ontology_context_mode`` via the standard
   :func:`~ontocast.stategraph.context_resolver.resolve_unit_ontology_context`
   call inside the loop.
3. **Facts loop** (if ``render_mode`` includes facts): extracts facts from the
   input text. When the ontology loop ran, its ``current_ontology`` is passed as
   pre-resolved context so facts reuse that output instead of re-querying the
   catalog or triple store.
"""

import logging
from copy import deepcopy

from ontocast.agent.convert_document import convert_document
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.atomic import facts_loop, ontology_loop
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


class DocumentConversionError(Exception):
    """Raised by :func:`run_unit_pipeline` when document conversion fails."""

    def __init__(self, reason: str, stage: str | None = None) -> None:
        super().__init__(reason)
        self.stage = stage


async def run_unit_pipeline(
    agent_state: AgentState,
    tools: ToolBox,
) -> tuple[UnitOntologyState | None, UnitFactsState | None]:
    """Run conversion, ontology, and facts loops for a single content unit.

    Accepts a raw ``AgentState`` with ``raw_input`` set and handles document
    conversion internally as its first step, matching the contract of the full
    :func:`~ontocast.stategraph.create.create_agent_graph` workflow.

    Args:
        agent_state: Fully configured agent state with ``raw_input`` populated.
            ``render_mode``, ``ontology_context_mode``,
            ``ontology_user_instruction``, ``facts_user_instruction``, and
            budget/visit settings are all read from this state.
        tools: Configured tool-box.

    Returns:
        A ``(onto_result, facts_result)`` tuple.  Either element is ``None``
        when the corresponding loop was skipped based on ``render_mode``.

    Raises:
        DocumentConversionError: If document conversion fails.
    """
    convert_document(agent_state, tools)
    if agent_state.failure_stage is not None or agent_state.status == Status.FAILED:
        raise DocumentConversionError(
            agent_state.failure_reason or "Document conversion failed",
            stage=str(agent_state.failure_stage),
        )

    full_text = (
        agent_state.docling_doc.export_to_markdown()
        if agent_state.docling_doc is not None
        else ""
    )
    unit = ContentUnit(
        text=full_text,
        index=0,
        doc_iri=agent_state.doc_iri,
    )
    agent_state.content_units = [unit]

    onto_result: UnitOntologyState | None = None
    facts_result: UnitFactsState | None = None

    max_visits = agent_state.max_visits

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
            llm_graph_format=agent_state.llm_graph_format,
        )
        logger.info("run_unit_pipeline: starting ontology loop")
        onto_result = await ontology_loop(ontology_state, tools, agent_state)
        logger.info(
            "run_unit_pipeline: ontology loop finished (status=%s)", onto_result.status
        )
        agent_state.budget_tracker = onto_result.budget_tracker
        if not onto_result.current_ontology.is_null():
            agent_state.reduced_ontology_artifacts = [onto_result.current_ontology]

    facts_pre_resolved_context: UnitOntologyContext | None = None
    if onto_result is not None and not onto_result.current_ontology.is_null():
        facts_pre_resolved_context = UnitOntologyContext(
            anchor_iri=onto_result.assembly_anchor_iri,
            ontology_snapshot=onto_result.current_ontology,
            patch_sources=list(onto_result.ontology_patch_sources),
            assembly_mode=onto_result.assembly_mode_used,
        )

    if agent_state.render_facts:
        facts_state = UnitFactsState(
            content_unit=unit,
            ontology_snapshot=NULL_ONTOLOGY,
            ontology_patch_sources=[],
            facts_user_instruction=agent_state.facts_user_instruction,
            budget_tracker=deepcopy(agent_state.budget_tracker),
            max_visits_per_node=max_visits,
            llm_graph_format=agent_state.llm_graph_format,
        )
        logger.info("run_unit_pipeline: starting facts loop")
        facts_result = await facts_loop(
            facts_state,
            tools,
            agent_state,
            pre_resolved_context=facts_pre_resolved_context,
        )
        logger.info(
            "run_unit_pipeline: facts loop finished (status=%s)", facts_result.status
        )
        agent_state.budget_tracker = facts_result.budget_tracker

    return onto_result, facts_result
