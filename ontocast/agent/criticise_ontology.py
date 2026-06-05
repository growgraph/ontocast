"""Ontology criticism agent.

This module provides functionality for analyzing and validating ontologies.
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import OntologyCritiqueReport, Suggestions
from ontocast.onto.ontology_access import ontology_access_for_unit_ontology
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.prompt.common import (
    system_preamble_ontology as system_preamble,
)
from ontocast.prompt.common import text_template
from ontocast.prompt.criticise_ontology import (
    intro_instruction,
    ontology_criteria,
    template_prompt,
)
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.prompt.web_grounding import persist_search_request, search_guidelines_for
from ontocast.tool import LLMTool
from ontocast.tool.atomic import AtomicToolBox

logger = logging.getLogger(__name__)


async def criticise_ontology(
    state: UnitOntologyState, tools: AtomicToolBox
) -> UnitOntologyState:
    """Critically analyze the ontology in the current content unit.

    Args:
        state: The current unit ontology state containing the ontology to analyze.
        tools: The toolbox instance providing utility functions.

    Returns:
        UnitOntologyState: Updated state with analysis results.
    """

    progress_info = state.get_content_unit_progress_string()
    logger.info(
        f"Ontology Critic for {progress_info}: visit {state.node_visits[WorkflowNode.CRITICISE_ONTOLOGY]}/{state.max_visits_per_node}"
    )

    if state.content_unit is None:
        state.status = Status.FAILED
        return state

    access = ontology_access_for_unit_ontology(state)
    current = access.effective_ontology_for_prompt()
    if current.is_null():
        raise ValueError(
            f"Null ontology cannot be criticised: {current.iri} is not a valid ontology"
        )

    profile = get_graph_format_profile(state.llm_graph_format)
    parser = PydanticOutputParser(pydantic_object=OntologyCritiqueReport)
    llm_tool: LLMTool = await tools.get_llm_tool(state.budget_tracker)

    ontology_chapter = profile.format_ontology_chapter(current.graph)

    text_chapter = text_template.format(text=state.content_unit.extraction_text)

    user_instruction = state.ontology_user_instruction
    external_evidence = state.external_evidence_text
    if external_evidence:
        state.mark_external_evidence_used(WorkflowNode.CRITICISE_ONTOLOGY)

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "intro_instruction",
            "ontology_criteria",
            "user_instruction",
            "ontology_chapter",
            "text_chapter",
            "external_evidence",
            "graph_format_instruction",
            "format_instructions",
        ],
    )

    graph_format_instruction = profile.critique_graph_instruction()
    web_search_enabled = tools.web_grounding_enabled_for_node(
        WorkflowNode.CRITICISE_ONTOLOGY
    )
    search_guidelines = search_guidelines_for(
        WorkflowNode.CRITICISE_ONTOLOGY, web_search_enabled
    )
    ontology_criteria_str = ontology_criteria
    if search_guidelines:
        ontology_criteria_str = f"{ontology_criteria}\n{search_guidelines}"

    try:
        critique: OntologyCritiqueReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "preamble": system_preamble,
                "intro_instruction": intro_instruction,
                "ontology_criteria": ontology_criteria_str,
                "text_chapter": text_chapter,
                "user_instruction": user_instruction,
                "ontology_chapter": ontology_chapter,
                "external_evidence": external_evidence,
                "graph_format_instruction": graph_format_instruction,
                "format_instructions": profile.format_instructions(
                    OntologyCritiqueReport,
                    web_search_enabled=web_search_enabled,
                ),
            },
            llm_graph_format=state.llm_graph_format,
        )
        persist_search_request(
            state,
            WorkflowNode.CRITICISE_ONTOLOGY,
            critique.external_evidence_request,
            web_search_enabled,
        )
        logger.info(
            f"Parsed critique report - success: {critique.success}, "
            f"score: {critique.score}, n fixes: {len(critique.actionable_ontology_fixes)}."
        )

        if critique.success or critique.score > 90:
            state.status = Status.SUCCESS
            state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.SUCCESS)
            logger.info("Ontology critique passed")
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStage.ONTOLOGY_CRITIQUE
            state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.FAILED)
            state.suggestions = Suggestions.from_critique_report(critique)
            state.failure_reason = "Ontology Critic suggests improvements"
            logger.info(
                f"Ontology critique failed: {critique.systemic_critique_summary}"
            )
        return state

    except Exception as e:
        logger.error(f"Failed to critique ontology: {str(e)}")
        state.set_failure(FailureStage.ONTOLOGY_CRITIQUE, str(e))
        state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.FAILED)
        return state
