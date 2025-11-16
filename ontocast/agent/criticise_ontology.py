"""Enhanced ontology criticism agent with SPARQL operations.

This module provides enhanced functionality for analyzing and validating ontologies of previous critiques and SPARQL operation support.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import OntologyCritiqueReport, Suggestions
from ontocast.onto.state import AgentState
from ontocast.prompt.common import ontology_template, text_template
from ontocast.prompt.common import system_preamble_ontology as system_preamble
from ontocast.prompt.criticise_ontology import (
    intro_instruction,
    ontology_criteria,
    template_prompt,
)
from ontocast.tool import LLMTool
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


async def criticise_ontology(state: AgentState, tools: ToolBox) -> AgentState:
    """Enhanced ontology criticism with SPARQL operations.

    This function performs a critical analysis of the ontology in the current
    state, with SPARQL operation support.

    Args:
        state: The current agent state containing the ontology to analyze.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with analysis results.
    """

    progress_info = state.get_chunk_progress_string()
    logger.info(
        f"Ontology Critic for {progress_info}: visit {state.node_visits[WorkflowNode.CRITICISE_ONTOLOGY] + 1}/{state.max_visits}"
    )

    if state.current_chunk is None:
        state.status = Status.FAILED
        return state

    if state.current_ontology.is_null():
        raise ValueError(
            f"Null ontology cannot be criticised: {state.current_ontology.iri} is not a valid ontology"
        )

    parser = PydanticOutputParser(pydantic_object=OntologyCritiqueReport)
    llm_tool: LLMTool = await tools.get_llm_tool(state.budget_tracker)

    ontology_ttl = state.current_ontology.graph.serialize(format="turtle")

    ontology_chapter = ontology_template.format(
        ontology_ttl=ontology_ttl,
    )

    text_chapter = text_template.format(text=state.current_chunk.text)

    user_instruction = state.ontology_user_instruction

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "facts_instruction",
            "ontology_instruction",
            "user_instruction",
            "text_chapter",
            "improvement_instruction",
            "format_instructions",
        ],
    )

    try:
        critique: OntologyCritiqueReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "preamble": system_preamble,
                "intro_instruction": intro_instruction,
                "ontology_criteria": ontology_criteria,
                "text_chapter": text_chapter,
                "user_instruction": user_instruction,
                "ontology_chapter": ontology_chapter,
                "format_instructions": parser.get_format_instructions(),
            },
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
