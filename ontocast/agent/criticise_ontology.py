"""Enhanced ontology criticism agent with memory and SPARQL operations.

This module provides enhanced functionality for analyzing and validating ontologies
with memory of previous critiques and SPARQL operation support.
"""

import logging

from langchain.output_parsers import PydanticOutputParser

from ontocast.onto.constants import ONTOLOGY_NULL_IRI
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import OntologyUpdateCritiqueReport
from ontocast.onto.state import AgentState
from ontocast.prompt.criticise_ontology import (
    document_template,
    intro_first_no_seed_instruction,
    intro_first_with_seed_instruction,
    intro_subsequent_instruction,
    ontology_template,
    ontology_update_template,
    system_preamble,
    template_prompt,
)
from ontocast.tool import LLMTool
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def criticise_ontology(state: AgentState, tools: ToolBox) -> AgentState:
    """Enhanced ontology criticism with memory and SPARQL operations.

    This function performs a critical analysis of the ontology in the current
    state, with memory of previous critiques and SPARQL operation support.

    Args:
        state: The current agent state containing the ontology to analyze.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with analysis results.
    """
    logger.info("Enhanced ontology criticism with memory")

    if state.current_chunk is None:
        state.status = Status.FAILED
        return state

    is_first_visit = (
        state.get_node_status(WorkflowNode.CRITICISE_ONTOLOGY) == Status.NOT_VISITED
    )

    if state.current_ontology.iri == ONTOLOGY_NULL_IRI:
        raise ValueError(
            f"{state.current_ontology.ontology_id} : {state.current_ontology.iri} is not a valid ontology"
        )

    if is_first_visit:
        return criticise_ontology_first_visit(state, tools)
    else:
        return criticise_ontology_with_updates(state, tools)


def criticise_ontology_first_visit(state: AgentState, tools: ToolBox) -> AgentState:
    parser = PydanticOutputParser(pydantic_object=OntologyUpdateCritiqueReport)
    llm_tool: LLMTool = tools.llm

    ontology_ttl = state.current_ontology.graph.serialize(format="turtle")
    if state.ontology_updates:
        ontology_update_str = ontology_update_template.format(
            ontology_update=state.generate_ontology_updates_markdown()
        )
        intro_instruction = intro_first_with_seed_instruction
    else:
        ontology_update_str = ""
        intro_instruction = intro_first_no_seed_instruction

    ontology_chapter = ontology_template.format(
        ontology_ttl=ontology_ttl, ontology_updates=ontology_update_str
    )

    document_chapter = document_template.format(document=state.current_chunk.text)
    try:
        response = llm_tool(
            template_prompt.format(
                preamble=system_preamble,
                intro_instruction=intro_instruction,
                ontology_criteria=state.current_chunk.text,
                document_chapter=document_chapter,
                ontology_chapter=ontology_chapter,
                format_instructions=parser.get_format_instructions(),
            )
        )

        critique: OntologyUpdateCritiqueReport = parser.parse(response.content)
        logger.debug(
            f"Parsed critique report - success: {critique.update_successful}, "
            f"score: {critique.score}"
        )

        if critique.is_satisfactory:
            state.status = Status.SUCCESS
            state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.SUCCESS)
            logger.info("Ontology critique passed")
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStage.ONTOLOGY_CRITIQUE
            state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.FAILED)
            state.failure_reason = f"Ontology critique failed: {critique.issues}"
            logger.warning(f"Ontology critique failed: {critique.issues}")

        return state

    except Exception as e:
        logger.error(f"Failed to critique ontology: {str(e)}")
        state.set_failure(FailureStage.ONTOLOGY_CRITIQUE, str(e))
        state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.FAILED)
        return state


def criticise_ontology_with_updates(state: AgentState, tools: ToolBox) -> AgentState:
    parser = PydanticOutputParser(pydantic_object=OntologyUpdateCritiqueReport)
    llm_tool: LLMTool = tools.llm

    # ontology_updated_str = state.render_uptodate_ontology().graph.serialize(
    #     format="turtle"
    # )

    if state.ontology_updates:
        ontology_update_str = ontology_update_template.format(
            ontology_update=state.generate_ontology_updates_markdown()
        )
    else:
        ontology_update_str = ""

    document_chapter = ""
    try:
        response = llm_tool(
            template_prompt.format(
                preamble=system_preamble,
                intro_instruction=intro_subsequent_instruction,
                ontology_criteria=state.current_chunk.text,
                document_chapter=document_chapter,
                ontology_chapter=ontology_update_str,
                format_instructions=parser.get_format_instructions(),
            )
        )

        critique: OntologyUpdateCritiqueReport = parser.parse(response.content)
        logger.debug(
            f"Parsed critique report - success: {critique.update_successful}, "
            f"score: {critique.score}"
        )

        if critique.update_successful:
            state.status = Status.SUCCESS
            state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.SUCCESS)

            logger.info("Ontology critique passed")
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStage.ONTOLOGY_CRITIQUE
            state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.FAILED)
            state.failure_reason = (
                f"Ontology critique failed: {critique.improvement_instructions}"
            )
            logger.warning(f"Ontology critique failed: {critique.issues}")

        return state

    except Exception as e:
        logger.error(f"Failed to critique ontology: {str(e)}")
        state.set_failure(FailureStage.ONTOLOGY_CRITIQUE, str(e))
        state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.FAILED)
        return state
