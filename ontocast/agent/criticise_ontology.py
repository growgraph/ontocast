"""Enhanced ontology criticism agent with memory and SPARQL operations.

This module provides enhanced functionality for analyzing and validating ontologies
with memory of previous critiques and SPARQL operation support.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.onto.constants import ONTOLOGY_NULL_IRI
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import OntologyUpdateCritiqueReport
from ontocast.onto.state import AgentState
from ontocast.prompt.criticise_ontology import (
    document_template,
    intro_instruction,
    ontology_template,
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

    if state.current_ontology.iri == ONTOLOGY_NULL_IRI:
        raise ValueError(
            f"{state.current_ontology.ontology_id} : {state.current_ontology.iri} is not a valid ontology"
        )

    parser = PydanticOutputParser(pydantic_object=OntologyUpdateCritiqueReport)
    llm_tool: LLMTool = tools.llm

    ontology_ttl = state.current_ontology.graph.serialize(format="turtle")

    ontology_chapter = ontology_template.format(
        ontology_ttl=ontology_ttl,
    )

    document_chapter = document_template.format(document=state.current_chunk.text)

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "facts_instruction",
            "ontology_instruction",
            "text_instruction",
            "critique_instruction",
            "format_instructions",
        ],
    )

    try:
        response = llm_tool(
            prompt.format_prompt(
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
