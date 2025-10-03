"""Fact criticism agent for OntoCast.

This module provides functionality for analyzing and validating facts extracted
from text chunks, ensuring their consistency and correctness.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.onto.enum import FailureStages
from ontocast.onto.model import KGCritiqueReport
from ontocast.onto.state import AgentState
from ontocast.prompt.criticise_facts import prompt as criticise_facts_prompt
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def criticise_facts(state: AgentState, tools: ToolBox) -> AgentState:
    """Analyze and validate facts in the current chunk.

    This function performs a critical analysis of the facts in the current chunk,
    checking for consistency, correctness, and potential issues.

    Args:
        state: The current agent state containing the chunk to analyze.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with analysis results.
    """
    if not state.current_chunk:
        logger.warning("No current chunk to analyze")
        return state

    logger.info("Criticize facts")

    llm_tool = tools.llm
    parser = PydanticOutputParser(pydantic_object=KGCritiqueReport)

    prompt = PromptTemplate(
        template=criticise_facts_prompt,
        input_variables=[
            "ontology",
            "document",
            "knowledge_graph",
            "format_instructions",
        ],
    )

    response = llm_tool(
        prompt.format_prompt(
            ontology=state.current_ontology.graph.serialize(format="turtle"),
            document=state.current_chunk.text,
            knowledge_graph=state.current_chunk.graph.serialize(format="turtle"),
            format_instructions=parser.get_format_instructions(),
        )
    )
    critique: KGCritiqueReport = parser.parse(response.content)
    logger.debug(
        f"Parsed critique report - success: {critique.facts_graph_derivation_success}, "
        f"score: {critique.facts_graph_derivation_score}"
    )

    if critique.facts_graph_derivation_success:
        logger.debug("Facts critique successful, clearing failure state")
        state.clear_failure()
    else:
        logger.debug("Facts critique failed, setting failure state")
        state.set_failure(
            stage=FailureStages.FACTS_CRITIQUE,
            reason=critique.facts_graph_derivation_critique_comment,
            success_score=critique.facts_graph_derivation_score,
        )
    return state
