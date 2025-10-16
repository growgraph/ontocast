"""Enhanced fact criticism agent with memory and SPARQL operations.

This module provides enhanced functionality for analyzing and validating facts
with memory of previous critiques and SPARQL operation support.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.onto.context import AgentType
from ontocast.onto.enum import FailureStage, Status
from ontocast.onto.model import FactsCritiqueReport
from ontocast.onto.state import AgentState
from ontocast.prompt.common import (
    facts_template,
    ontology_template,
    output_instruction_crit_facts,
    text_template,
    user_template,
)
from ontocast.prompt.criticise_facts import (
    evaluation_instruction,
    preamble,
    template_prompt,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def criticise_facts(state: AgentState, tools: ToolBox) -> AgentState:
    """Enhanced criticize facts with memory and SPARQL operations.

    This function performs a critical analysis of the facts in the current chunk,
    with memory of previous critiques and SPARQL operation support.

    Args:
        state: The current agent state containing the chunk to analyze.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with analysis results.
    """
    if not state.current_chunk:
        logger.warning("No current chunk to analyze")
        return state

    logger.info("Criticize facts with memory")

    llm_tool = tools.llm
    version_manager = tools.version_manager
    parser = PydanticOutputParser(pydantic_object=FactsCritiqueReport)

    ontology_ttl = state.current_ontology.graph.serialize(format="turtle")

    ontology_chapter = ontology_template.format(
        ontology_ttl=ontology_ttl,
    )

    facts_ttl = state.current_chunk.graph.serialize(format="turtle")

    facts_chapter = facts_template.format(
        facts_ttl=facts_ttl,
    )

    text_chapter = text_template.format(text=state.current_chunk.text)

    user_instruction = (
        user_template.format(user_instruction=state.facts_user_instruction)
        if state.facts_user_instruction
        else ""
    )

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "evaluation_instruction",
            "user_instruction",
            "ontology_chapter",
            "facts_chapter",
            "text_chapter",
            "output_instruction",
            "format_instructions",
        ],
    )

    prompt_data = {
        "preamble": preamble,
        "evaluation_instruction": evaluation_instruction,
        "user_instruction": user_instruction,
        "ontology_chapter": ontology_chapter,
        "facts_chapter": facts_chapter,
        "text_chapter": text_chapter,
        "format_instructions": parser.get_format_instructions(),
        "output_instruction": output_instruction_crit_facts,
    }

    try:
        response = llm_tool(prompt.format_prompt(**prompt_data))

        critique: FactsCritiqueReport = parser.parse(response.content)
        logger.debug(
            f"Parsed critique report - success: {critique.facts_graph_derivation_success}, "
            f"score: {critique.facts_graph_derivation_score}"
        )

        # Store critique in version manager and update context
        if version_manager:
            chunk_id = getattr(state.current_chunk, "chunk_id", "unknown")
            latest_facts_version = version_manager.get_latest_facts_version(chunk_id)
            if latest_facts_version:
                latest_facts_version.metadata.update(
                    {
                        "last_critique": critique.facts_graph_derivation_critique_comment,
                        "critique_score": critique.facts_graph_derivation_score,
                        "critique_satisfactory": critique.facts_graph_derivation_success,
                    }
                )

                # Update context with critique information
                state.update_context_for_agent(
                    agent_type=AgentType.CRITIC_FACTS,
                    facts_critique={
                        "issues": critique.facts_graph_derivation_critique_comment,
                        "score": critique.facts_graph_derivation_score,
                        "satisfactory": critique.facts_graph_derivation_success,
                    },
                    metadata={"critique_timestamp": "now", "chunk_id": chunk_id},
                )

        if critique.facts_graph_derivation_success:
            state.status = Status.SUCCESS
            logger.info("Facts critique passed")
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStage.FACTS_CRITIQUE
            state.failure_reason = f"Facts critique failed: {critique.facts_graph_derivation_critique_comment}"
            logger.warning(
                f"Facts critique failed: {critique.facts_graph_derivation_critique_comment}"
            )

        return state

    except Exception as e:
        logger.error(f"Failed to critique facts: {str(e)}")
        state.set_failure(FailureStage.FACTS_CRITIQUE, str(e))
        return state
