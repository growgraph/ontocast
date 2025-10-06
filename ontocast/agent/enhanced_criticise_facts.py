"""Enhanced fact criticism agent with memory and SPARQL operations.

This module provides enhanced functionality for analyzing and validating facts
with memory of previous critiques and SPARQL operation support.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.onto.context import AgentType
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.model import KGCritiqueReport
from ontocast.onto.state import AgentState
from ontocast.prompt.enhanced_criticise_facts import (
    prompt_enhanced,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def enhanced_criticise_facts(state: AgentState, tools: ToolBox) -> AgentState:
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

    logger.info("Enhanced criticize facts with memory")

    llm_tool = tools.llm
    version_manager = tools.version_manager
    parser = PydanticOutputParser(pydantic_object=KGCritiqueReport)

    # Get context for this agent with conversation memory
    agent_context = state.get_context_for_agent("facts_critic", AgentType.CRITIC)

    # Add current interaction to conversation memory
    agent_context.add_conversation_memory(
        role="system",
        content=f"Starting facts critique for chunk: {state.current_chunk.text[:100]}...",
        metadata={
            "interaction_type": "facts_critique",
            "chunk_id": getattr(state.current_chunk, "chunk_id", "unknown"),
        },
    )

    # Build dynamic context for this interaction
    agent_context.build_dynamic_context(
        interaction_type="facts_critique",
        chunk_text=state.current_chunk.text[:200],
        ontology_iri=state.current_ontology.iri,
    )

    previous_critique_context = agent_context.get_llm_context()

    prompt = PromptTemplate(
        template=prompt_enhanced,
        input_variables=[
            "ontology",
            "document",
            "knowledge_graph",
            "format_instructions",
        ],
    )

    try:
        response = llm_tool(
            prompt.format_prompt(
                previous_context=previous_critique_context,
                ontology=state.current_ontology.graph.serialize(format="turtle"),
                document=state.current_chunk.text,
                knowledge_graph=state.current_chunk.graph.serialize(format="turtle")
                if state.current_chunk.graph
                else "",
                format_instructions=parser.get_format_instructions(),
            )
        )

        critique: KGCritiqueReport = parser.parse(response.content)
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
                    agent_name="facts_critic",
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
            state.failure_stage = FailureStages.FACTS_CRITIQUE
            state.failure_reason = f"Facts critique failed: {critique.facts_graph_derivation_critique_comment}"
            logger.warning(
                f"Facts critique failed: {critique.facts_graph_derivation_critique_comment}"
            )

        return state

    except Exception as e:
        logger.error(f"Failed to critique facts: {str(e)}")
        state.set_failure(FailureStages.FACTS_CRITIQUE, str(e))
        return state
