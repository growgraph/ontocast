"""Enhanced ontology criticism agent with memory and SPARQL operations.

This module provides enhanced functionality for analyzing and validating ontologies
with memory of previous critiques and SPARQL operation support.
"""

import logging

from langchain.output_parsers import PydanticOutputParser

from ontocast.onto.constants import ONTOLOGY_NULL_IRI
from ontocast.onto.context import AgentType, Role
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.model import OntologyUpdateCritiqueReport
from ontocast.onto.state import AgentState
from ontocast.prompt.enhanced_criticise_ontology import (
    prompt_fresh_enhanced,
    prompt_update_enhanced,
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
    llm_tool: LLMTool = tools.llm
    version_manager = tools.version_manager
    parser = PydanticOutputParser(pydantic_object=OntologyUpdateCritiqueReport)

    if state.current_chunk is None:
        state.status = Status.FAILED
        return state

    # Get context for this agent with conversation memory
    agent_context = state.get_context_for_agent(AgentType.CRITIC_ONTOLOGY)

    # Add current interaction to conversation memory
    agent_context.add_conversation_memory(
        role=Role.SYSTEM,
        content=f"Starting ontology critique for ontology: {state.current_ontology.iri}",
        metadata={
            "interaction_type": "ontology_critique",
            "ontology_iri": state.current_ontology.iri,
        },
    )

    # Build dynamic context for this interaction
    agent_context.build_dynamic_context(
        interaction_type="ontology_critique",
        ontology_iri=state.current_ontology.iri,
        document_text=state.current_chunk.text[:200],
    )

    previous_critique_context = agent_context.get_llm_context()

    if state.current_ontology.iri == ONTOLOGY_NULL_IRI:
        prompt_template = prompt_fresh_enhanced
        ontology_original_str = ""
    else:
        prompt_template = prompt_update_enhanced
        ontology_original_str = state.current_ontology.graph.serialize(format="turtle")

    try:
        response = llm_tool(
            prompt_template.format(
                previous_context=previous_critique_context,
                ontology_original_str=ontology_original_str,
                document=state.current_chunk.text,
                ontology_update=state.current_ontology.graph.serialize(format="turtle"),
                format_instructions=parser.get_format_instructions(),
            )
        )

        critique: OntologyUpdateCritiqueReport = parser.parse(response.content)
        logger.debug(
            f"Parsed critique report - success: {critique.is_satisfactory}, "
            f"score: {critique.score}"
        )

        # Add LLM response to conversation memory
        agent_context.add_conversation_memory(
            role=Role.ASSISTANT,
            content=f"Ontology critique completed. Success: {critique.is_satisfactory}, Score: {critique.score}",
            metadata={
                "critique_success": critique.is_satisfactory,
                "critique_score": critique.score,
                "critique_issues": critique.issues,
            },
        )

        # Store critique in version manager and update context
        if version_manager and state.current_ontology.iri != ONTOLOGY_NULL_IRI:
            latest_version = version_manager.get_latest_ontology_version(
                state.current_ontology.iri
            )
            if latest_version:
                latest_version.metadata.update(
                    {
                        "last_critique": critique.issues,
                        "critique_score": critique.score,
                        "critique_satisfactory": critique.is_satisfactory,
                    }
                )

                # Update context with critique information
                state.update_context_for_agent(
                    agent_type=AgentType.CRITIC_ONTOLOGY,
                    ontology_critique={
                        "issues": critique.issues,
                        "score": critique.score,
                        "satisfactory": critique.is_satisfactory,
                    },
                    metadata={
                        "critique_timestamp": "now",
                        "ontology_iri": state.current_ontology.iri,
                    },
                )

        if critique.is_satisfactory:
            state.status = Status.SUCCESS
            logger.info("Ontology critique passed")
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStages.ONTOLOGY_CRITIQUE
            state.failure_reason = f"Ontology critique failed: {critique.issues}"
            logger.warning(f"Ontology critique failed: {critique.issues}")

        return state

    except Exception as e:
        logger.error(f"Failed to critique ontology: {str(e)}")
        state.set_failure(FailureStages.ONTOLOGY_CRITIQUE, str(e))
        return state
