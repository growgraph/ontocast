"""Ontology criticism agent for OntoCast.

This module provides functionality for analyzing and validating ontologies,
ensuring their structural integrity, consistency, and alignment with domain
requirements.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.onto.constants import ONTOLOGY_NULL_IRI
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.model import OntologyUpdateCritiqueReport
from ontocast.onto.state import AgentState
from ontocast.prompt.criticise_ontology import prompt_fresh, prompt_update
from ontocast.tool import LLMTool, OntologyManager
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def criticise_ontology(state: AgentState, tools: ToolBox) -> AgentState:
    """Analyze and validate the current ontology.

    This function performs a critical analysis of the ontology in the current
    state, checking for structural integrity, consistency, and alignment with
    domain requirements.

    Args:
        state: The current agent state containing the ontology to analyze.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with analysis results.
    """
    logger.info("Criticize ontology")
    llm_tool: LLMTool = tools.llm
    om_tool: OntologyManager = tools.ontology_manager
    parser = PydanticOutputParser(pydantic_object=OntologyUpdateCritiqueReport)

    if state.current_chunk is None:
        state.status = Status.FAILED
        return state

    if state.current_ontology.iri == ONTOLOGY_NULL_IRI:
        prompt = prompt_fresh
        ontology_original_str = ""
    else:
        ontology_original_str = (
            f"Here is the original ontology:"
            f"\n```ttl\n{state.current_ontology.graph.serialize(format='turtle')}\n```"
        )
        prompt = prompt_update

    prompt = PromptTemplate(
        template=prompt,
        input_variables=[
            "ontology_update",
            "document",
            "format_instructions",
            "ontology_original_str",
        ],
    )

    response = llm_tool(
        prompt.format_prompt(
            ontology_update=state.ontology_addendum.graph.serialize(format="turtle"),
            document=state.current_chunk.text,
            format_instructions=parser.get_format_instructions(),
            ontology_original_str=ontology_original_str,
        )
    )
    critique: OntologyUpdateCritiqueReport = parser.parse(response.content)
    logger.debug(
        f"Parsed critique report status: {critique.ontology_update_success}, "
        f"score: {critique.ontology_update_score}"
    )

    if state.ontology_addendum.iri == ONTOLOGY_NULL_IRI:
        state.set_failure(
            stage=FailureStages.ONTOLOGY_CRITIQUE,
            reason=critique.ontology_update_failed,
            success_score=critique.ontology_update_score,
        )

    if (
        state.current_ontology.iri == ONTOLOGY_NULL_IRI
        or state.current_ontology.iri is None
        or state.ontology_addendum.iri not in tools.ontology_manager
    ):
        logger.debug("Adding new ontology to manager")
        om_tool.ontologies.append(state.ontology_addendum)
        state.current_ontology = state.ontology_addendum
    else:
        logger.info(f"Updating existing ontology: {state.current_ontology.ontology_id}")
        om_tool.update_ontology(
            state.current_ontology.ontology_id, state.ontology_addendum.graph
        )

    if critique.ontology_update_success:
        logger.info("Ontology critique successful, clearing failure state")
        state.clear_failure()
    else:
        logger.info("Ontology critique failed, setting failure state")
        state.set_failure(
            stage=FailureStages.ONTOLOGY_CRITIQUE,
            reason=critique.ontology_update_critique,
            success_score=critique.ontology_update_score,
        )

    return state
