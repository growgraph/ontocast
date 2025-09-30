"""Ontology selection agent for OntoCast.

This module provides functionality for selecting appropriate ontologies based on
the content of text chunks, ensuring that the chosen ontology best matches the
domain and requirements of the text.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.onto.enum import FailureStages
from ontocast.onto.extras import NULL_ONTOLOGY
from ontocast.onto.model import OntologySelectorReport
from ontocast.onto.state import AgentState
from ontocast.prompt.select_ontology import template_prompt
from ontocast.tool import OntologyManager
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def select_ontology(state: AgentState, tools: ToolBox) -> AgentState:
    """Select an appropriate ontology for the current chunk.

    This function analyzes the current chunk and selects the most appropriate
    ontology based on its content and requirements.

    Args:
        state: The current agent state containing the chunk to process.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with selected ontology.
    """
    logger.info("Selecting ontology")
    llm_tool = tools.llm
    om_tool: OntologyManager = tools.ontology_manager

    parser = PydanticOutputParser(pydantic_object=OntologySelectorReport)

    if len(om_tool.ontologies) > 0:
        ontologies_desc = "\n\n".join([o.describe() for o in om_tool.ontologies])
        logger.info(f"Retrieved descriptions for {len(om_tool.ontologies)} ontologies")

        if state.current_chunk is None:
            if state.chunks:
                state.current_chunk = state.chunks.pop(0)
            else:
                state.set_failure(
                    FailureStages.NO_CHUNKS_TO_PROCESS, "No chunks to process"
                )

        excerpt = state.current_chunk.text[:1000] + " ..."

        prompt = PromptTemplate(
            template=template_prompt,
            input_variables=["excerpt", "ontologies_desc", "format_instructions"],
        )

        response = llm_tool(
            prompt.format_prompt(
                excerpt=excerpt,
                ontologies_desc=ontologies_desc,
                format_instructions=parser.get_format_instructions(),
            )
        )
        selector = parser.parse(response.content)
        logger.debug(
            f"Parsed selector report - Selected ontology: {selector.ontology_id}"
        )
        # Always select ontology using id and/or IRI (consistency check is handled in OntologyManager)
        state.current_ontology = om_tool.get_ontology(
            selector.ontology_id, selector.ontology_iri
        )
    else:
        state.current_ontology = NULL_ONTOLOGY
    logger.debug(f"Current ontology set to: {state.current_ontology.ontology_id}")
    return state
