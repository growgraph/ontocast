"""Ontology triple rendering agent for OntoCast.

This module provides functionality for rendering RDF triples from ontologies into
human-readable formats, making the ontological knowledge more accessible and
understandable.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.onto.constants import ONTOLOGY_NULL_ID
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.state import AgentState
from ontocast.prompt.render_ontology import (
    failure_instruction,
    instructions,
    ontology_instruction_fresh,
    ontology_instruction_update,
    specific_ontology_instruction_fresh,
    specific_ontology_instruction_update,
    template_prompt,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def render_onto_triples(state: AgentState, tools: ToolBox) -> AgentState:
    """Render ontology triples into a human-readable format.

    This function takes the triples from the current ontology and renders them
    into a more accessible format, making the ontological knowledge easier to
    understand.

    Args:
        state: The current agent state containing the ontology to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with rendered triples.
    """
    logger.info("Starting to render ontology triples")
    llm_tool = tools.llm

    parser = PydanticOutputParser(pydantic_object=Ontology)

    logger.debug(f"Using domain: {state.current_domain}")

    if state.current_ontology.ontology_id == ONTOLOGY_NULL_ID or (
        state.current_ontology.iri not in tools.ontology_manager
    ):
        logger.info("Creating a fresh ontology")
        ontology_instruction = ontology_instruction_fresh
        specific_ontology_instruction = specific_ontology_instruction_fresh.format(
            current_domain=state.current_domain
        )
    else:
        ontology_iri = state.current_ontology.iri
        ontology_str = state.current_ontology.graph.serialize(format="turtle")
        ontology_desc = state.current_ontology.describe()
        ontology_instruction = ontology_instruction_update.format(
            ontology_iri=ontology_iri,
            ontology_desc=ontology_desc,
            ontology_str=ontology_str,
        )
        specific_ontology_instruction = specific_ontology_instruction_update.format(
            ontology_namespace=state.current_ontology.namespace
        )

    _instructions = instructions.format(
        specific_ontology_instruction=specific_ontology_instruction
    )

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "text",
            "instructions",
            "ontology_instruction",
            "failure_instruction",
            "format_instructions",
        ],
    )

    if state.status != Status.SUCCESS and state.failure_reason is not None:
        _failure_instruction = failure_instruction.format(
            failure_stage=state.failure_stage,
            failure_reason=state.failure_reason,
        )
    else:
        _failure_instruction = ""

    try:
        response = llm_tool(
            prompt.format_prompt(
                text=state.current_chunk.text,
                instructions=_instructions,
                ontology_instruction=ontology_instruction,
                failure_instruction=_failure_instruction,
                format_instructions=parser.get_format_instructions(),
            )
        )

        state.ontology_addendum = parser.parse(response.content)
        state.ontology_addendum.graph.sanitize_prefixes_namespaces()

        logger.info(
            f"Ontology addendum has {len(state.ontology_addendum.graph)} triples."
        )
        state.clear_failure()
        return state

    except Exception as e:
        logger.error(f"Failed to generate triples: {str(e)}")
        state.set_failure(FailureStages.PARSE_TEXT_TO_ONTOLOGY_TRIPLES, str(e))
        return state
