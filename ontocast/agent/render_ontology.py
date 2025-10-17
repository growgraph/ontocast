"""Ontology triple rendering agent for OntoCast.

This module provides functionality for rendering RDF triples from ontologies into
human-readable formats, making the ontological knowledge more accessible and
understandable.
The agent decides between generating bare Turtle for fresh ontologies and SPARQL operations for updates.

"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.onto.constants import ONTOLOGY_NULL_ID
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.ontology import Ontology
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState
from ontocast.prompt.common import output_instruction_sparql, output_instruction_ttl
from ontocast.prompt.common import system_preamble_ontology as system_preamble
from ontocast.prompt.render_ontology import (
    general_ontology_instruction,
    improvement_instruction_template,
    intro_instruction_fresh,
    intro_instruction_update,
    template_prompt,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def render_ontology(state: AgentState, tools: ToolBox) -> AgentState:
    """Structured hybrid ontology renderer with Turtle/SPARQL decision logic.

    This function decides between generating bare Turtle for fresh ontologies
    and SPARQL operations for updates based on whether the ontology exists.

    Args:
        state: The current agent state
        tools: The toolbox containing necessary tools

    Returns:
        AgentState: Updated state with rendered ontology
    """
    logger.info("Structured ontology rendering with Turtle/SPARQL output")

    has_no_seed_ontology = state.ontology_id == ONTOLOGY_NULL_ID

    if has_no_seed_ontology:
        return render_ontology_fresh(state, tools)
    else:
        return render_ontology_update(state, tools)


def render_ontology_fresh(state: AgentState, tools: ToolBox) -> AgentState:
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

    parser = PydanticOutputParser(pydantic_object=Ontology)
    logger.info("Rendering fresh ontology")
    intro_instruction = intro_instruction_fresh
    output_instruction = output_instruction_ttl
    ontology_ttl = ""
    improvement_instruction_str = ""

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "system_preamble",
            "intro_instruction",
            "ontology_instruction",
            "output_instruction",
            "user_instruction",
            "improvement_instruction",
            "ontology_ttl",
            "text",
            "format_instructions",
        ],
    )

    try:
        response = tools.llm(
            prompt.format_prompt(
                system_preamble=system_preamble,
                intro_instruction=intro_instruction,
                ontology_instruction=general_ontology_instruction,
                output_instruction=output_instruction,
                ontology_ttl=ontology_ttl,
                user_instruction=state.ontology_user_instruction,
                improvement_instruction=improvement_instruction_str,
                text=state.current_chunk.text,
                format_instructions=parser.get_format_instructions(),
            )
        )

        state.current_ontology = parser.parse(response.content)
        state.current_ontology.graph.sanitize_prefixes_namespaces()

        logger.info(
            f"New ontology created with {len(state.current_ontology.graph)} triples."
        )
        state.clear_failure()
        state.set_node_status(WorkflowNode.TEXT_TO_ONTOLOGY, Status.SUCCESS)
        return state

    except Exception as e:
        logger.error(f"Failed to generate triples: {str(e)}")
        state.set_node_status(WorkflowNode.TEXT_TO_ONTOLOGY, Status.FAILED)
        state.set_failure(FailureStage.GENERATE_TTL_FOR_ONTOLOGY, str(e))
        return state


def render_ontology_update(state: AgentState, tools: ToolBox) -> AgentState:
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

    parser = PydanticOutputParser(pydantic_object=GraphUpdate)
    ontology_iri = state.current_ontology.iri
    ontology_desc = state.current_ontology.describe()
    intro_instruction = intro_instruction_update.format(
        ontology_iri=ontology_iri, ontology_desc=ontology_desc
    )
    ontology_ttl = state.current_ontology.graph.serialize(format="turtle")
    output_instruction = output_instruction_sparql
    if state.improvements_suggestions:
        improvement_instruction_str = improvement_instruction_template.format(
            "\n- ".join(state.improvements_suggestions)
        )
    else:
        improvement_instruction_str = ""

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "system_preamble",
            "intro_instruction",
            "ontology_instruction",
            "output_instruction",
            "user_instruction",
            "improvement_instruction",
            "ontology_ttl",
            "text",
            "format_instructions",
        ],
    )

    try:
        response = tools.llm(
            prompt.format_prompt(
                system_preamble=system_preamble,
                intro_instruction=intro_instruction,
                ontology_instruction=general_ontology_instruction,
                output_instruction=output_instruction,
                improvement_instruction=improvement_instruction_str,
                ontology_ttl=ontology_ttl,
                user_instruction=state.ontology_user_instruction,
                text=state.current_chunk.text,
                format_instructions=parser.get_format_instructions(),
            )
        )

        graph_update: GraphUpdate = parser.parse(response.content)
        state.ontology_updates.append(graph_update)
        state.update_ontology()

        logger.info(f"Ontology update has {len(graph_update.operations)} operations.")
        state.clear_failure()
        state.set_node_status(WorkflowNode.TEXT_TO_ONTOLOGY, Status.SUCCESS)
        return state

    except Exception as e:
        logger.error(f"Failed to generate ontology update: {str(e)}")
        state.set_node_status(WorkflowNode.TEXT_TO_ONTOLOGY, Status.FAILED)
        state.set_failure(FailureStage.GENERATE_SPARQL_UPDATE_FOR_ONTOLOGY, str(e))
        return state
