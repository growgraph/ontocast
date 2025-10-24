"""Ontology triple rendering agent for OntoCast.

This module provides functionality for rendering RDF triples from ontologies into
human-readable formats, making the ontological knowledge more accessible and
understandable.
The agent decides between generating bare Turtle for fresh ontologies and SPARQL operations for updates.

"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import render_suggestions_prompt
from ontocast.onto.constants import ONTOLOGY_NULL_ID
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.ontology import Ontology
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState
from ontocast.prompt.common import output_instruction_sparql, output_instruction_ttl
from ontocast.prompt.common import system_preamble_ontology as system_preamble
from ontocast.prompt.render_ontology import (
    general_ontology_instruction,
    intro_instruction_fresh,
    intro_instruction_update,
    prefix_instruction_fresh,
    prefix_instruction_update,
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
    progress_info = state.get_chunk_progress_string()
    logger.info(
        f"Structured ontology rendering for {progress_info} with Turtle/SPARQL output"
    )

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
    general_ontology_instruction_str = general_ontology_instruction.format(
        prefix_instruction=prefix_instruction_fresh
    )

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
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
        llm_tool = tools.get_llm_tool_with_budget_tracker(state.llm_budget_tracker)
        response = llm_tool(
            prompt.format_prompt(
                preamble=system_preamble,
                intro_instruction=intro_instruction,
                ontology_instruction=general_ontology_instruction_str,
                output_instruction=output_instruction,
                ontology_ttl=ontology_ttl,
                user_instruction=state.ontology_user_instruction,
                improvement_instruction=improvement_instruction_str,
                text=state.current_chunk.text,
                format_instructions=parser.get_format_instructions(),
            ),
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
    improvement_instruction_str = render_suggestions_prompt(
        state.suggestions, WorkflowNode.TEXT_TO_ONTOLOGY
    )

    general_ontology_instruction_str = general_ontology_instruction.format(
        prefix_instruction=prefix_instruction_update.format(
            ontology_prefix=state.current_ontology.prefix
        )
    )

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
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
        llm_tool = tools.get_llm_tool_with_budget_tracker(state.llm_budget_tracker)
        response = llm_tool(
            prompt.format_prompt(
                preamble=system_preamble,
                intro_instruction=intro_instruction,
                ontology_instruction=general_ontology_instruction_str,
                output_instruction=output_instruction,
                improvement_instruction=improvement_instruction_str,
                ontology_ttl=ontology_ttl,
                user_instruction=state.ontology_user_instruction,
                text=state.current_chunk.text,
                format_instructions=parser.get_format_instructions(),
            ),
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
