"""Fact rendering agent for OntoCast.

This module provides functionality for rendering facts from RDF graphs into
human-readable formats, making the extracted knowledge more accessible and
understandable.
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.onto.constants import DEFAULT_CHUNK_IRI
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import SemanticTriplesFactsReport
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState
from ontocast.prompt.common import (
    critique_instruction_template,
)
from ontocast.prompt.render_facts import (
    facts_instruction_template,
    ontology_instruction_template,
    preamble_first_visit,
    preamble_subsequent_visit,
    template_prompt,
    text_instruction_template,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def render_facts(state: AgentState, tools: ToolBox) -> AgentState:
    """Structured hybrid facts renderer with Turtle/SPARQL decision logic.

    This function decides between generating bare Turtle for fresh facts
    and SPARQL operations for updates based on whether facts exist.

    Args:
        state: The current agent state
        tools: The toolbox containing necessary tools

    Returns:
        AgentState: Updated state with rendered facts
    """

    is_first_visit = len(state.current_chunk.graph) == 0

    if is_first_visit:
        logger.info("Generating fresh facts as Turtle")
        return render_facts_fresh(state, tools)
    else:
        logger.info("Generating facts update")
        return render_facts_update(state, tools)


def _prepare_prompt_data(state: AgentState) -> dict[str, str]:
    """Prepare common prompt data for both fresh and update rendering.

    Args:
        state: The current agent state

    Returns:
        Dictionary containing formatted prompt components
    """
    ontology_instruction = ontology_instruction_template.format(
        ontology_str=state.current_ontology.graph.serialize(format="turtle")
    )

    facts_instruction_str = facts_instruction_template.format(
        ontology_namespace=state.current_ontology.namespace,
        current_doc_namespace=DEFAULT_CHUNK_IRI,
    )

    text_instruction = text_instruction_template.format(text=state.current_chunk.text)

    return {
        "ontology_instruction": ontology_instruction,
        "facts_instruction": facts_instruction_str,
        "text_instruction": text_instruction,
    }


def _create_prompt_template() -> PromptTemplate:
    """Create the common prompt template used by both rendering functions.

    Returns:
        Configured PromptTemplate instance
    """
    return PromptTemplate(
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


def _handle_rendering_error(
    state: AgentState, error: Exception, stage: FailureStage
) -> AgentState:
    """Handle rendering errors consistently.

    Args:
        state: The current agent state
        error: The exception that occurred
        stage: The failure stage to set

    Returns:
        Updated state with failure information
    """
    logger.error(f"Failed to generate triples: {str(error)}")
    state.set_failure(stage, str(error))
    return state


def render_facts_fresh(state: AgentState, tools: ToolBox) -> AgentState:
    """Render fresh facts from the current chunk into Turtle format.

    Args:
        state: The current agent state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with rendered facts.
    """
    logger.info("Rendering fresh facts")
    llm_tool = tools.llm
    parser = PydanticOutputParser(pydantic_object=SemanticTriplesFactsReport)

    prompt_data = _prepare_prompt_data(state)
    prompt_data["preamble"] = preamble_first_visit
    prompt_data["critique_instruction"] = ""

    prompt = _create_prompt_template()

    try:
        response = llm_tool(
            prompt.format_prompt(
                format_instructions=parser.get_format_instructions(), **prompt_data
            )
        )

        proj = parser.parse(response.content)
        proj.semantic_graph.sanitize_prefixes_namespaces()
        state.current_chunk.graph = proj.semantic_graph

        state.clear_failure()
        return state

    except Exception as e:
        return _handle_rendering_error(state, e, FailureStage.GENERATE_TTL_FOR_FACTS)


def render_facts_update(state: AgentState, tools: ToolBox) -> AgentState:
    """Render facts updates using SPARQL operations.

    Args:
        state: The current agent state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with rendered facts.
    """
    logger.info("Rendering updates for facts")
    llm_tool = tools.llm
    parser = PydanticOutputParser(pydantic_object=GraphUpdate)

    prompt_data = _prepare_prompt_data(state)
    prompt_data["preamble"] = preamble_subsequent_visit

    if state.improvements_suggestions:
        prompt_data["critique_instruction"] = critique_instruction_template.format(
            "\n- ".join(state.improvements_suggestions)
        )
    else:
        prompt_data["critique_instruction"] = ""

    prompt = _create_prompt_template()

    try:
        response = llm_tool(
            prompt.format_prompt(
                format_instructions=parser.get_format_instructions(), **prompt_data
            )
        )

        graph_update = parser.parse(response.content)
        state.facts_updates.append(graph_update)
        state.set_node_status(WorkflowNode.TEXT_TO_FACTS, Status.SUCCESS)
        state.clear_failure()
        return state

    except Exception as e:
        return _handle_rendering_error(
            state, e, FailureStage.GENERATE_SPARQL_UPDATE_FOR_FACTS
        )
