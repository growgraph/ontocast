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
from ontocast.prompt.render_facts import (
    critique_instruction_template,
    facts_instruction_template,
    ontology_instruction_template,
    preamble_first_visit,
    preamble_subsequent_visit,
    template_prompt,
    text_instruction_template,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def hybrid_render_facts(state: AgentState, tools: ToolBox) -> AgentState:
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
        return render_facts_first_visit(state, tools)
    else:
        logger.info("Generating facts update")
        return render_facts_subsequent_visit(state, tools)


def render_facts_first_visit(state: AgentState, tools: ToolBox) -> AgentState:
    """Render facts from the current chunk into a human-readable format.

    This function takes the facts in the current chunk and renders them into a
    more accessible format, making the extracted knowledge easier to understand.

    Args:
        state: The current agent state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with rendered facts.
    """
    logger.info("Rendering fresh facts")
    llm_tool = tools.llm
    parser = PydanticOutputParser(pydantic_object=SemanticTriplesFactsReport)

    preamble_str = preamble_first_visit

    ontology_instruction = ontology_instruction_template.format(
        ontology_str=state.current_ontology.graph.serialize(format="turtle")
    )

    facts_instruction_str = facts_instruction_template.format(
        ontology_namespace=state.current_ontology.namespace,
        current_doc_namespace=DEFAULT_CHUNK_IRI,
    )

    text_instruction = text_instruction_template.format(text=state.current_chunk.text)
    critique_instruction = ""

    prompt = PromptTemplate(
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
    try:
        response = llm_tool(
            prompt.format_prompt(
                preamble=preamble_str,
                facts_instruction=facts_instruction_str,
                ontology_instruction=ontology_instruction,
                text_instruction=text_instruction,
                critique_instruction=critique_instruction,
                format_instructions=parser.get_format_instructions(),
            )
        )

        proj = parser.parse(response.content)
        proj.semantic_graph.sanitize_prefixes_namespaces()
        state.current_chunk.graph = proj.semantic_graph

        state.clear_failure()
        return state

    except Exception as e:
        logger.error(f"Failed to generate triples: {str(e)}")
        state.set_failure(FailureStage.GENERATE_TTL_FOR_FACTS, str(e))
        return state


def render_facts_subsequent_visit(state: AgentState, tools: ToolBox) -> AgentState:
    """Render facts from the current chunk into a human-readable format.

    This function takes the facts in the current chunk and renders them into a
    more accessible format, making the extracted knowledge easier to understand.

    Args:
        state: The current agent state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with rendered facts.
    """
    logger.info("Rendering updates for facts")
    llm_tool = tools.llm

    parser = PydanticOutputParser(pydantic_object=GraphUpdate)

    preamble_str = preamble_subsequent_visit

    facts_instruction_str = facts_instruction_template.format(
        ontology_namespace=state.current_ontology.namespace,
        current_doc_namespace=DEFAULT_CHUNK_IRI,
    )

    ontology_instruction = ""
    text_instruction = ""

    if state.improvements_suggestions:
        critique_instruction_str = critique_instruction_template.format(
            "\n- ".join(state.improvements_suggestions)
        )
    else:
        critique_instruction_str = ""

    prompt = PromptTemplate(
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
    try:
        response = llm_tool(
            prompt.format_prompt(
                preamble=preamble_str,
                facts_instruction=facts_instruction_str,
                ontology_instruction=ontology_instruction,
                text_instruction=text_instruction,
                critique_instruction=critique_instruction_str,
                format_instructions=parser.get_format_instructions(),
            )
        )

        graph_update = parser.parse(response.content)

        state.facts_updates.append(graph_update)
        state.set_node_status(WorkflowNode.TEXT_TO_FACTS, Status.SUCCESS)
        state.clear_failure()
        return state

    except Exception as e:
        logger.error(f"Failed to generate triples: {str(e)}")
        state.set_failure(FailureStage.GENERATE_SPARQL_UPDATE_FOR_FACTS, str(e))
        return state
