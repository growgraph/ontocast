"""Ontology triple rendering agent for OntoCast.

This module provides functionality for rendering RDF triples from ontologies into
human-readable formats, making the ontological knowledge more accessible and
understandable.
The agent decides between generating bare Turtle for fresh ontologies and SPARQL operations for updates.

"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry, render_suggestions_prompt
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.ontology import Ontology
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState
from ontocast.prompt.common import (
    ontology_template,
    output_instruction_sparql,
    output_instruction_ttl,
    text_template,
)
from ontocast.prompt.common import system_preamble_ontology as system_preamble
from ontocast.prompt.render_ontology import (
    general_ontology_instruction,
    intro_instruction_fresh,
    intro_instruction_update,
    prefix_instruction,
    prefix_instruction_fresh,
    template_prompt,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


async def render_ontology(state: AgentState, tools: ToolBox) -> AgentState:
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
        f"Ontology Renderer for {progress_info}: visit {state.node_visits[WorkflowNode.TEXT_TO_ONTOLOGY] + 1}/{state.max_visits}"
    )
    has_no_seed_ontology = state.current_ontology.is_null()

    if has_no_seed_ontology:
        return await render_ontology_fresh(state, tools)
    else:
        return await render_ontology_update(state, tools)


async def render_ontology_fresh(state: AgentState, tools: ToolBox) -> AgentState:
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
    intro_instruction = intro_instruction_fresh.format(
        current_domain=state.current_domain
    )
    output_instruction = output_instruction_ttl
    ontology_ttl = ""
    improvement_instruction_str = ""
    general_ontology_instruction_str = general_ontology_instruction.format(
        prefix_instruction=prefix_instruction_fresh
    )

    text_chapter = text_template.format(text=state.current_chunk.text)

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
        llm_tool = await tools.get_llm_tool(state.budget_tracker)
        state.current_ontology = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "preamble": system_preamble,
                "intro_instruction": intro_instruction,
                "ontology_instruction": general_ontology_instruction_str,
                "output_instruction": output_instruction,
                "ontology_ttl": ontology_ttl,
                "user_instruction": state.ontology_user_instruction,
                "improvement_instruction": improvement_instruction_str,
                "text": text_chapter,
                "format_instructions": parser.get_format_instructions(),
            },
        )
        state.current_ontology.graph.sanitize_prefixes_namespaces()

        num_triples = len(state.current_ontology.graph)
        logger.info(f"New ontology created with {num_triples} triple(s).")

        # Track triples in budget tracker (fresh ontology)
        state.budget_tracker.add_ontology_update(
            num_operations=1, num_triples=num_triples
        )

        state.clear_failure()
        state.set_node_status(WorkflowNode.TEXT_TO_ONTOLOGY, Status.SUCCESS)
        return state

    except Exception as e:
        logger.error(f"Failed to generate triples: {str(e)}")
        state.set_node_status(WorkflowNode.TEXT_TO_ONTOLOGY, Status.FAILED)
        state.set_failure(FailureStage.GENERATE_TTL_FOR_ONTOLOGY, str(e))
        return state


async def render_ontology_update(state: AgentState, tools: ToolBox) -> AgentState:
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
    ontology_chapter = ontology_template.format(
        ontology_ttl=state.current_ontology.graph.serialize(format="turtle")
    )
    output_instruction = output_instruction_sparql
    improvement_instruction_str = render_suggestions_prompt(
        state.suggestions, WorkflowNode.TEXT_TO_ONTOLOGY
    )

    general_ontology_instruction_str = general_ontology_instruction.format(
        prefix_instruction=prefix_instruction.format(
            ontology_prefix=state.current_ontology.prefix
        )
    )
    text_chapter = text_template.format(text=state.current_chunk.text)

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
        llm_tool = await tools.get_llm_tool(state.budget_tracker)
        graph_update: GraphUpdate = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "preamble": system_preamble,
                "intro_instruction": intro_instruction,
                "ontology_instruction": general_ontology_instruction_str,
                "output_instruction": output_instruction,
                "improvement_instruction": improvement_instruction_str,
                "ontology_ttl": ontology_chapter,
                "user_instruction": state.ontology_user_instruction,
                "text": text_chapter,
                "format_instructions": parser.get_format_instructions(),
            },
        )
        state.ontology_updates.append(graph_update)
        state.update_ontology()

        num_operations, num_triples = graph_update.count_total_triples()
        logger.info(
            f"Ontology update has {num_operations} operation(s) "
            f"with {num_triples} total triple(s)."
        )

        # Track triples in budget tracker
        state.budget_tracker.add_ontology_update(num_operations, num_triples)

        state.clear_failure()
        state.set_node_status(WorkflowNode.TEXT_TO_ONTOLOGY, Status.SUCCESS)
        return state

    except Exception as e:
        logger.error(f"Failed to generate ontology update: {str(e)}")
        state.set_node_status(WorkflowNode.TEXT_TO_ONTOLOGY, Status.FAILED)
        state.set_failure(FailureStage.GENERATE_SPARQL_UPDATE_FOR_ONTOLOGY, str(e))
        return state
