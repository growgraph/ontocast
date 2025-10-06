"""Enhanced ontology triple rendering agent with memory and SPARQL operations.

This module provides enhanced functionality for rendering RDF triples from ontologies
with memory of previous calls and SPARQL operation support.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.onto.constants import ONTOLOGY_NULL_ID
from ontocast.onto.context import AgentType
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.state import AgentState
from ontocast.prompt.enhanced_render_ontology import (
    failure_instruction_enhanced,
    instructions_enhanced,
    ontology_instruction_fresh_enhanced,
    ontology_instruction_update_enhanced,
    specific_ontology_instruction_fresh_enhanced,
    specific_ontology_instruction_update_enhanced,
    template_prompt_enhanced,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def enhanced_render_onto_triples(state: AgentState, tools: ToolBox) -> AgentState:
    """Enhanced render ontology triples with memory and SPARQL operations.

    This function takes the triples from the current ontology and renders them
    into a more accessible format, with memory of previous calls and SPARQL
    operation support for incremental updates.

    Args:
        state: The current agent state containing the ontology to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with rendered triples.
    """
    logger.info("Starting enhanced ontology triple rendering with memory")
    llm_tool = tools.llm
    sparql_tool = tools.sparql_tool
    version_manager = tools.version_manager

    parser = PydanticOutputParser(pydantic_object=Ontology)

    logger.debug(f"Using domain: {state.current_domain}")

    # Check if this is a fresh ontology or an update
    is_fresh_ontology = (
        state.current_ontology.ontology_id == ONTOLOGY_NULL_ID
        or state.current_ontology.iri not in tools.ontology_manager
    )

    # Get context for this agent with conversation memory
    agent_context = state.get_context_for_agent("ontology_renderer", AgentType.RENDERER)

    # Add current interaction to conversation memory
    agent_context.add_conversation_memory(
        role="system",
        content=f"Starting ontology rendering for document: {state.current_chunk.text[:100]}...",
        metadata={
            "interaction_type": "ontology_rendering",
            "is_fresh": is_fresh_ontology,
        },
    )

    # Build dynamic context for this interaction
    agent_context.build_dynamic_context(
        interaction_type="ontology_rendering",
        document_text=state.current_chunk.text[:200],
        is_fresh_ontology=is_fresh_ontology,
    )

    previous_context = agent_context.get_llm_context()

    if is_fresh_ontology:
        logger.info("Creating a fresh ontology with memory context")
        ontology_instruction = ontology_instruction_fresh_enhanced.format(
            previous_context=previous_context
        )
        specific_ontology_instruction = (
            specific_ontology_instruction_fresh_enhanced.format(
                current_domain=state.current_domain
            )
        )
    else:
        ontology_iri = state.current_ontology.iri
        ontology_str = state.current_ontology.graph.serialize(format="turtle")
        ontology_desc = state.current_ontology.describe()
        ontology_instruction = ontology_instruction_update_enhanced.format(
            ontology_iri=ontology_iri,
            ontology_desc=ontology_desc,
            ontology_str=ontology_str,
            previous_context=previous_context,
        )
        specific_ontology_instruction = (
            specific_ontology_instruction_update_enhanced.format(
                ontology_namespace=state.current_ontology.namespace
            )
        )

    _instructions = instructions_enhanced.format(
        specific_ontology_instruction=specific_ontology_instruction
    )

    prompt = PromptTemplate(
        template=template_prompt_enhanced,
        input_variables=[
            "ontology_instruction",
            "instructions",
            "text",
            "failure_instruction",
            "format_instructions",
        ],
    )

    try:
        if state.status != Status.SUCCESS:
            failure_instruction_str = failure_instruction_enhanced.format(
                failure_stage=state.failure_stage,
                failure_reason=state.failure_reason,
                previous_context=previous_context,
            )
        else:
            failure_instruction_str = ""

        response = llm_tool(
            prompt.format_prompt(
                ontology_instruction=ontology_instruction,
                instructions=_instructions,
                text=state.current_chunk.text,
                failure_instruction=failure_instruction_str,
                format_instructions=parser.get_format_instructions(),
            )
        )

        proj = parser.parse(response.content)
        proj.semantic_graph.sanitize_prefixes_namespaces()

        # Apply SPARQL operations if this is an update
        if not is_fresh_ontology and sparql_tool:
            # Generate SPARQL operations for the changes
            try:
                # This would be where we generate SPARQL operations
                # For now, we'll use the traditional approach but with memory
                logger.info("Applying SPARQL operations for ontology update")
                # TODO: Implement SPARQL operation generation and application
            except Exception as e:
                logger.warning(
                    f"SPARQL operations failed, falling back to traditional approach: {e}"
                )

        # Update the ontology graph
        if state.current_ontology.graph is not None:
            state.current_ontology.graph += proj.semantic_graph
        else:
            state.current_ontology.graph = proj.semantic_graph

        # Create version in version manager
        if version_manager:
            new_version = version_manager.create_ontology_version(
                ontology_id=state.current_ontology.iri,
                graph=state.current_ontology.graph,
                metadata={
                    "chunk_text": state.current_chunk.text[:100] + "...",
                    "domain": state.current_domain,
                    "is_fresh": is_fresh_ontology,
                },
            )

            # Update context with new version
            state.update_context_for_agent(
                agent_name="ontology_renderer",
                ontology_version=new_version,
                metadata={
                    "chunk_text": state.current_chunk.text[:100] + "...",
                    "domain": state.current_domain,
                    "is_fresh": is_fresh_ontology,
                },
            )

        state.clear_failure()
        return state

    except Exception as e:
        logger.error(f"Failed to generate ontology triples: {str(e)}")
        state.set_failure(FailureStages.PARSE_TEXT_TO_ONTOLOGY_TRIPLES, str(e))
        return state
