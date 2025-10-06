"""Enhanced fact rendering agent with memory and SPARQL operations.

This module provides enhanced functionality for rendering facts from RDF graphs
with memory of previous calls and SPARQL operation support.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.onto.constants import DEFAULT_CHUNK_IRI
from ontocast.onto.context import AgentType
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.model import SemanticTriplesFactsReport
from ontocast.onto.state import AgentState
from ontocast.prompt.enhanced_render_facts import (
    failure_instruction_enhanced,
    ontology_instruction_enhanced,
    template_prompt_enhanced,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def enhanced_render_facts(state: AgentState, tools: ToolBox) -> AgentState:
    """Enhanced render facts with memory and SPARQL operations.

    This function takes the facts in the current chunk and renders them into a
    more accessible format, with memory of previous calls and SPARQL operation
    support for incremental updates.

    Args:
        state: The current agent state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with rendered facts.
    """
    logger.info("Starting enhanced facts rendering with memory")
    llm_tool = tools.llm
    sparql_tool = tools.sparql_tool
    version_manager = tools.version_manager

    parser = PydanticOutputParser(pydantic_object=SemanticTriplesFactsReport)

    # Get context for this agent with conversation memory
    agent_context = state.get_context_for_agent("facts_renderer", AgentType.RENDERER)

    # Add current interaction to conversation memory
    if state.current_chunk:
        agent_context.add_conversation_memory(
            role="system",
            content=f"Starting facts rendering for chunk: {state.current_chunk.text[:100]}...",
            metadata={
                "interaction_type": "facts_rendering",
                "chunk_id": getattr(state.current_chunk, "chunk_id", "unknown"),
            },
        )

        # Build dynamic context for this interaction
        agent_context.build_dynamic_context(
            interaction_type="facts_rendering",
            chunk_text=state.current_chunk.text[:200],
            ontology_iri=state.current_ontology.iri,
        )
    else:
        agent_context.add_conversation_memory(
            role="system",
            content="Starting facts rendering for chunk: No chunk available",
            metadata={"interaction_type": "facts_rendering", "chunk_id": "unknown"},
        )

        # Build dynamic context for this interaction
        agent_context.build_dynamic_context(
            interaction_type="facts_rendering",
            chunk_text="No chunk available",
            ontology_iri=state.current_ontology.iri,
        )

    previous_context = agent_context.get_llm_context()

    ontology_str = state.current_ontology.graph.serialize(format="turtle")

    ontology_instruction_str = ontology_instruction_enhanced.format(
        ontology_iri=state.current_ontology.iri,
        ontology_str=ontology_str,
        previous_context=previous_context,
    )

    prompt = PromptTemplate(
        template=template_prompt_enhanced,
        input_variables=[
            "ontology_namespace",
            "current_doc_namespace",
            "text",
            "ontology_instruction",
            "failure_instruction",
            "format_instructions",
        ],
    )

    try:
        if state.status != Status.SUCCESS:
            failure_instruction = failure_instruction_enhanced.format(
                failure_stage=state.failure_stage,
                failure_reason=state.failure_reason,
                previous_context=previous_context,
            )
        else:
            failure_instruction = ""

        if not state.current_chunk:
            state.set_failure(
                FailureStages.PARSE_TEXT_TO_FACTS_TRIPLES, "No current chunk available"
            )
            return state

        response = llm_tool(
            prompt.format_prompt(
                ontology_namespace=state.current_ontology.namespace,
                ontology_prefix=state.current_ontology.prefix,
                current_doc_namespace=DEFAULT_CHUNK_IRI,
                text=state.current_chunk.text,
                ontology_instruction=ontology_instruction_str,
                failure_instruction=failure_instruction,
                format_instructions=parser.get_format_instructions(),
            )
        )

        proj = parser.parse(response.content)
        proj.semantic_graph.sanitize_prefixes_namespaces()

        # Apply SPARQL operations if this is an update
        if sparql_tool and version_manager and state.current_chunk:
            chunk_id = getattr(state.current_chunk, "chunk_id", "unknown")
            latest_facts_version = version_manager.get_latest_facts_version(chunk_id)
            if latest_facts_version:
                try:
                    # This would be where we generate SPARQL operations
                    logger.info("Applying SPARQL operations for facts update")
                    # TODO: Implement SPARQL operation generation and application
                except Exception as e:
                    logger.warning(
                        f"SPARQL operations failed, falling back to traditional approach: {e}"
                    )

        # Update the chunk graph
        if state.current_chunk and state.current_chunk.graph is not None:
            state.current_chunk.graph += proj.semantic_graph
        elif state.current_chunk:
            state.current_chunk.graph = proj.semantic_graph

        # Create version in version manager
        if version_manager and state.current_chunk and state.current_chunk.graph:
            chunk_id = getattr(state.current_chunk, "chunk_id", "unknown")
            new_version = version_manager.create_facts_version(
                chunk_id=chunk_id,
                graph=state.current_chunk.graph,
                metadata={
                    "chunk_text": state.current_chunk.text[:100] + "...",
                    "ontology_iri": state.current_ontology.iri,
                    "domain": state.current_domain,
                },
            )

            # Update context with new version
            state.update_context_for_agent(
                agent_name="facts_renderer",
                facts_version=new_version,
                metadata={
                    "chunk_text": state.current_chunk.text[:100] + "...",
                    "ontology_iri": state.current_ontology.iri,
                    "domain": state.current_domain,
                },
            )

        state.clear_failure()
        return state

    except Exception as e:
        logger.error(f"Failed to generate triples: {str(e)}")
        state.set_failure(FailureStages.PARSE_TEXT_TO_FACTS_TRIPLES, str(e))
        return state
