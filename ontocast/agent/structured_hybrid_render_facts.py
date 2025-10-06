"""Structured hybrid facts renderer with Turtle/SPARQL decision logic.

This module provides a hybrid renderer that decides between generating
bare Turtle for fresh facts and SPARQL operations for updates.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.onto.constants import DEFAULT_CHUNK_IRI
from ontocast.onto.context import AgentType
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import StructuredSPARQLQueryModel
from ontocast.onto.state import AgentState
from ontocast.prompt.structured_sparql_facts import (
    pydantic_facts_format_instructions,
    structured_sparql_instruction,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def structured_hybrid_render_facts(state: AgentState, tools: ToolBox) -> AgentState:
    """Structured hybrid facts renderer with Turtle/SPARQL decision logic.

    This function decides between generating bare Turtle for fresh facts
    and SPARQL operations for updates based on whether facts exist.

    Args:
        state: The current agent state
        tools: The toolbox containing necessary tools

    Returns:
        AgentState: Updated state with rendered facts
    """
    logger.info("Structured hybrid facts rendering with Turtle/SPARQL decision")

    if not state.current_chunk:
        logger.error("No current chunk available for facts rendering")
        state.status = Status.FAILED
        state.failure_stage = FailureStages.RENDER_FACTS
        return state

    # Get context for this agent
    agent_context = state.get_context_for_agent(
        agent_name="structured_hybrid_facts_renderer",
        agent_type=AgentType.RENDERER,
    )

    # Build previous context from memory
    previous_context = agent_context.get_conversation_context()
    if previous_context:
        previous_context_str = f"Previous context: {previous_context}"
    else:
        previous_context_str = "No previous context available."

    # Determine if this is fresh facts or an update
    current_facts = getattr(state.current_chunk, "graph", None)
    is_fresh_facts = (
        current_facts is None
        or not isinstance(current_facts, RDFGraph)
        or len(current_facts) == 0
    )

    if is_fresh_facts:
        logger.info("Generating fresh facts as Turtle")
        # Generate fresh facts as Turtle
        turtle_result = _generate_fresh_facts_turtle(state, tools, previous_context_str)

        # Update state with fresh facts
        if turtle_result:
            # Update the chunk with the new facts
            if not state.current_chunk.graph:
                state.current_chunk.graph = RDFGraph()
            state.current_chunk.graph += turtle_result
            state.status = Status.SUCCESS
            logger.info("Fresh facts generated successfully")
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStages.RENDER_FACTS
            logger.error("Failed to generate fresh facts")
    else:
        logger.info("Generating facts updates as SPARQL operations")
        # Generate SPARQL operations for updates
        sparql_operations = _generate_facts_sparql_updates(
            state, tools, previous_context_str
        )

        # Update state with SPARQL operations
        if sparql_operations:
            # Store operations in context for later execution
            agent_context.add_conversation_memory(
                {
                    "type": "facts_sparql_operations",
                    "operations": [op.dict() for op in sparql_operations.operations],
                    "namespaces": sparql_operations.namespaces,
                }
            )
            state.status = Status.SUCCESS
            logger.info(
                f"Generated {len(sparql_operations.operations)} SPARQL operations"
            )
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStages.RENDER_FACTS
            logger.error("Failed to generate SPARQL operations")

    # Update context for this agent
    state.update_context_for_agent(
        agent_name="structured_hybrid_facts_renderer",
        facts_version=None,  # Will be created after execution
        metadata={
            "is_fresh_facts": is_fresh_facts,
            "previous_context": previous_context_str,
        },
    )

    return state


def _generate_fresh_facts_turtle(
    state: AgentState, tools: ToolBox, previous_context: str
) -> RDFGraph | None:
    """Generate fresh facts as Turtle using the original renderer.

    Args:
        state: Current agent state
        tools: Toolbox with necessary tools
        previous_context: Previous context string

    Returns:
        RDFGraph object or None if failed
    """
    try:
        # Use the original render_facts logic for fresh content
        from ontocast.agent.render_facts import render_facts

        # Create a temporary state for the original renderer
        temp_state = AgentState(
            document=state.document,
            ontology_id=state.ontology_id,
            skip_ontology_development=state.skip_ontology_development,
            max_visits=state.max_visits,
            context_manager=state.context_manager,
            current_chunk=state.current_chunk,
        )

        # Call the original renderer
        result_state = render_facts(temp_state, tools)

        if result_state.status == Status.SUCCESS and result_state.current_chunk:
            return result_state.current_chunk.graph
        else:
            logger.error("Original renderer failed to generate fresh facts")
            return None

    except Exception as e:
        logger.error(f"Error generating fresh facts: {e}")
        return None


def _generate_facts_sparql_updates(
    state: AgentState, tools: ToolBox, previous_context: str
) -> StructuredSPARQLQueryModel | None:
    """Generate SPARQL operations for facts updates.

    Args:
        state: Current agent state
        tools: Toolbox with necessary tools
        previous_context: Previous context string

    Returns:
        StructuredSPARQLQueryModel or None if failed
    """
    try:
        llm_tool = tools.llm
        chunk_id = getattr(state.current_chunk, "chunk_id", DEFAULT_CHUNK_IRI)

        # Get current facts for context
        current_facts = getattr(state.current_chunk, "graph", None)
        if not current_facts or not isinstance(current_facts, RDFGraph):
            logger.error("Could not find current facts graph")
            return None

        # Build prompt for SPARQL updates
        prompt = _build_facts_sparql_prompt(
            state.document, chunk_id, current_facts, previous_context
        )

        # Parse response with Pydantic
        parser = PydanticOutputParser(pydantic_object=StructuredSPARQLQueryModel)

        # Get LLM response
        response = llm_tool.acreate(prompt)
        if not response or not response.content:
            logger.error("No response from LLM")
            return None

        # Parse the response
        structured_query = parser.parse(response.content)

        logger.info(
            f"Generated structured SPARQL query: {structured_query.get_summary()}"
        )
        return structured_query

    except Exception as e:
        logger.error(f"Error generating SPARQL operations: {e}")
        return None


def _build_facts_sparql_prompt(
    document: str, chunk_id: str, current_facts: RDFGraph, previous_context: str
) -> str:
    """Build prompt for facts SPARQL updates.

    Args:
        document: Input document
        chunk_id: Current chunk ID
        current_facts: Current facts graph
        previous_context: Previous context string

    Returns:
        Formatted prompt string
    """
    # Get current facts description
    facts_desc = f"Chunk ID: {chunk_id}\n"
    facts_desc += f"Current facts count: {len(current_facts)}\n"

    # Build the prompt using structured SPARQL template
    prompt_template = f"""
{structured_sparql_instruction}

{previous_context}

Document to process:
{{document}}

Current Facts:
{facts_desc}

{pydantic_facts_format_instructions}
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["document"],
    )

    return prompt.format(document=document)
