"""Structured hybrid ontology renderer with Turtle/SPARQL decision logic.

This module provides a hybrid renderer that decides between generating
bare Turtle for fresh ontologies and SPARQL operations for updates.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.onto.constants import ONTOLOGY_NULL_ID
from ontocast.onto.context import AgentType
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.sparql_models import StructuredSPARQLQueryModel
from ontocast.onto.state import AgentState
from ontocast.prompt.structured_sparql_ontology import (
    pydantic_format_instructions,
    structured_sparql_instruction,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def structured_hybrid_render_ontology(state: AgentState, tools: ToolBox) -> AgentState:
    """Structured hybrid ontology renderer with Turtle/SPARQL decision logic.

    This function decides between generating bare Turtle for fresh ontologies
    and SPARQL operations for updates based on whether the ontology exists.

    Args:
        state: The current agent state
        tools: The toolbox containing necessary tools

    Returns:
        AgentState: Updated state with rendered ontology
    """
    logger.info("Structured hybrid ontology rendering with Turtle/SPARQL decision")

    # Get context for this agent
    agent_context = state.get_context_for_agent(
        agent_name="structured_hybrid_ontology_renderer",
        agent_type=AgentType.RENDERER,
    )

    # Build previous context from memory
    previous_context = agent_context.get_conversation_context()
    if previous_context:
        previous_context_str = f"Previous context: {previous_context}"
    else:
        previous_context_str = "No previous context available."

    # Determine if this is a fresh ontology or an update
    ontology_id = state.ontology_id
    is_fresh_ontology = (
        ontology_id == ONTOLOGY_NULL_ID or ontology_id not in tools.ontology_manager
    )

    if is_fresh_ontology:
        logger.info("Generating fresh ontology as Turtle")
        # Generate fresh ontology as Turtle
        turtle_result = _generate_fresh_ontology_turtle(
            state, tools, previous_context_str
        )

        # Update state with fresh ontology
        if turtle_result:
            state.ontology = turtle_result
            state.status = Status.SUCCESS
            logger.info("Fresh ontology generated successfully")
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStages.RENDER_ONTOLOGY
            logger.error("Failed to generate fresh ontology")
    else:
        logger.info("Generating ontology updates as SPARQL operations")
        # Generate SPARQL operations for updates
        sparql_operations = _generate_ontology_sparql_updates(
            state, tools, previous_context_str
        )

        # Update state with SPARQL operations
        if sparql_operations:
            # Store operations in context for later execution
            agent_context.add_conversation_memory(
                {
                    "type": "ontology_sparql_operations",
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
            state.failure_stage = FailureStages.RENDER_ONTOLOGY
            logger.error("Failed to generate SPARQL operations")

    # Update context for this agent
    state.update_context_for_agent(
        agent_name="structured_hybrid_ontology_renderer",
        ontology_version=None,  # Will be created after execution
        metadata={
            "is_fresh_ontology": is_fresh_ontology,
            "previous_context": previous_context_str,
        },
    )

    return state


def _generate_fresh_ontology_turtle(
    state: AgentState, tools: ToolBox, previous_context: str
) -> Ontology | None:
    """Generate fresh ontology as Turtle using the original renderer.

    Args:
        state: Current agent state
        tools: Toolbox with necessary tools
        previous_context: Previous context string

    Returns:
        Ontology object or None if failed
    """
    try:
        # Use the original render_onto_triples logic for fresh content
        from ontocast.agent.render_ontology_triples import render_onto_triples

        # Create a temporary state for the original renderer
        temp_state = AgentState(
            document=state.document,
            ontology_id=state.ontology_id,
            skip_ontology_development=state.skip_ontology_development,
            max_visits=state.max_visits,
            context_manager=state.context_manager,
        )

        # Call the original renderer
        result_state = render_onto_triples(temp_state, tools)

        if result_state.status == Status.SUCCESS:
            return result_state.ontology
        else:
            logger.error("Original renderer failed to generate fresh ontology")
            return None

    except Exception as e:
        logger.error(f"Error generating fresh ontology: {e}")
        return None


def _generate_ontology_sparql_updates(
    state: AgentState, tools: ToolBox, previous_context: str
) -> StructuredSPARQLQueryModel | None:
    """Generate SPARQL operations for ontology updates.

    Args:
        state: Current agent state
        tools: Toolbox with necessary tools
        previous_context: Previous context string

    Returns:
        StructuredSPARQLQueryModel or None if failed
    """
    try:
        llm_tool = tools.llm
        ontology_id = state.ontology_id

        # Get current ontology for context
        current_ontology = tools.ontology_manager.get_ontology(ontology_id)
        if not current_ontology:
            logger.error(f"Could not find current ontology: {ontology_id}")
            return None

        # Build prompt for SPARQL updates
        prompt = _build_ontology_sparql_prompt(
            state.document, ontology_id, current_ontology, previous_context
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


def _build_ontology_sparql_prompt(
    document: str, ontology_id: str, current_ontology: Ontology, previous_context: str
) -> str:
    """Build prompt for ontology SPARQL updates.

    Args:
        document: Input document
        ontology_id: Current ontology ID
        current_ontology: Current ontology object
        previous_context: Previous context string

    Returns:
        Formatted prompt string
    """
    # Get current ontology description
    ontology_desc = f"Ontology ID: {ontology_id}\n"
    if hasattr(current_ontology, "description"):
        ontology_desc += f"Description: {current_ontology.description}\n"

    # Build the prompt using structured SPARQL template
    prompt_template = f"""
{structured_sparql_instruction}

{previous_context}

Document to process:
{{document}}

Current Ontology:
{ontology_desc}

{pydantic_format_instructions}
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["document"],
    )

    return prompt.format(document=document)
