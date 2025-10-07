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
from ontocast.onto.context import AgentType, Role
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.sparql_models import StructuredSPARQLQueryModel
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
from ontocast.prompt.structured_sparql_ontology import (
    pydantic_format_instructions,
    structured_sparql_instruction,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def hybrid_render_ontology(state: AgentState, tools: ToolBox) -> AgentState:
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
        agent_type=AgentType.RENDERER_ONTOLOGY,
    )

    # Build previous context from memory
    previous_context = agent_context.get_conversation_context()
    if previous_context:
        previous_context_str = f"Previous context: {previous_context}"
    else:
        previous_context_str = "No previous context available."

    is_fresh_ontology = (
        state.ontology_id == ONTOLOGY_NULL_ID
        or state.ontology_id not in tools.ontology_manager
    )

    if is_fresh_ontology:
        logger.info("Generating fresh ontology as Turtle")
        # Generate fresh ontology as Turtle
        turtle_result = _generate_fresh_ontology_turtle(state, tools)

        # Update state with fresh ontology
        if turtle_result:
            state.ontology = turtle_result
            state.status = Status.SUCCESS
            logger.info("Fresh ontology generated successfully")
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStages.GENERATE_TTL_FOR_ONTOLOGY
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
                role=Role.SYSTEM,
                content=f"Generated {len(sparql_operations.operations)} SPARQL operations for ontology updates",
                metadata={
                    "type": "ontology_sparql_operations",
                    "operations": [
                        op.model_dump_json() for op in sparql_operations.operations
                    ],
                    "namespaces": sparql_operations.namespaces,
                },
            )
            state.status = Status.SUCCESS
            logger.info(
                f"Generated {len(sparql_operations.operations)} SPARQL operations"
            )
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStages.GENERATE_SPARQL_UPDATE_FOR_ONTOLOGY
            logger.error("Failed to generate SPARQL operations")

    # Update context for this agent
    state.update_context_for_agent(
        agent_type=AgentType.RENDERER_ONTOLOGY,
        ontology_version=None,  # Will be created after execution
        metadata={
            "is_fresh_ontology": is_fresh_ontology,
            "previous_context": previous_context_str,
        },
    )

    return state


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
        state.set_failure(FailureStages.GENERATE_TTL_FOR_ONTOLOGY, str(e))
        return state


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
        response = llm_tool(prompt)
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


def _generate_fresh_ontology_turtle(
    state: AgentState, tools: ToolBox
) -> Ontology | None:
    """Generate fresh ontology as Turtle using the original renderer.

    Args:
        state: Current agent state
        tools: Toolbox with necessary tools

    Returns:
        Ontology object or None if failed
    """
    try:
        # Create a temporary state for the original renderer
        # temp_state = AgentState(
        #     document=state.document,
        #     ontology_id=state.current_ontology.ontology_id,
        #     skip_ontology_development=state.skip_ontology_development,
        #     max_visits=state.max_visits,
        #     context_manager=state.context_manager,
        # )

        # Call the original renderer
        result_state = render_onto_triples(state, tools)

        if result_state.status == Status.SUCCESS:
            return result_state.ontology
        else:
            logger.error("Original renderer failed to generate fresh ontology")
            return None

    except Exception as e:
        logger.error(f"Error generating fresh ontology: {e}")
        return None
