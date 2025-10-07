"""Fact rendering agent for OntoCast.

This module provides functionality for rendering facts from RDF graphs into
human-readable formats, making the extracted knowledge more accessible and
understandable.
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.onto.constants import DEFAULT_CHUNK_IRI
from ontocast.onto.context import AgentType, Role
from ontocast.onto.enum import FailureStages, Status
from ontocast.onto.model import SemanticTriplesFactsReport
from ontocast.onto.sparql_models import StructuredSPARQLQueryModel
from ontocast.onto.state import AgentState
from ontocast.prompt.render_facts import (
    ontology_instruction,
)
from ontocast.prompt.render_facts import (
    template_prompt as template_prompt_str,
)
from ontocast.prompt.render_facts_update import (
    pydantic_facts_format_instructions,
    structured_sparql_instruction,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def render_facts(state: AgentState, tools: ToolBox) -> AgentState:
    """Render facts from the current chunk into a human-readable format.

    This function takes the facts in the current chunk and renders them into a
    more accessible format, making the extracted knowledge easier to understand.

    Args:
        state: The current agent state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with rendered facts.
    """
    logger.info("Starting to render facts")
    llm_tool = tools.llm

    parser = PydanticOutputParser(pydantic_object=SemanticTriplesFactsReport)

    ontology_str = state.current_ontology.graph.serialize(format="turtle")

    ontology_instruction_str = ontology_instruction.format(
        ontology_iri=state.current_ontology.iri, ontology_str=ontology_str
    )

    prompt = PromptTemplate(
        template=template_prompt_str,
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
        if state.status != Status.SUCCESS and state.failure_reason is not None:
            failure_instruction = "# FAILURE INSTRUCTION\n"
            failure_instruction += "The previous attempt to generate triples failed."
            if state.failure_stage is not None:
                failure_instruction += (
                    f"\n\nIt failed at the stage: {state.failure_stage}"
                )
            failure_instruction += f"\n\n{state.failure_reason}"
            failure_instruction += (
                "\n\nPlease fix the errors "
                "and do your best to generate fact triples again."
            )
        else:
            failure_instruction = ""

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
        if state.current_chunk.graph is not None:
            state.current_chunk.graph += proj.semantic_graph

        state.clear_failure()
        return state

    except Exception as e:
        logger.error(f"Failed to generate triples: {str(e)}")
        state.set_failure(FailureStages.GENERATE_TTL_FOR_FACTS, str(e))
        return state


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
    logger.info("Structured hybrid facts rendering with Turtle/SPARQL decision")

    # Get context for this agent
    agent_context = state.get_context_for_agent(
        agent_type=AgentType.RENDERER_FACTS,
    )

    is_fresh_facts = len(state.current_chunk.graph) == 0

    if is_fresh_facts:
        logger.info("Generating fresh facts as Turtle")
        # Generate fresh facts as Turtle
        return render_facts(state, tools)
    else:
        logger.info("Generating facts updates as SPARQL operations")

        # Build previous context from memory
        previous_context = agent_context.get_conversation_context()
        if previous_context:
            previous_context_str = f"Previous context: {previous_context}"
        else:
            previous_context_str = "No previous context available."

        # Generate SPARQL operations for updates
        sparql_operations = _generate_facts_sparql_updates(
            state, tools, previous_context_str
        )

        # Update state with SPARQL operations
        if sparql_operations:
            # Store operations in context for later execution
            agent_context.add_conversation_memory(
                role=Role.SYSTEM,
                content=f"Generated {len(sparql_operations.operations)} SPARQL operations for facts updates",
                metadata={
                    "type": "facts_sparql_operations",
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
            state.failure_stage = FailureStages.GENERATE_SPARQL_UPDATE_FOR_FACTS
            logger.error("Failed to generate SPARQL operations")

    # Update context for this agent
    state.update_context_for_agent(
        agent_type=AgentType.RENDERER_FACTS,
        facts_version=None,  # Will be created after execution
        metadata={
            "is_fresh_facts": is_fresh_facts,
            "previous_context": previous_context_str,
        },
    )

    return state


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

        # Build prompt for SPARQL updates
        prompt = _build_facts_sparql_prompt(
            state.document, state.failure_previous_context
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


def _build_facts_sparql_prompt(document: str, previous_context: str) -> str:
    """Build prompt for facts SPARQL updates.

    Args:
        document: Input document
        previous_context: Previous context string

    Returns:
        Formatted prompt string
    """
    # Build the prompt using structured SPARQL template
    prompt_template = f"""
{structured_sparql_instruction}

{previous_context}

Document to process:
{{document}}

{pydantic_facts_format_instructions}
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["document"],
    )

    return prompt.format(document=document)
