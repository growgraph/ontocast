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
from ontocast.onto.enum import FailureStages, Status, WorkflowNode
from ontocast.onto.ontology import Ontology
from ontocast.onto.sparql_models import StructuredSPARQLQueryModel
from ontocast.onto.state import AgentState
from ontocast.prompt.render_ontology import (
    general_ontology_instruction,
    intro_instruction_first_visit_no_seed,
    intro_instruction_first_visit_seed,
    output_instruction_sparql,
    output_instruction_ttl,
    system_preamble,
    template_prompt,
)
from ontocast.prompt.render_ontology_update import (
    failure_instruction,
    ontology_sparql_prompt_template,
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
        or state.statuses[WorkflowNode.TEXT_TO_ONTOLOGY] == Status.NOT_VISITED
    )

    if is_fresh_ontology:
        logger.info("Generating fresh ontology as Turtle")
        return render_onto_first_visit(state, tools)

    else:
        logger.info("Generating ontology updates as SPARQL operations")
        sparql_operations = generate_onto_update_sparql(
            state, tools, previous_context_str
        )

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


def render_onto_first_visit(state: AgentState, tools: ToolBox) -> AgentState:
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
    if state.current_ontology.ontology_id == ONTOLOGY_NULL_ID:
        logger.info("Creating fresh ontology")
        intro_instruction = intro_instruction_first_visit_no_seed
        output_instruction = output_instruction_ttl
        ontology_ttl = ""
    else:
        ontology_iri = state.current_ontology.iri
        ontology_desc = state.current_ontology.describe()
        intro_instruction = intro_instruction_first_visit_seed.format(
            ontology_iri=ontology_iri,
            ontology_desc=ontology_desc)
        ontology_ttl = state.current_ontology.graph.serialize(format="turtle")
        output_instruction = output_instruction_sparql

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "system_preamble",
            "intro_instruction",
            "ontology_instruction",
            "output_instruction",
            "ontology_ttl",
            "text",
            "format_instructions",
        ],
    )

    try:
        response = tools.llm(
            prompt.format_prompt(
                system_preamble=system_preamble,
                intro_instruction=intro_instruction,
                ontology_instruction=general_ontology_instruction,
                output_instruction=output_instruction,
                ontology_ttl=ontology_ttl,
                text=state.current_chunk.text,
                format_instructions=parser.get_format_instructions(),
            )
        )

        state.ontology_addendum = parser.parse(response.content)
        state.ontology_addendum.graph.sanitize_prefixes_namespaces()

        logger.info(
            f"Ontology addendum has {len(state.ontology_addendum.graph)} triples."
        )
        state.clear_failure()
        state.statuses[WorkflowNode.TEXT_TO_ONTOLOGY] = Status.SUCCESS
        return state

    except Exception as e:
        logger.error(f"Failed to generate triples: {str(e)}")
        state.statuses[WorkflowNode.TEXT_TO_ONTOLOGY] = Status.FAILED
        state.set_failure(FailureStages.GENERATE_TTL_FOR_ONTOLOGY, str(e))
        return state


def _build_ontology_sparql_prompt(
    state: AgentState, document: str, ontology_id: str
) -> str:
    """Build prompt for ontology SPARQL updates.

    Args:
        document: Input document
        ontology_id: Current ontology ID

    Returns:
        Formatted prompt string
    """
    # Get current ontology description
    ontology_desc = f"Ontology ID: {ontology_id}\n"

    # Build the prompt using structured SPARQL template

    if state.status != Status.SUCCESS and state.failure_reason is not None:
        _failure_instruction = failure_instruction.format(
            failure_stage=state.failure_stage,
            failure_reason=state.failure_reason,
        )
    else:
        _failure_instruction = ""

    prompt = PromptTemplate(
        template=ontology_sparql_prompt_template,
        input_variables=["document"],
    )

    return prompt.format(document=document)


def generate_onto_update_sparql(
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
        ontology_iri = state.current_ontology.iri
        ontology_str = state.current_ontology.graph.serialize(format="turtle")
        ontology_desc = state.current_ontology.describe()
        # ontology_instruction = ontology_instruction_update.format(
        #     ontology_iri=ontology_iri,
        #     ontology_desc=ontology_desc,
        #     ontology_str=ontology_str,
        # )
        # specific_ontology_instruction = specific_ontology_instruction_update.format(
        #     ontology_namespace=state.current_ontology.namespace
        # )

        llm_tool = tools.llm
        ontology_id = state.ontology_id

        # Build prompt for SPARQL updates
        prompt = _build_ontology_sparql_prompt(
            state.document, ontology_id, previous_context
        )

        # Parse response with Pydantic
        parser = PydanticOutputParser(pydantic_object=StructuredSPARQLQueryModel)

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
