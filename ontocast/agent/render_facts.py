"""Fact rendering agent for OntoCast.

This module provides functionality for rendering facts from RDF graphs into
human-readable formats, making the extracted knowledge more accessible and
understandable.
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry, render_suggestions_prompt
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import FactsRenderReport, GraphUpdateRenderReport
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.common import (
    facts_template,
    ontology_template,
    output_instruction_empty,
    output_instruction_sparql,
    text_template,
    user_template,
)
from ontocast.prompt.render_facts import (
    facts_instruction_template,
    preamble,
    template_prompt,
)
from ontocast.tool.atomic import AtomicToolBox

logger = logging.getLogger(__name__)


def _extract_known_prefixes(state: UnitFactsState) -> dict[str, str]:
    """Extract ontology prefixes used to patch missing declarations in LLM TTL output."""
    known_prefixes: dict[str, str] = {}

    if state.ontology_snapshot and state.ontology_snapshot.graph:
        for prefix, namespace_uri in state.ontology_snapshot.graph.namespaces():
            if prefix:  # Skip empty prefixes
                known_prefixes[prefix] = str(namespace_uri)

    # Also add the ontology prefix explicitly if available.
    if state.ontology_snapshot.prefix and state.ontology_snapshot.namespace:
        known_prefixes[state.ontology_snapshot.prefix] = (
            state.ontology_snapshot.namespace
        )

    return known_prefixes


async def render_facts(state: UnitFactsState, tools: AtomicToolBox) -> UnitFactsState:
    """Structured hybrid facts renderer with Turtle/SPARQL decision logic.

    This function decides between generating bare Turtle for fresh facts
    and SPARQL operations for updates based on whether facts exist.

    Args:
        state: The current unit facts state
        tools: The toolbox containing necessary tools

    Returns:
        UnitFactsState: Updated state with rendered facts
    """

    is_fresh_facts_graph = len(state.content_unit.graph) == 0

    progress_info = state.get_content_unit_progress_string()
    logger.info(f"Render facts for {progress_info}")

    if is_fresh_facts_graph:
        logger.info("Generating fresh facts as Turtle")
        return await render_facts_fresh(state, tools)
    else:
        logger.info("Generating facts update")
        return await render_facts_update(state, tools)


def _prepare_prompt_data(state: UnitFactsState) -> dict[str, str]:
    """Prepare common prompt data for both fresh and update rendering.

    Args:
        state: The current unit facts state

    Returns:
        Dictionary containing formatted prompt components
    """
    ontology_chapter = ontology_template.format(
        ontology_ttl=state.ontology_snapshot.graph.serialize(format="turtle")
    )

    facts_instruction_str = facts_instruction_template.format(
        ontology_namespace=state.ontology_snapshot.namespace,
        ontology_prefix=state.ontology_snapshot.prefix,
        facts_namespace=DEFAULT_IRI,
    )

    text_chapter = text_template.format(text=state.content_unit.text)

    fact_chapter = ""

    user_instruction = (
        user_template.format(user_instruction=state.facts_user_instruction)
        if state.facts_user_instruction
        else ""
    )

    return {
        "ontology_chapter": ontology_chapter,
        "user_instruction": user_instruction,
        "facts_instruction": facts_instruction_str,
        "text_chapter": text_chapter,
        "fact_chapter": fact_chapter,
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
            "user_instruction",
            "ontology_chapter",
            "text_chapter",
            "improvement_instruction",
            "output_instruction",
            "format_instructions",
        ],
    )


def _handle_rendering_error(
    state: UnitFactsState, error: Exception, stage: FailureStage
) -> UnitFactsState:
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
    state.set_node_status(WorkflowNode.TEXT_TO_FACTS, Status.FAILED)
    return state


async def render_facts_fresh(
    state: UnitFactsState, tools: AtomicToolBox
) -> UnitFactsState:
    """Render fresh facts from the current chunk into Turtle format.

    Args:
        state: The current unit facts state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        UnitFactsState: Updated state with rendered facts.
    """
    logger.info("Rendering fresh facts")
    llm_tool = await tools.get_llm_tool(state.budget_tracker)
    parser = PydanticOutputParser(pydantic_object=FactsRenderReport)

    known_prefixes = _extract_known_prefixes(state)

    prompt_data = _prepare_prompt_data(state)
    prompt_data_fresh = {
        "preamble": preamble,
        "improvement_instruction": "",
        "output_instruction": output_instruction_empty,
    }
    prompt_data.update(prompt_data_fresh)

    prompt = _create_prompt_template()

    try:
        # Set known prefixes in context before parsing
        RDFGraph.set_known_prefixes(known_prefixes if known_prefixes else None)

        render_report: FactsRenderReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "format_instructions": parser.get_format_instructions(),
                **prompt_data,
            },
        )
        state.set_external_evidence_request(
            WorkflowNode.TEXT_TO_FACTS, render_report.external_evidence_request
        )
        facts_report = render_report.facts_report
        facts_report.semantic_graph.sanitize_prefixes_namespaces()
        state.content_unit.graph = facts_report.semantic_graph

        # Track triples in budget tracker (fresh facts)
        num_triples = len(facts_report.semantic_graph)
        logger.info(f"Fresh facts generated with {num_triples} triple(s).")
        state.budget_tracker.add_facts_update(num_operations=1, num_triples=num_triples)

        state.clear_failure()
        state.set_node_status(WorkflowNode.TEXT_TO_FACTS, Status.SUCCESS)
        return state

    except Exception as e:
        return _handle_rendering_error(state, e, FailureStage.GENERATE_TTL_FOR_FACTS)
    finally:
        # Clear the context after parsing
        RDFGraph.set_known_prefixes(None)


async def render_facts_update(
    state: UnitFactsState, tools: AtomicToolBox
) -> UnitFactsState:
    """Render facts updates using SPARQL operations.

    Args:
        state: The current unit facts state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        UnitFactsState: Updated state with rendered facts.
    """
    logger.info("Rendering updates for facts")
    llm_tool = await tools.get_llm_tool(state.budget_tracker)
    parser = PydanticOutputParser(pydantic_object=GraphUpdateRenderReport)

    prompt_data = _prepare_prompt_data(state)
    prompt_data_update = {
        "preamble": preamble,
        "improvement_instruction": render_suggestions_prompt(
            state.suggestions, WorkflowNode.TEXT_TO_FACTS
        ),
        "output_instruction": output_instruction_sparql,
        "fact_chapter": facts_template.format(
            facts_ttl=state.content_unit.graph.serialize(format="turtle")
        ),
    }
    prompt_data.update(prompt_data_update)
    prompt = _create_prompt_template()
    known_prefixes = _extract_known_prefixes(state)

    try:
        # Set known prefixes in context before parsing
        RDFGraph.set_known_prefixes(known_prefixes if known_prefixes else None)

        render_report: GraphUpdateRenderReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "format_instructions": parser.get_format_instructions(),
                **prompt_data,
            },
        )
        state.set_external_evidence_request(
            WorkflowNode.TEXT_TO_FACTS, render_report.external_evidence_request
        )
        graph_update = render_report.graph_update
        state.facts_updates.append(graph_update)
        state.update_facts()

        num_operations, num_triples = graph_update.count_total_triples()
        logger.info(
            f"Facts update has {num_operations} operation(s) "
            f"with {num_triples} total triple(s)."
        )

        # Track triples in budget tracker
        state.budget_tracker.add_facts_update(num_operations, num_triples)

        state.set_node_status(WorkflowNode.TEXT_TO_FACTS, Status.SUCCESS)
        state.clear_failure()
        return state

    except Exception as e:
        return _handle_rendering_error(
            state, e, FailureStage.GENERATE_SPARQL_UPDATE_FOR_FACTS
        )
    finally:
        # Clear the context after parsing
        RDFGraph.set_known_prefixes(None)
