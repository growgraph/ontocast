"""Fact rendering agent for OntoCast.

This module provides functionality for rendering facts from RDF graphs into
human-readable formats, making the extracted knowledge more accessible and
understandable.
"""

import logging
from collections.abc import Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry, render_suggestions_prompt
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import FactsRenderReport, GraphUpdateRenderReport
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import (
    UnitFactsOntologyAccess,
    build_llm_prefix_map,
    ontology_access_for_unit_facts,
)
from ontocast.onto.rdfgraph import RDFGraph, finalize_llm_graph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.common import text_template, user_template
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.prompt.ontology_context import (
    build_ontology_index,
    format_ontologies_clause,
)
from ontocast.prompt.render_facts import (
    preamble,
    template_prompt,
)
from ontocast.tool.atomic import AtomicToolBox

logger = logging.getLogger(__name__)


async def render_facts(
    state: UnitFactsState,
    tools: AtomicToolBox,
    supplemental_ontologies: Sequence[Ontology] | None = None,
) -> UnitFactsState:
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

    extras = list(supplemental_ontologies or ())
    if is_fresh_facts_graph:
        logger.info("Generating fresh facts as Turtle")
        return await render_facts_fresh(state, tools, supplemental_ontologies=extras)
    else:
        logger.info("Generating facts update")
        return await render_facts_update(state, tools, supplemental_ontologies=extras)


def _prepare_prompt_data(
    state: UnitFactsState,
    access: UnitFactsOntologyAccess,
    profile,
) -> dict[str, str]:
    """Prepare common prompt data for both fresh and update rendering.

    Args:
        state: The current unit facts state
        access: Read-only ontology context (facts prompts use snapshot only).
        profile: Active graph format profile.

    Returns:
        Dictionary containing formatted prompt components
    """
    ctx = access.effective_ontology_for_prompt()
    if not isinstance(ctx.graph, RDFGraph):
        normalized_graph = RDFGraph()
        for triple in ctx.graph:
            normalized_graph.add(triple)
        for prefix, namespace_uri in ctx.graph.namespaces():
            normalized_graph.bind(prefix, namespace_uri)
        ctx.graph = normalized_graph
    domain_pairs = access.domain_prefix_pairs()
    ontology_index = build_ontology_index(ctx.graph)
    ontology_chapter = profile.format_ontology_chapter(ctx.graph, suffix=ontology_index)

    facts_instruction_str = profile.facts_operational_guidelines(
        facts_namespace=DEFAULT_IRI,
        domain_ontologies_clause=format_ontologies_clause(domain_pairs),
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
    state: UnitFactsState,
    tools: AtomicToolBox,
    supplemental_ontologies: Sequence[Ontology] | None = None,
) -> UnitFactsState:
    """Render fresh facts from the current chunk into Turtle format.

    Args:
        state: The current unit facts state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        UnitFactsState: Updated state with rendered facts.
    """
    logger.info("Rendering fresh facts")
    state.quarantined_literal_triples = []
    llm_tool = await tools.get_llm_tool(state.budget_tracker)
    profile = get_graph_format_profile(state.llm_graph_format)
    parser = PydanticOutputParser(pydantic_object=FactsRenderReport)

    access = ontology_access_for_unit_facts(state)

    known_prefixes = build_llm_prefix_map(
        access.ontology_for_prefixes(),
        supplemental_ontologies or (),
    )

    prompt_data = _prepare_prompt_data(state, access, profile)
    prompt_data_fresh = {
        "preamble": preamble,
        "improvement_instruction": "",
        "output_instruction": profile.render_fresh_output_instruction(target="facts"),
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
                "format_instructions": profile.format_instructions(FactsRenderReport),
                **prompt_data,
            },
            llm_graph_format=state.llm_graph_format,
        )
        state.set_external_evidence_request(
            WorkflowNode.TEXT_TO_FACTS, render_report.external_evidence_request
        )
        render_report.semantic_graph.sanitize_prefixes_namespaces()
        clean_graph, rejected = finalize_llm_graph(render_report.semantic_graph)
        state.content_unit.graph = clean_graph
        state.quarantined_literal_triples = rejected
        if rejected:
            logger.warning(
                "Fresh facts quarantined %d triple(s) with invalid typed literals",
                len(rejected),
            )

        # Track triples in budget tracker (fresh facts)
        num_triples = len(clean_graph)
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
    state: UnitFactsState,
    tools: AtomicToolBox,
    supplemental_ontologies: Sequence[Ontology] | None = None,
) -> UnitFactsState:
    """Render facts updates using SPARQL operations.

    Args:
        state: The current unit facts state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        UnitFactsState: Updated state with rendered facts.
    """
    logger.info("Rendering updates for facts")
    state.quarantined_literal_triples = []
    llm_tool = await tools.get_llm_tool(state.budget_tracker)
    profile = get_graph_format_profile(state.llm_graph_format)
    parser = PydanticOutputParser(pydantic_object=GraphUpdateRenderReport)

    access = ontology_access_for_unit_facts(state)
    prompt_data = _prepare_prompt_data(state, access, profile)
    prompt_data_update = {
        "preamble": preamble,
        "improvement_instruction": render_suggestions_prompt(
            state.suggestions, WorkflowNode.TEXT_TO_FACTS
        ),
        "output_instruction": profile.render_update_output_instruction(),
        "fact_chapter": profile.format_facts_chapter(state.content_unit.graph),
    }
    prompt_data.update(prompt_data_update)
    prompt = _create_prompt_template()
    known_prefixes = build_llm_prefix_map(
        access.ontology_for_prefixes(),
        supplemental_ontologies or (),
    )

    try:
        # Set known prefixes in context before parsing
        RDFGraph.set_known_prefixes(known_prefixes if known_prefixes else None)

        render_report: GraphUpdateRenderReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "format_instructions": profile.format_instructions(
                    GraphUpdateRenderReport
                ),
                **prompt_data,
            },
            llm_graph_format=state.llm_graph_format,
        )
        state.set_external_evidence_request(
            WorkflowNode.TEXT_TO_FACTS, render_report.external_evidence_request
        )
        graph_update = render_report.graph_update
        all_rejected = []
        for op in graph_update.triple_operations:
            clean_graph, rejected = finalize_llm_graph(op.graph)
            op.graph = clean_graph
            all_rejected.extend(rejected)
        state.quarantined_literal_triples = all_rejected
        if all_rejected:
            logger.warning(
                "Facts update quarantined %d triple(s) with invalid typed literals",
                len(all_rejected),
            )
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
