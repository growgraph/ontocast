"""Ontology triple rendering agent for OntoCast.

This module provides functionality for rendering RDF triples from ontologies into
human-readable formats, making the ontological knowledge more accessible and
understandable.
The agent decides between generating bare Turtle for fresh ontologies and structured graph updates for patches.

"""

import logging
from collections.abc import Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry, render_suggestions_prompt
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import GraphUpdateRenderReport, OntologyRenderReport
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import (
    UnitOntologyAccess,
    build_llm_prefix_map,
    ontology_access_for_unit_ontology,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.prompt.common import system_preamble_ontology as system_preamble
from ontocast.prompt.common import text_template
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.prompt.ontology_context import format_ontologies_clause
from ontocast.prompt.render_ontology import (
    general_ontology_instruction,
    intro_instruction_fresh,
    intro_instruction_update,
    template_prompt,
)
from ontocast.prompt.web_grounding import persist_search_request, search_guidelines_for
from ontocast.tool.atomic import AtomicToolBox

logger = logging.getLogger(__name__)


def _create_ontology_render_prompt_template() -> PromptTemplate:
    return PromptTemplate(
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
            "external_evidence",
            "format_instructions",
        ],
    )


def _handle_ontology_render_error(
    state: UnitOntologyState, error: Exception, stage: FailureStage
) -> UnitOntologyState:
    logger.error("Failed to generate ontology output: %s", error)
    state.set_node_status(WorkflowNode.TEXT_TO_ONTOLOGY, Status.FAILED)
    state.set_failure(stage, str(error))
    return state


def _prepare_ontology_common_prompt_layers(
    state: UnitOntologyState,
    access: UnitOntologyAccess,
    *,
    search_guidelines: str,
) -> tuple[str, str, str]:
    domain_pairs = access.domain_prefix_pairs()
    general_ontology_instruction_str = general_ontology_instruction.format(
        domain_ontologies_clause=format_ontologies_clause(domain_pairs),
        search_guidelines=search_guidelines,
    )
    text_chapter = text_template.format(text=state.content_unit.extraction_text)
    external_evidence = state.external_evidence_text
    if external_evidence:
        state.mark_external_evidence_used(WorkflowNode.TEXT_TO_ONTOLOGY)
    return general_ontology_instruction_str, text_chapter, external_evidence


async def render_ontology(
    state: UnitOntologyState,
    tools: AtomicToolBox,
    supplemental_ontologies: Sequence[Ontology] | None = None,
) -> UnitOntologyState:
    """Structured hybrid ontology renderer: fresh Turtle or structured graph updates.

    This function decides between generating bare Turtle for fresh ontologies
    and structured TripleOp graph patches for updates based on whether the ontology exists.

    Args:
        state: The current unit ontology state
        tools: The toolbox containing necessary tools

    Returns:
        UnitOntologyState: Updated state with rendered ontology
    """

    progress_info = state.get_content_unit_progress_string()
    logger.info(
        f"Ontology Renderer for {progress_info}: visit {state.node_visits[WorkflowNode.TEXT_TO_ONTOLOGY]}/{state.max_visits_per_node}"
    )
    access = ontology_access_for_unit_ontology(state)
    current = access.effective_ontology_for_prompt()
    # Guardrail for map/reduce flow: if a non-null snapshot exists, stay in update mode.
    has_seed_ontology = access.has_non_null_seed_snapshot()
    has_no_seed_ontology = current.is_null() and not has_seed_ontology

    extras = list(supplemental_ontologies or ())
    if has_no_seed_ontology:
        return await render_ontology_fresh(state, tools, supplemental_ontologies=extras)
    else:
        return await render_ontology_update(
            state, tools, supplemental_ontologies=extras
        )


async def render_ontology_fresh(
    state: UnitOntologyState,
    tools: AtomicToolBox,
    supplemental_ontologies: Sequence[Ontology] | None = None,
) -> UnitOntologyState:
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

    profile = get_graph_format_profile(state.llm_graph_format)
    parser = PydanticOutputParser(pydantic_object=OntologyRenderReport)
    logger.info("Rendering fresh ontology")
    intro_instruction = intro_instruction_fresh.format(
        current_domain=state.current_domain
    )
    output_instruction = profile.render_fresh_output_instruction(target="ontology")
    ontology_ttl = ""
    improvement_instruction_str = ""
    access = ontology_access_for_unit_ontology(state)
    web_search_enabled = tools.web_grounding_enabled_for_node(
        WorkflowNode.TEXT_TO_ONTOLOGY
    )
    (
        general_ontology_instruction_str,
        text_chapter,
        external_evidence,
    ) = _prepare_ontology_common_prompt_layers(
        state,
        access,
        search_guidelines=search_guidelines_for(
            WorkflowNode.TEXT_TO_ONTOLOGY, web_search_enabled
        ),
    )

    prompt = _create_ontology_render_prompt_template()
    known_prefixes = build_llm_prefix_map(
        access.ontology_for_prefixes(),
        supplemental_ontologies or (),
    )

    try:
        RDFGraph.set_known_prefixes(known_prefixes if known_prefixes else None)
        llm_tool = await tools.get_llm_tool(state.budget_tracker)
        render_report: OntologyRenderReport = await call_llm_with_retry(
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
                "external_evidence": external_evidence,
                "format_instructions": profile.format_instructions(
                    OntologyRenderReport,
                    web_search_enabled=web_search_enabled,
                ),
            },
            llm_graph_format=state.llm_graph_format,
        )
        persist_search_request(
            state,
            WorkflowNode.TEXT_TO_ONTOLOGY,
            render_report.external_evidence_request,
            web_search_enabled,
        )
        state.current_ontology = render_report.ontology
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
        return _handle_ontology_render_error(
            state, e, FailureStage.GENERATE_TTL_FOR_ONTOLOGY
        )
    finally:
        RDFGraph.set_known_prefixes(None)


async def render_ontology_update(
    state: UnitOntologyState,
    tools: AtomicToolBox,
    supplemental_ontologies: Sequence[Ontology] | None = None,
) -> UnitOntologyState:
    """Render ontology triples into a human-readable format.

    This function takes the triples from the current ontology and renders them
    into a more accessible format, making the ontological knowledge easier to
    understand.

    Args:
        state: The current unit ontology state containing the ontology to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        UnitOntologyState: Updated state with rendered triples.
    """

    profile = get_graph_format_profile(state.llm_graph_format)
    parser = PydanticOutputParser(pydantic_object=GraphUpdateRenderReport)
    access = ontology_access_for_unit_ontology(state)
    current = access.effective_ontology_for_prompt()
    ontology_iri = current.iri
    ontology_desc = current.describe()
    multi_source_note = ""
    if state.ontology_patch_sources:
        joined_sources = ", ".join(state.ontology_patch_sources[:10])
        multi_source_note = (
            "\nThe provided ontology context may combine patches from multiple source "
            f"ontologies: {joined_sources}. Preserve existing IRIs, namespace boundaries, "
            "and avoid collapsing distinct source namespaces unless explicitly justified."
        )
    intro_instruction = intro_instruction_update.format(
        ontology_iri=ontology_iri,
        ontology_desc=ontology_desc,
        multi_source_note=multi_source_note,
    )
    ontology_chapter = profile.format_ontology_chapter(current.graph)
    output_instruction = profile.render_update_output_instruction()
    improvement_instruction_str = render_suggestions_prompt(
        state.suggestions, WorkflowNode.TEXT_TO_ONTOLOGY
    )

    web_search_enabled = tools.web_grounding_enabled_for_node(
        WorkflowNode.TEXT_TO_ONTOLOGY
    )
    (
        general_ontology_instruction_str,
        text_chapter,
        external_evidence,
    ) = _prepare_ontology_common_prompt_layers(
        state,
        access,
        search_guidelines=search_guidelines_for(
            WorkflowNode.TEXT_TO_ONTOLOGY, web_search_enabled
        ),
    )

    prompt = _create_ontology_render_prompt_template()
    known_prefixes = build_llm_prefix_map(
        access.ontology_for_prefixes(),
        supplemental_ontologies or (),
    )

    try:
        llm_tool = await tools.get_llm_tool(state.budget_tracker)
        RDFGraph.set_known_prefixes(known_prefixes if known_prefixes else None)

        render_report: GraphUpdateRenderReport = await call_llm_with_retry(
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
                "external_evidence": external_evidence,
                "format_instructions": profile.format_instructions(
                    GraphUpdateRenderReport,
                    web_search_enabled=web_search_enabled,
                ),
            },
            llm_graph_format=state.llm_graph_format,
        )
        persist_search_request(
            state,
            WorkflowNode.TEXT_TO_ONTOLOGY,
            render_report.external_evidence_request,
            web_search_enabled,
        )
        graph_update = render_report.graph_update
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
        return _handle_ontology_render_error(
            state, e, FailureStage.GENERATE_GRAPH_UPDATE_FOR_ONTOLOGY
        )
    finally:
        # Clear the context after parsing
        RDFGraph.set_known_prefixes(None)
