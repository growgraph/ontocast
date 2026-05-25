"""Enhanced fact criticism agent with memory and SPARQL operations.

This module provides enhanced functionality for analyzing and validating facts
with SPARQL operation support.
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import FactsCritiqueReport, Suggestions
from ontocast.onto.ontology_access import ontology_access_for_unit_facts
from ontocast.onto.rdfgraph import format_quarantine_for_prompt
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.common import text_template, user_template
from ontocast.prompt.criticise_facts import (
    evaluation_instruction,
    preamble,
    template_prompt,
)
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.tool.atomic import AtomicToolBox

logger = logging.getLogger(__name__)


def _build_quarantine_chapter(state: UnitFactsState) -> str:
    if not state.quarantined_literal_triples:
        return ""

    formatted = format_quarantine_for_prompt(
        state.quarantined_literal_triples,
        state.llm_graph_format,
    )
    return (
        "\n\n## Quarantined triples (invalid XSD typed literals, excluded from applied graph)\n"
        "The following triples were not merged into the facts graph. Replace them using "
        "structured representations defined in the ontology chapter above.\n\n"
        f"{formatted}\n"
    )


async def criticise_facts(
    state: UnitFactsState, tools: AtomicToolBox
) -> UnitFactsState:
    """Enhanced criticize facts with SPARQL operations.

    This function performs a critical analysis of the facts in the current content unit,
    with SPARQL operation support.

    Args:
        state: The current unit facts state containing the chunk to analyze.
        tools: The toolbox instance providing utility functions.

    Returns:
        UnitFactsState: Updated state with analysis results.
    """
    if not state.content_unit:
        logger.warning("No current content unit to analyze")
        return state

    progress_info = state.get_content_unit_progress_string()
    logger.info(
        f"Facts critic for {progress_info}: visit {state.node_visits[WorkflowNode.CRITICISE_FACTS]}/{state.max_visits_per_node}"
    )

    llm_tool = await tools.get_llm_tool(state.budget_tracker)
    profile = get_graph_format_profile(state.llm_graph_format)
    parser = PydanticOutputParser(pydantic_object=FactsCritiqueReport)

    ctx = ontology_access_for_unit_facts(state).effective_ontology_for_prompt()
    ontology_chapter = profile.format_ontology_chapter(ctx.graph)
    facts_chapter = profile.format_facts_chapter(
        state.content_unit.graph
    ) + _build_quarantine_chapter(state)

    text_chapter = text_template.format(text=state.content_unit.text)

    user_instruction = (
        user_template.format(user_instruction=state.facts_user_instruction)
        if state.facts_user_instruction
        else ""
    )

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "evaluation_instruction",
            "user_instruction",
            "ontology_chapter",
            "facts_chapter",
            "text_chapter",
            "graph_format_instruction",
            "format_instructions",
        ],
    )

    graph_format_instruction = profile.critique_graph_instruction()

    prompt_data = {
        "preamble": preamble,
        "evaluation_instruction": evaluation_instruction,
        "user_instruction": user_instruction,
        "ontology_chapter": ontology_chapter,
        "facts_chapter": facts_chapter,
        "text_chapter": text_chapter,
        "graph_format_instruction": graph_format_instruction,
        "format_instructions": profile.format_instructions(FactsCritiqueReport),
    }

    try:
        critique: FactsCritiqueReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs=prompt_data,
            llm_graph_format=state.llm_graph_format,
        )
        state.set_external_evidence_request(
            WorkflowNode.CRITICISE_FACTS, critique.external_evidence_request
        )

        logger.debug(
            f"Parsed critique report - success: {critique.success}, "
            f"score: {critique.score}"
        )

        if critique.success or critique.score > 90:
            state.status = Status.SUCCESS
            state.set_node_status(WorkflowNode.CRITICISE_FACTS, Status.SUCCESS)
            logger.info("Facts critique passed")
        else:
            state.status = Status.FAILED
            state.set_node_status(WorkflowNode.CRITICISE_FACTS, Status.FAILED)
            state.failure_stage = FailureStage.FACTS_CRITIQUE
            state.suggestions = Suggestions.from_critique_report(critique)
            state.failure_reason = "Facts Critic suggests improvements"

        return state

    except Exception as e:
        logger.error(f"Failed to criticize facts: {str(e)}")
        state.set_failure(FailureStage.FACTS_CRITIQUE, str(e))
        state.set_node_status(WorkflowNode.CRITICISE_FACTS, Status.FAILED)
        return state
