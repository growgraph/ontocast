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
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.common import (
    facts_template,
    ontology_template,
    text_template,
    user_template,
)
from ontocast.prompt.criticise_facts import (
    evaluation_instruction,
    preamble,
    template_prompt,
)
from ontocast.tool.atomic import AtomicToolBox

logger = logging.getLogger(__name__)


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
    parser = PydanticOutputParser(pydantic_object=FactsCritiqueReport)

    ontology_ttl = state.ontology_snapshot.graph.serialize(format="turtle")

    ontology_chapter = ontology_template.format(
        ontology_ttl=ontology_ttl,
    )

    facts_ttl = state.content_unit.graph.serialize(format="turtle")

    facts_chapter = facts_template.format(
        facts_ttl=facts_ttl,
    )

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
            "format_instructions",
        ],
    )

    prompt_data = {
        "preamble": preamble,
        "evaluation_instruction": evaluation_instruction,
        "user_instruction": user_instruction,
        "ontology_chapter": ontology_chapter,
        "facts_chapter": facts_chapter,
        "text_chapter": text_chapter,
        "format_instructions": parser.get_format_instructions(),
    }

    try:
        critique: FactsCritiqueReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs=prompt_data,
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
