"""Fact criticism agent.

This module provides functionality for analyzing and validating extracted facts.
"""

import logging
from collections import Counter

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import (
    FactsCritiqueReport,
    LoopAttempt,
    Suggestions,
    format_findings_for_prompt,
)
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
from ontocast.prompt.web_grounding import persist_search_request, search_guidelines_for
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import (
    accept_reason,
    material_defects,
)

logger = logging.getLogger(__name__)


def _build_quarantine_chapter(state: UnitFactsState) -> str:
    sections: list[str] = []
    if state.quarantined_literal_triples:
        formatted = format_quarantine_for_prompt(
            state.quarantined_literal_triples,
            state.llm_graph_format,
        )
        sections.append(
            "\n\n## Quarantined triples (invalid XSD typed literals, excluded from applied graph)\n"
            "The following triples were not merged into the facts graph. Replace them using "
            "structured representations defined in the ontology chapter above.\n\n"
            f"{formatted}\n"
        )
    if state.deterministic_findings:
        sections.append(
            "\n\n"
            + format_findings_for_prompt(state.deterministic_findings)
            + "\nTreat every MANDATORY item as a required actionable fix.\n"
        )
    return "".join(sections)


async def criticise_facts(
    state: UnitFactsState, tools: AtomicToolBox
) -> UnitFactsState:
    """Critically analyze facts in the current content unit.

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
    # Same chapter the renderer gets, index appendix included. Building it
    # without the suffix left the critic reading opaque IRIs while guideline 6a
    # told the renderer to resolve them through the TERM INDEX -- so the critic
    # judged term choices it could not read. Also memoised on the shared
    # snapshot, so this stops re-serialising the ontology on every visit.
    ontology_chapter = ctx.prompt_chapter(
        profile, max_triples=state.ontology_context_max_triples
    )
    # Every statement gets a citable id, and the index is kept on the state so
    # the fixes that come back can be resolved by lookup. The critic used to be
    # asked to requote the statements it wanted changed, which it reproduces
    # correctly only a minority of the time -- for a bare removal, almost never.
    indexed_facts = profile.format_facts_chapter_indexed(state.content_unit.graph)
    state.prompt_triple_index = indexed_facts.index
    facts_chapter = indexed_facts.text + _build_quarantine_chapter(state)

    text_chapter = text_template.format(text=state.content_unit.extraction_text)

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
            "conformance_chapter",
            "facts_chapter",
            "text_chapter",
            "graph_format_instruction",
            "format_instructions",
        ],
    )

    graph_format_instruction = profile.critique_graph_instruction()
    web_search_enabled = tools.web_grounding_enabled_for_node(
        WorkflowNode.CRITICISE_FACTS
    )
    search_guidelines = search_guidelines_for(
        WorkflowNode.CRITICISE_FACTS, web_search_enabled
    )
    evaluation_instruction_str = evaluation_instruction
    if search_guidelines:
        evaluation_instruction_str = f"{evaluation_instruction}\n\n{search_guidelines}"

    prompt_data = {
        "preamble": preamble,
        "evaluation_instruction": evaluation_instruction_str,
        "user_instruction": user_instruction,
        "ontology_chapter": ontology_chapter,
        # Same rulebook the gate validates against; critique and render
        # share one contract.
        "conformance_chapter": state.conformance_chapter,
        "facts_chapter": facts_chapter,
        "text_chapter": text_chapter,
        "graph_format_instruction": graph_format_instruction,
        "format_instructions": profile.format_instructions(
            FactsCritiqueReport,
            web_search_enabled=web_search_enabled,
        ),
    }

    try:
        critique: FactsCritiqueReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs=prompt_data,
            llm_graph_format=state.llm_graph_format,
        )
        persist_search_request(
            state,
            WorkflowNode.CRITICISE_FACTS,
            critique.external_evidence_request,
            web_search_enabled,
        )

        logger.debug(
            f"Parsed critique report - success: {critique.success}, "
            f"score: {critique.score}"
        )

        # Acceptance is decided from defects that can be pointed at: the
        # deterministic findings already collected against this graph, plus the
        # critic's own fixes at the configured severity. `score` and `success`
        # are recorded and no longer consulted -- see acceptance.py for what the
        # score gate measured and why it could not be calibrated.
        defects = material_defects(
            state.deterministic_findings,
            critique.actionable_triple_fixes,
            tools.acceptance_policy,
        )
        reason = accept_reason(defects)

        state.attempt_log.append(
            LoopAttempt(
                render_attempt=state.node_visits[WorkflowNode.TEXT_TO_FACTS],
                critic_attempt=state.node_visits[WorkflowNode.CRITICISE_FACTS],
                kind="critic",
                score=critique.score,
                success=not defects,
                accept_reason=reason,
                n_actionable_fixes=len(critique.actionable_triple_fixes),
                severity_counts=Counter(
                    fix.severity for fix in critique.actionable_triple_fixes
                ),
                action_severity_counts=Counter(
                    f"{fix.action}:{fix.severity}"
                    for fix in critique.actionable_triple_fixes
                ),
                n_deterministic_findings=len(state.deterministic_findings),
                n_mandatory_findings=sum(
                    1 for finding in state.deterministic_findings if finding.mandatory
                ),
                triple_count=len(state.content_unit.graph),
            )
        )

        if not defects:
            state.status = Status.SUCCESS
            state.set_node_status(WorkflowNode.CRITICISE_FACTS, Status.SUCCESS)
            # Accepting means "no defect worth another render", NOT "the
            # critique was empty". The fixes are kept: the repair lane compiles
            # the mechanical ones for free and records the rest as residual.
            # Clearing them here used to discard the entire critique of every
            # accepted render -- the bulk of everything the critic produced,
            # since a REMOVE fix can never make a render blocking.
            state.suggestions = Suggestions.from_critique_report(critique)
            logger.info(
                "Facts critique passed (score %s, no material defect)",
                critique.score,
            )
        else:
            state.status = Status.FAILED
            state.set_node_status(WorkflowNode.CRITICISE_FACTS, Status.FAILED)
            state.failure_stage = FailureStage.FACTS_CRITIQUE
            state.suggestions = Suggestions.from_critique_report(critique)
            state.failure_reason = f"Facts unit has {len(defects)} material defect(s)"
            logger.info(
                "Facts critique rejected on %s: %s (score %s)",
                reason,
                "; ".join(defect.message for defect in defects[:3]),
                critique.score,
            )

        return state

    except Exception as e:
        logger.error(f"Failed to criticize facts: {str(e)}")
        state.set_failure(FailureStage.FACTS_CRITIQUE, str(e))
        state.set_node_status(WorkflowNode.CRITICISE_FACTS, Status.FAILED)
        return state
