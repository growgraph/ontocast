"""Ontology criticism agent.

This module provides functionality for analyzing and validating ontologies.
"""

import logging
from collections import Counter

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from rdflib import URIRef

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import (
    FailureStage,
    OntologyAssemblyMode,
    Status,
    WorkflowNode,
)
from ontocast.onto.model import (
    LoopAttempt,
    OntologyCritiqueReport,
    Suggestions,
    format_findings_for_prompt,
)
from ontocast.onto.ontology_access import ontology_access_for_unit_ontology
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.prompt.common import (
    system_preamble_ontology as system_preamble,
)
from ontocast.prompt.common import text_template
from ontocast.prompt.criticise_ontology import (
    intro_instruction,
    ontology_criteria,
    partial_context_critique_notice,
    template_prompt,
)
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.prompt.ontology_context import build_ontology_index
from ontocast.prompt.web_grounding import persist_search_request, search_guidelines_for
from ontocast.tool import LLMTool
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import accept_reason, material_defects
from ontocast.tool.ontology_validation import count_fixes_targeting_snapshot

logger = logging.getLogger(__name__)


async def criticise_ontology(
    state: UnitOntologyState, tools: AtomicToolBox
) -> UnitOntologyState:
    """Critically analyze the ontology in the current content unit.

    Args:
        state: The current unit ontology state containing the ontology to analyze.
        tools: The toolbox instance providing utility functions.

    Returns:
        UnitOntologyState: Updated state with analysis results.
    """

    progress_info = state.get_content_unit_progress_string()
    logger.info(
        f"Ontology Critic for {progress_info}: visit {state.node_visits[WorkflowNode.CRITICISE_ONTOLOGY]}/{state.max_visits_per_node}"
    )

    if state.content_unit is None:
        state.status = Status.FAILED
        return state

    access = ontology_access_for_unit_ontology(state)
    if (
        access.has_non_empty_seed() is False
        and len(access.effective_graph_for_prompt()) == 0
    ):
        raise ValueError("Empty ontology context cannot be criticised")
    current_graph = access.effective_graph_for_prompt()

    profile = get_graph_format_profile(state.llm_graph_format)
    parser = PydanticOutputParser(pydantic_object=OntologyCritiqueReport)
    llm_tool: LLMTool = await tools.get_llm_tool(state.budget_tracker)

    # With the index appendix, as the renderer sends it: a critic shown bare
    # opaque IRIs cannot judge the term choices it is asked about. No memo here
    # -- effective_graph_for_prompt returns a bare graph, not a snapshot.
    # Ids are scoped to this unit's own delta. The chapter still shows the
    # retrieved catalog -- the critic cannot judge a term choice without it --
    # but those statements carry no id, so a delete that would propagate onto a
    # shared, versioned terminal is not expressible rather than merely flagged.
    indexed_ontology = profile.format_ontology_chapter_indexed(
        current_graph,
        scope=state.build_delta().inserts,
        suffix=build_ontology_index(current_graph),
        max_triples=state.ontology_context_max_triples,
    )
    state.prompt_triple_index = indexed_ontology.index
    ontology_chapter = indexed_ontology.text
    if state.deterministic_findings:
        # Machine-found delta defects, presented exactly the way the facts
        # critic receives its findings block.
        ontology_chapter += (
            "\n\n"
            + format_findings_for_prompt(
                state.deterministic_findings,
                advisory_heading="## Advisory findings (verify; fix when warranted)",
            )
            + "\nTreat every MANDATORY item as a required actionable fix.\n"
        )

    text_chapter = text_template.format(text=state.content_unit.extraction_text)

    user_instruction = state.ontology_user_instruction
    external_evidence = state.external_evidence_text

    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "intro_instruction",
            "ontology_criteria",
            "user_instruction",
            "ontology_chapter",
            "text_chapter",
            "external_evidence",
            "graph_format_instruction",
            "format_instructions",
        ],
    )

    graph_format_instruction = profile.critique_graph_instruction()
    web_search_enabled = tools.web_grounding_enabled_for_node(
        WorkflowNode.CRITICISE_ONTOLOGY
    )
    search_guidelines = search_guidelines_for(
        WorkflowNode.CRITICISE_ONTOLOGY, web_search_enabled
    )
    ontology_criteria_str = ontology_criteria
    if (
        state.ontology_snapshot.assembly_mode
        == OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE
    ):
        ontology_criteria_str = (
            f"{ontology_criteria_str}\n{partial_context_critique_notice}"
        )
    if search_guidelines:
        ontology_criteria_str = f"{ontology_criteria_str}\n{search_guidelines}"

    try:
        critique: OntologyCritiqueReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "preamble": system_preamble,
                "intro_instruction": intro_instruction,
                "ontology_criteria": ontology_criteria_str,
                "text_chapter": text_chapter,
                "user_instruction": user_instruction,
                "ontology_chapter": ontology_chapter,
                "external_evidence": external_evidence,
                "graph_format_instruction": graph_format_instruction,
                "format_instructions": profile.format_instructions(
                    OntologyCritiqueReport,
                    web_search_enabled=web_search_enabled,
                ),
            },
            llm_graph_format=state.llm_graph_format,
        )
        persist_search_request(
            state,
            WorkflowNode.CRITICISE_ONTOLOGY,
            critique.external_evidence_request,
            web_search_enabled,
        )
        logger.info(
            f"Parsed critique report - success: {critique.success}, "
            f"score: {critique.score}, n fixes: {len(critique.actionable_ontology_fixes)}."
        )

        # The gate is the deterministic findings, as on the facts side. The
        # incumbent rule was `success or score > 90` -- the top band of the
        # prompt's own rubric, which calls 70-89 "Good", so it rejected
        # ontologies its own instructions considered good. Its verdict is still
        # recorded, because replacing a gate deserves a distribution rather than
        # an argument, and the ontology critic has never run on recorded data.
        #
        # The blocking set is the destructive-or-lossy subset only. Blocking on
        # every mandatory finding would include `missing_label`, which fires
        # whenever a render mints an unlabelled term: routine, and a permanent
        # per-unit tax rather than a defect signal.
        incumbent_accepted = critique.success or critique.score > 90
        defects = material_defects(
            state.deterministic_findings,
            critique.actionable_ontology_fixes,
            tools.ontology_acceptance_policy,
        )
        accepted = not defects
        delta = state.build_delta()
        state.attempt_log.append(
            LoopAttempt(
                render_attempt=state.node_visits[WorkflowNode.TEXT_TO_ONTOLOGY],
                critic_attempt=state.node_visits[WorkflowNode.CRITICISE_ONTOLOGY],
                kind="critic",
                score=critique.score,
                success=accepted,
                accept_reason=accept_reason(defects),
                incumbent_accepted=incumbent_accepted,
                n_actionable_fixes=len(critique.actionable_ontology_fixes),
                severity_counts=Counter(
                    fix.severity for fix in critique.actionable_ontology_fixes
                ),
                n_deterministic_findings=len(state.deterministic_findings),
                n_mandatory_findings=sum(
                    1 for finding in state.deterministic_findings if finding.mandatory
                ),
                triple_count=len(state.working_graph),
                delta_triple_count=len(delta.inserts),
                n_fixes_targeting_snapshot=count_fixes_targeting_snapshot(
                    critique.actionable_ontology_fixes,
                    None
                    if state.ontology_snapshot.is_empty()
                    else state.ontology_snapshot.graph,
                    {
                        str(subject)
                        for subject in delta.inserts.subjects()
                        if isinstance(subject, URIRef)
                    },
                ),
            )
        )

        if accepted:
            state.status = Status.SUCCESS
            state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.SUCCESS)
            # Kept, not cleared. Acceptance decides whether the unit may leave
            # the loop; it does not decide whether the critique is worth
            # applying. Clearing here discarded every fix attached to an
            # accepted render -- which, since a REMOVE can never by itself
            # cause a rejection, was most of them.
            state.suggestions = Suggestions.from_critique_report(critique)
            logger.info("Ontology critique passed")
        else:
            state.status = Status.FAILED
            state.failure_stage = FailureStage.ONTOLOGY_CRITIQUE
            state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.FAILED)
            state.suggestions = Suggestions.from_critique_report(critique)
            state.failure_reason = (
                f"Ontology unit has {len(defects)} material defect(s)"
            )
            logger.info(
                f"Ontology critique failed: {critique.systemic_critique_summary}"
            )
        return state

    except Exception as e:
        logger.error(f"Failed to critique ontology: {str(e)}")
        state.set_failure(FailureStage.ONTOLOGY_CRITIQUE, str(e))
        state.set_node_status(WorkflowNode.CRITICISE_ONTOLOGY, Status.FAILED)
        return state
