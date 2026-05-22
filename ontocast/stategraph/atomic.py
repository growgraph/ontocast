"""Reusable per-unit render/critic retry loops.

These loops are designed for map/reduce execution where each content unit
is processed independently. They deep-copy the incoming unit state, then run
render -> critic until success or retry exhaustion. After the last allowed
render succeeds, the critic is skipped: no further extract exists for feedback
to inform.

Ontology context assembly (``resolve_unit_ontology_context``) runs at the
start of both ``ontology_loop`` and ``facts_loop`` so each unit chooses its
own ontology context according to mode/policy.
"""

import logging
from copy import deepcopy

from ontocast.agent.criticise_facts import criticise_facts
from ontocast.agent.criticise_ontology import criticise_ontology
from ontocast.agent.external_evidence import (
    fetch_external_evidence_for_node,
    plan_external_evidence_for_node,
)
from ontocast.agent.render_facts import render_facts
from ontocast.agent.render_ontology import render_ontology
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import ExternalEvidenceCacheEntry, ExternalEvidenceRequest
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.rdfgraph import format_quarantine_for_prompt
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.context_resolver import (
    UnitOntologyContext,
    resolve_effective_facts_ontology_context,
    resolve_unit_ontology_context,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def _document_supplemental_ontologies(document_state: AgentState) -> list[Ontology]:
    """Non-null reduced ontology artifacts for LLM ingest prefix repair."""
    return [
        ontology
        for ontology in document_ontology_access(document_state).reduced_artifacts()
        if not ontology.is_null()
    ]


def _resolve_max_visits_limit(state_visits: int, override: int | None) -> int:
    """Return a safe visit limit while respecting explicit overrides."""
    visits = state_visits if override is None else override
    return max(1, visits)


def _skip_critic_after_final_render(render_attempt: int, max_visits: int) -> bool:
    """True when this render attempt is the last allowed; critic cannot drive a retry."""
    return render_attempt == max_visits


def _surface_unresolved_quarantine(unit_state: UnitFactsState) -> None:
    """Log and record quarantined literals when the critic is skipped on the final render."""
    if not unit_state.quarantined_literal_triples:
        return

    logger.warning(
        "%d quarantined literal triple(s) were not critiqued (final render)",
        len(unit_state.quarantined_literal_triples),
    )
    formatted = format_quarantine_for_prompt(
        unit_state.quarantined_literal_triples,
        unit_state.llm_graph_format,
    )
    notice = (
        "Unresolved quarantined typed literals (invalid XSD lexical forms, not applied):\n"
        f"{formatted}"
    )
    existing = unit_state.suggestions.systemic_critique_summary.strip()
    if existing:
        unit_state.suggestions.systemic_critique_summary = f"{existing}\n\n{notice}"
    else:
        unit_state.suggestions.systemic_critique_summary = notice


def _reset_node_evidence_context(
    state: UnitFactsState | UnitOntologyState, node: WorkflowNode
) -> None:
    """Start node execution in no-search mode with empty evidence context."""
    state.set_external_evidence_request(node, ExternalEvidenceRequest())
    state.set_external_evidence_cache_entry(node, ExternalEvidenceCacheEntry())
    state.load_external_evidence_for_node(node)


async def _apply_facts_ontology_context(
    unit_state: UnitFactsState,
    document_state: AgentState,
    tools: ToolBox,
) -> UnitFactsState:
    """Set ontology_snapshot for facts from per-unit context resolver."""
    ctx = await resolve_effective_facts_ontology_context(
        document_state, tools, unit_state.content_unit
    )
    logger.info(
        f"Ontology selected for mode {document_state.ontology_context_mode}: {ctx.ontology_snapshot.iri}",
    )
    unit_state.ontology_snapshot = deepcopy(ctx.ontology_snapshot)
    unit_state.ontology_patch_sources = list(ctx.patch_sources)
    unit_state.assembly_anchor_iri = ctx.anchor_iri
    unit_state.assembly_mode_used = ctx.assembly_mode
    return unit_state


async def facts_loop(
    state: UnitFactsState,
    tools: ToolBox,
    document_state: AgentState,
    max_visits_per_node: int | None = None,
    pre_resolved_ontology: Ontology | None = None,
    pre_resolved_context: UnitOntologyContext | None = None,
) -> UnitFactsState:
    """Run facts render/critic loop for one content unit.

    Ontology context is resolved per unit before rendering unless
    ``pre_resolved_ontology`` is provided, in which case it is used directly
    and the store-based context resolution is skipped. This is intended for
    sequential unit-level pipelines where the ontology loop has already run
    and its output should feed directly into fact extraction.
    """
    atomic = tools.get_atomic_tools()
    unit_state = state.model_copy(deep=True)
    try:
        if pre_resolved_context is not None and pre_resolved_ontology is not None:
            raise ValueError(
                "Provide either pre_resolved_context or pre_resolved_ontology, not both."
            )
        if pre_resolved_context is not None:
            unit_state.ontology_snapshot = deepcopy(
                pre_resolved_context.ontology_snapshot
            )
            unit_state.ontology_patch_sources = list(pre_resolved_context.patch_sources)
            unit_state.assembly_anchor_iri = pre_resolved_context.anchor_iri
            unit_state.assembly_mode_used = pre_resolved_context.assembly_mode
        elif pre_resolved_ontology is not None:
            unit_state.ontology_snapshot = deepcopy(pre_resolved_ontology)
        else:
            unit_state = await _apply_facts_ontology_context(
                unit_state, document_state, tools
            )
        max_visits = _resolve_max_visits_limit(
            unit_state.max_visits_per_node, max_visits_per_node
        )
        unit_state.max_visits_per_node = max_visits

        for render_attempt in range(1, max_visits + 1):
            unit_state.node_visits[WorkflowNode.TEXT_TO_FACTS] += 1
            _reset_node_evidence_context(unit_state, WorkflowNode.TEXT_TO_FACTS)
            supplemental = _document_supplemental_ontologies(document_state)
            unit_state = await render_facts(
                unit_state, atomic, supplemental_ontologies=supplemental
            )
            if unit_state.status != Status.SUCCESS:
                render_request = unit_state.get_external_evidence_request(
                    WorkflowNode.TEXT_TO_FACTS
                )
                if render_request.initiate_search:
                    unit_state = await plan_external_evidence_for_node(
                        unit_state, atomic, WorkflowNode.TEXT_TO_FACTS
                    )
                    unit_state = await fetch_external_evidence_for_node(
                        unit_state, atomic, WorkflowNode.TEXT_TO_FACTS
                    )
                    unit_state = await render_facts(
                        unit_state, atomic, supplemental_ontologies=supplemental
                    )
                    if unit_state.status == Status.SUCCESS:
                        logger.info(
                            "Unit facts render recovered with search at attempt %s/%s",
                            render_attempt,
                            max_visits,
                        )
                    else:
                        logger.info(
                            "Unit facts render failed at attempt %s/%s (with search)",
                            render_attempt,
                            max_visits,
                        )
                        continue
                else:
                    logger.info(
                        "Unit facts render failed at attempt %s/%s (no search request)",
                        render_attempt,
                        max_visits,
                    )
                    continue

            if _skip_critic_after_final_render(render_attempt, max_visits):
                logger.info(
                    "Unit facts loop finishing on final render attempt %s/%s "
                    "(no further extract; skipping critic)",
                    render_attempt,
                    max_visits,
                )
                _surface_unresolved_quarantine(unit_state)
                return unit_state

            for critic_attempt in range(1, max_visits + 1):
                unit_state.node_visits[WorkflowNode.CRITICISE_FACTS] += 1
                _reset_node_evidence_context(unit_state, WorkflowNode.CRITICISE_FACTS)
                unit_state = await criticise_facts(unit_state, atomic)
                if unit_state.status == Status.SUCCESS:
                    logger.info(
                        "Unit facts loop converged at render %s/%s critic %s/%s",
                        render_attempt,
                        max_visits,
                        critic_attempt,
                        max_visits,
                    )
                    return unit_state

                critic_request = unit_state.get_external_evidence_request(
                    WorkflowNode.CRITICISE_FACTS
                )
                if not critic_request.initiate_search:
                    logger.info(
                        "Unit facts critic failed at render %s/%s critic %s/%s "
                        "without search request",
                        render_attempt,
                        max_visits,
                        critic_attempt,
                        max_visits,
                    )
                    break

                unit_state = await plan_external_evidence_for_node(
                    unit_state, atomic, WorkflowNode.CRITICISE_FACTS
                )
                unit_state = await fetch_external_evidence_for_node(
                    unit_state, atomic, WorkflowNode.CRITICISE_FACTS
                )
                unit_state = await criticise_facts(unit_state, atomic)
                if unit_state.status == Status.SUCCESS:
                    logger.info(
                        "Unit facts loop converged with critic search at "
                        "render %s/%s critic %s/%s",
                        render_attempt,
                        max_visits,
                        critic_attempt,
                        max_visits,
                    )
                    return unit_state

        logger.info("Unit facts loop exhausted retries")
        return unit_state
    except Exception as exc:
        logger.exception("Unhandled exception in facts_loop")
        unit_state.set_failure(FailureStage.FACTS_CRITIQUE, str(exc))
        return unit_state


async def ontology_loop(
    state: UnitOntologyState,
    tools: ToolBox,
    document_state: AgentState,
    max_visits_per_node: int | None = None,
) -> UnitOntologyState:
    """Run ontology render/critic loop for one content unit.

    Per-unit ontology context is assembled via ``resolve_unit_ontology_context``
    before the first render.
    """
    atomic = tools.get_atomic_tools()
    unit_state = state.model_copy(deep=True)
    try:
        ctx = await resolve_unit_ontology_context(
            document_state, tools, unit_state.content_unit
        )
        unit_state.ontology_snapshot = deepcopy(ctx.ontology_snapshot)
        unit_state.ontology_patch_sources = list(ctx.patch_sources)
        unit_state.current_ontology = deepcopy(unit_state.ontology_snapshot)
        unit_state.assembly_anchor_iri = ctx.anchor_iri
        unit_state.assembly_mode_used = ctx.assembly_mode

        max_visits = _resolve_max_visits_limit(
            unit_state.max_visits_per_node, max_visits_per_node
        )
        unit_state.max_visits_per_node = max_visits

        for render_attempt in range(1, max_visits + 1):
            unit_state.node_visits[WorkflowNode.TEXT_TO_ONTOLOGY] += 1
            _reset_node_evidence_context(unit_state, WorkflowNode.TEXT_TO_ONTOLOGY)
            supplemental = _document_supplemental_ontologies(document_state)
            unit_state = await render_ontology(
                unit_state, atomic, supplemental_ontologies=supplemental
            )
            if unit_state.status != Status.SUCCESS:
                render_request = unit_state.get_external_evidence_request(
                    WorkflowNode.TEXT_TO_ONTOLOGY
                )
                if render_request.initiate_search:
                    unit_state = await plan_external_evidence_for_node(
                        unit_state, atomic, WorkflowNode.TEXT_TO_ONTOLOGY
                    )
                    unit_state = await fetch_external_evidence_for_node(
                        unit_state, atomic, WorkflowNode.TEXT_TO_ONTOLOGY
                    )
                    unit_state = await render_ontology(
                        unit_state, atomic, supplemental_ontologies=supplemental
                    )
                    if unit_state.status == Status.SUCCESS:
                        logger.info(
                            "Unit ontology render recovered with search at attempt %s/%s",
                            render_attempt,
                            max_visits,
                        )
                    else:
                        logger.info(
                            "Unit ontology render failed at attempt %s/%s (with search)",
                            render_attempt,
                            max_visits,
                        )
                        continue
                else:
                    logger.info(
                        "Unit ontology render failed at attempt %s/%s (no search request)",
                        render_attempt,
                        max_visits,
                    )
                    continue

            if _skip_critic_after_final_render(render_attempt, max_visits):
                logger.info(
                    "Unit ontology loop finishing on final render attempt %s/%s "
                    "(no further extract; skipping critic)",
                    render_attempt,
                    max_visits,
                )
                return unit_state

            for critic_attempt in range(1, max_visits + 1):
                unit_state.node_visits[WorkflowNode.CRITICISE_ONTOLOGY] += 1
                _reset_node_evidence_context(
                    unit_state, WorkflowNode.CRITICISE_ONTOLOGY
                )
                unit_state = await criticise_ontology(unit_state, atomic)
                if unit_state.status == Status.SUCCESS:
                    logger.info(
                        "Unit ontology loop converged at render %s/%s critic %s/%s",
                        render_attempt,
                        max_visits,
                        critic_attempt,
                        max_visits,
                    )
                    return unit_state

                critic_request = unit_state.get_external_evidence_request(
                    WorkflowNode.CRITICISE_ONTOLOGY
                )
                if not critic_request.initiate_search:
                    logger.info(
                        "Unit ontology critic failed at render %s/%s critic %s/%s "
                        "without search request",
                        render_attempt,
                        max_visits,
                        critic_attempt,
                        max_visits,
                    )
                    break

                unit_state = await plan_external_evidence_for_node(
                    unit_state, atomic, WorkflowNode.CRITICISE_ONTOLOGY
                )
                unit_state = await fetch_external_evidence_for_node(
                    unit_state, atomic, WorkflowNode.CRITICISE_ONTOLOGY
                )
                unit_state = await criticise_ontology(unit_state, atomic)
                if unit_state.status == Status.SUCCESS:
                    logger.info(
                        "Unit ontology loop converged with critic search at "
                        "render %s/%s critic %s/%s",
                        render_attempt,
                        max_visits,
                        critic_attempt,
                        max_visits,
                    )
                    return unit_state

        logger.info("Unit ontology loop exhausted retries")
        return unit_state
    except Exception as exc:
        logger.exception("Unhandled exception in ontology_loop")
        unit_state.set_failure(FailureStage.ONTOLOGY_CRITIQUE, str(exc))
        return unit_state
