"""Reusable per-unit render/critic retry loops.

These loops are designed for map/reduce execution where each content unit
is processed independently. They deep-copy the incoming unit state, then run
render -> critic until success or retry exhaustion.

Ontology context assembly (``resolve_unit_ontology_context``) runs at the
start of ``ontology_loop``. For ``facts_loop``, when the document run is
``RenderMode.ONTOLOGY_AND_FACTS`` and a merged document ontology exists, facts
use that whole-document ontology and optional per-unit patch IRIs from the
ontology map phase (assembly mode ``PRIMARY_WITHOUT_RETRIEVAL``); otherwise
the resolver supplies context per ``OntologyContextMode`` / strategy.
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
from ontocast.onto.enum import OntologyAssemblyMode, RenderMode, Status, WorkflowNode
from ontocast.onto.model import ExternalEvidenceCacheEntry, ExternalEvidenceRequest
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.context_resolver import resolve_unit_ontology_context
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def _resolve_max_visits_limit(state_visits: int, override: int | None) -> int:
    """Return a safe visit limit while respecting explicit overrides."""
    visits = state_visits if override is None else override
    return max(1, visits)


def _reset_node_evidence_context(
    state: UnitFactsState | UnitOntologyState, node: WorkflowNode
) -> None:
    """Start node execution in no-search mode with empty evidence context."""
    state.set_external_evidence_request(node, ExternalEvidenceRequest())
    state.set_external_evidence_cache_entry(node, ExternalEvidenceCacheEntry())
    state.load_external_evidence_for_node(node)


async def _apply_facts_ontology_context(
    unit_state: UnitFactsState,
    document_state,
    tools: ToolBox,
) -> UnitFactsState:
    """Set ontology_snapshot for facts from merged document ontology or resolver."""
    from ontocast.onto.state import AgentState

    ds = document_state
    assert isinstance(ds, AgentState)
    use_merged = ds.render_mode == RenderMode.ONTOLOGY_AND_FACTS
    if use_merged:
        doc_onto = document_ontology_access(ds)
        primary = doc_onto.primary_ontology()
        if not primary.is_null():
            unit_state.ontology_snapshot = deepcopy(primary)
            ui = unit_state.content_unit.index
            ps = ds.unit_patch_sources.get(ui)
            if ps is not None:
                unit_state.ontology_patch_sources = list(ps)
            unit_state.assembly_anchor_iri = primary.iri
            unit_state.assembly_mode_used = (
                OntologyAssemblyMode.PRIMARY_WITHOUT_RETRIEVAL
            )
            return unit_state
    ctx = await resolve_unit_ontology_context(ds, tools, unit_state.content_unit)
    unit_state.ontology_snapshot = deepcopy(ctx.ontology_snapshot)
    unit_state.ontology_patch_sources = list(ctx.patch_sources)
    unit_state.assembly_anchor_iri = ctx.anchor_iri
    unit_state.assembly_mode_used = ctx.assembly_mode
    return unit_state


async def facts_loop(
    state: UnitFactsState,
    tools: ToolBox,
    document_state,
    max_visits_per_node: int | None = None,
) -> UnitFactsState:
    """Run facts render/critic loop for one content unit.

    When ``document_state`` is ``ONTOLOGY_AND_FACTS`` and a non-null merged
    ontology exists on document state, ``ontology_snapshot`` is taken from that
    whole-document ontology (and per-unit patch IRIs from the ontology map).
    Otherwise context is resolved per unit.
    """
    from ontocast.onto.state import AgentState

    assert isinstance(document_state, AgentState)
    atomic = tools.get_atomic_tools()
    unit_state = state.model_copy(deep=True)
    unit_state = await _apply_facts_ontology_context(unit_state, document_state, tools)
    max_visits = _resolve_max_visits_limit(
        unit_state.max_visits_per_node, max_visits_per_node
    )
    unit_state.max_visits_per_node = max_visits

    for render_attempt in range(1, max_visits + 1):
        unit_state.node_visits[WorkflowNode.TEXT_TO_FACTS] += 1
        _reset_node_evidence_context(unit_state, WorkflowNode.TEXT_TO_FACTS)
        unit_state = await render_facts(unit_state, atomic)
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
                unit_state = await render_facts(unit_state, atomic)
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


async def ontology_loop(
    state: UnitOntologyState,
    tools: ToolBox,
    document_state,
    max_visits_per_node: int | None = None,
) -> UnitOntologyState:
    """Run ontology render/critic loop for one content unit.

    Per-unit ontology context is assembled via ``resolve_unit_ontology_context``
    before the first render.
    """
    from ontocast.onto.state import AgentState

    assert isinstance(document_state, AgentState)
    atomic = tools.get_atomic_tools()
    unit_state = state.model_copy(deep=True)
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
        unit_state = await render_ontology(unit_state, atomic)
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
                unit_state = await render_ontology(unit_state, atomic)
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

        for critic_attempt in range(1, max_visits + 1):
            unit_state.node_visits[WorkflowNode.CRITICISE_ONTOLOGY] += 1
            _reset_node_evidence_context(unit_state, WorkflowNode.CRITICISE_ONTOLOGY)
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
