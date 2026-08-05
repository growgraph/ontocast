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
from collections.abc import Sequence
from copy import deepcopy
from typing import Literal

from ontocast.agent.criticise_facts import criticise_facts
from ontocast.agent.criticise_ontology import criticise_ontology
from ontocast.agent.external_evidence import (
    fetch_external_evidence_for_node,
    plan_external_evidence_for_node,
)
from ontocast.agent.render_facts import render_facts
from ontocast.agent.render_ontology import render_ontology
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import (
    ExternalEvidenceCacheEntry,
    ExternalEvidenceRequest,
    FactsLoopAttempt,
    FactsUnitFinding,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.context_resolver import (
    UnitOntologyContext,
    resolve_effective_facts_ontology_context,
    resolve_unit_ontology_context,
)
from ontocast.tool.facts_invariants import collect_unit_findings
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def _document_supplemental_ontologies(document_state: AgentState) -> list[Ontology]:
    """Non-null reduced ontology artifacts for LLM ingest prefix repair."""
    return [
        ontology
        for ontology in document_ontology_access(document_state).reduced_artifacts()
        if not ontology.is_null()
    ]


def _catalog_ontologies_for_patch_sources(
    tools: ToolBox,
    patch_sources: list[str],
) -> list[Ontology]:
    """Freshest catalog terminals for each working-context source IRI."""
    if not patch_sources:
        return []
    mgr = tools.ontology_manager
    result: list[Ontology] = []
    seen: set[str] = set()
    for ref in patch_sources:
        iri = mgr.resolve_ontology_ref(ref) or ref
        if iri in seen:
            continue
        onto = mgr.get_freshest_terminal_ontology_by_iri(iri)
        if onto is None or onto.is_null():
            continue
        seen.add(onto.iri)
        result.append(onto)
    return result


def _supplemental_ontologies_for_unit(
    document_state: AgentState,
    unit_state: UnitOntologyState | UnitFactsState,
    tools: ToolBox,
) -> list[Ontology]:
    """Document artifacts plus catalog entries for the unit's patch sources."""
    merged: list[Ontology] = []
    seen: set[str] = set()
    for ontology in (
        *_document_supplemental_ontologies(document_state),
        *_catalog_ontologies_for_patch_sources(
            tools, list(unit_state.ontology_patch_sources)
        ),
    ):
        if ontology.iri in seen:
            continue
        seen.add(ontology.iri)
        merged.append(ontology)
    return merged


def _resolve_max_visits_limit(state_visits: int, override: int | None) -> int:
    """Return a safe visit limit while respecting explicit overrides."""
    visits = state_visits if override is None else override
    return max(1, visits)


def _skip_critic_after_final_render(render_attempt: int, max_visits: int) -> bool:
    """True when this render attempt is the last allowed; critic cannot drive a retry."""
    return render_attempt == max_visits


def _collect_facts_findings(
    unit_state: UnitFactsState,
    additional_standard_namespaces: Sequence[str] = (),
) -> list[FactsUnitFinding]:
    """Run the deterministic per-unit validator against the current graph."""
    return collect_unit_findings(
        graph=unit_state.content_unit.graph,
        ontology_graph=unit_state.ontology_snapshot.graph,
        quarantined=unit_state.quarantined_literal_triples,
        extraction_text=unit_state.content_unit.extraction_text,
        fact_namespaces=[DEFAULT_IRI, str(unit_state.content_unit.doc_iri)],
        # Citation numerics (pages, years, volume numbers) are not extractable
        # quantities — never push coverage repair on bibliography units.
        coverage_limit=0 if unit_state.content_unit.is_citation_metadata else 30,
        additional_standard_namespaces=additional_standard_namespaces,
    )


def _record_facts_attempt(
    unit_state: UnitFactsState,
    *,
    kind: Literal["render", "critic", "repair"],
    render_attempt: int,
    critic_attempt: int = 0,
    n_findings: int = 0,
    n_mandatory: int = 0,
    repair_failed: bool = False,
) -> None:
    """Append one telemetry record for the current loop attempt."""
    graph = unit_state.content_unit.graph
    unit_state.attempt_log.append(
        FactsLoopAttempt(
            render_attempt=render_attempt,
            critic_attempt=critic_attempt,
            kind=kind,
            success=unit_state.status == Status.SUCCESS,
            n_deterministic_findings=n_findings,
            n_mandatory_findings=n_mandatory,
            repair_failed=repair_failed,
            triple_count=len(graph),
        )
    )


async def _run_deterministic_repair(
    unit_state: UnitFactsState,
    atomic,
    supplemental: list[Ontology],
    *,
    render_attempt: int,
) -> UnitFactsState:
    """Repair machine-found violations with bounded render-update visits.

    Runs after the final render (where the LLM critic is skipped): mandatory
    findings — quarantined literals, unknown/near-miss terms — drive the loop.
    Advisory findings (numeric coverage) ride along in the prompt when a repair
    does run, but never trigger one on their own: they fire on nearly every
    unit of numeric prose, so gating on them cost an extra render per unit.
    A failed repair leaves the pre-repair graph intact (the patch path applies
    only parsed operations) and is recorded rather than erased.
    """
    repair_visits = atomic.facts_repair_visits
    findings = _collect_facts_findings(
        unit_state, atomic.additional_standard_namespaces
    )
    for repair_attempt in range(1, repair_visits + 1):
        mandatory = [finding for finding in findings if finding.mandatory]
        if not mandatory:
            break
        logger.info(
            "Deterministic facts repair %s/%s: %d finding(s) (%d mandatory)",
            repair_attempt,
            repair_visits,
            len(findings),
            len(mandatory),
        )
        unit_state.deterministic_findings = findings
        unit_state = await render_facts(
            unit_state, atomic, supplemental_ontologies=supplemental
        )
        repair_failed = unit_state.status != Status.SUCCESS
        _record_facts_attempt(
            unit_state,
            kind="repair",
            render_attempt=render_attempt,
            critic_attempt=repair_attempt,
            n_findings=len(findings),
            n_mandatory=len(mandatory),
            repair_failed=repair_failed,
        )
        if repair_failed:
            # The pre-repair graph is intact, so the unit is still usable and
            # the loop reports SUCCESS -- but the crash is recorded on the
            # attempt log so "repair converged" stays distinguishable from
            # "repair never ran".
            logger.warning("Deterministic facts repair render failed; keeping graph")
            unit_state.clear_failure()
            unit_state.status = Status.SUCCESS
            break
        findings = _collect_facts_findings(
            unit_state, atomic.additional_standard_namespaces
        )

    unit_state.deterministic_findings = findings
    if findings:
        mandatory_count = sum(1 for finding in findings if finding.mandatory)
        if mandatory_count:
            logger.warning(
                "%d mandatory deterministic finding(s) remain unresolved",
                mandatory_count,
            )
    return unit_state


def _reset_node_evidence_context(
    state: UnitFactsState | UnitOntologyState, node: WorkflowNode
) -> None:
    """Start node execution in no-search mode with empty evidence context."""
    state.set_external_evidence_request(node, ExternalEvidenceRequest())
    state.set_external_evidence_cache_entry(node, ExternalEvidenceCacheEntry())
    state.load_external_evidence_for_node(node)


def _apply_unit_ontology_context(
    unit_state: UnitFactsState | UnitOntologyState,
    ctx: UnitOntologyContext,
) -> None:
    """Copy assemble product onto unit state (snapshot + writable + sources)."""
    unit_state.ontology_snapshot = deepcopy(ctx.snapshot)
    unit_state.ontology_patch_sources = list(ctx.patch_sources)
    unit_state.writable_iris = list(ctx.writable_iris)
    unit_state.assembly_anchor_iri = ctx.primary_writable_iri
    unit_state.assembly_mode_used = ctx.assembly_mode


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
        "Ontology context for mode %s: sources=%s writable=%s",
        document_state.ontology_context_mode,
        ctx.patch_sources,
        ctx.writable_iris,
    )
    _apply_unit_ontology_context(unit_state, ctx)
    return unit_state


async def facts_loop(
    state: UnitFactsState,
    tools: ToolBox,
    document_state: AgentState,
    max_visits_per_node: int | None = None,
    pre_resolved_context: UnitOntologyContext | None = None,
) -> UnitFactsState:
    """Run facts render/critic loop for one content unit.

    Ontology context is resolved per unit before rendering unless
    ``pre_resolved_context`` is provided (sequential unit pipelines).
    """
    atomic = tools.get_atomic_tools()
    unit_state = state.model_copy(deep=True)
    try:
        if pre_resolved_context is not None:
            _apply_unit_ontology_context(unit_state, pre_resolved_context)
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
            supplemental = _supplemental_ontologies_for_unit(
                document_state, unit_state, tools
            )
            unit_state = await render_facts(
                unit_state, atomic, supplemental_ontologies=supplemental
            )
            _record_facts_attempt(
                unit_state, kind="render", render_attempt=render_attempt
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
                    _record_facts_attempt(
                        unit_state, kind="render", render_attempt=render_attempt
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
                    "(skipping LLM critic; running deterministic repair)",
                    render_attempt,
                    max_visits,
                )
                return await _run_deterministic_repair(
                    unit_state,
                    atomic,
                    supplemental,
                    render_attempt=render_attempt,
                )

            for critic_attempt in range(1, max_visits + 1):
                unit_state.node_visits[WorkflowNode.CRITICISE_FACTS] += 1
                _reset_node_evidence_context(unit_state, WorkflowNode.CRITICISE_FACTS)
                unit_state.deterministic_findings = _collect_facts_findings(
                    unit_state, atomic.additional_standard_namespaces
                )
                unit_state = await criticise_facts(unit_state, atomic)
                if unit_state.status == Status.SUCCESS:
                    logger.info(
                        "Unit facts loop converged at render %s/%s critic %s/%s",
                        render_attempt,
                        max_visits,
                        critic_attempt,
                        max_visits,
                    )
                    return await _run_deterministic_repair(
                        unit_state,
                        atomic,
                        supplemental,
                        render_attempt=render_attempt,
                    )

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
                    return await _run_deterministic_repair(
                        unit_state,
                        atomic,
                        supplemental,
                        render_attempt=render_attempt,
                    )

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
        _apply_unit_ontology_context(unit_state, ctx)
        unit_state.working_graph = unit_state.ontology_snapshot.graph.copy()

        max_visits = _resolve_max_visits_limit(
            unit_state.max_visits_per_node, max_visits_per_node
        )
        unit_state.max_visits_per_node = max_visits

        for render_attempt in range(1, max_visits + 1):
            unit_state.node_visits[WorkflowNode.TEXT_TO_ONTOLOGY] += 1
            _reset_node_evidence_context(unit_state, WorkflowNode.TEXT_TO_ONTOLOGY)
            supplemental = _supplemental_ontologies_for_unit(
                document_state, unit_state, tools
            )
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
