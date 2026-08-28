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
import time
from collections.abc import Sequence
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
    TripleFix,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.context_resolver import (
    UnitOntologyContext,
    resolve_unit_ontology_context,
)
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import collect_unit_findings
from ontocast.tool.facts_validation.critic_findings import critic_fixes_to_findings
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def _document_supplemental_ontologies(context: UnitLoopContext) -> list[Ontology]:
    """Non-null reduced ontology artifacts for LLM ingest prefix repair."""
    return [
        ontology for ontology in context.reduced_artifacts() if not ontology.is_null()
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
    context: UnitLoopContext,
    unit_state: UnitOntologyState | UnitFactsState,
    tools: ToolBox,
) -> list[Ontology]:
    """Document artifacts plus catalog entries for the unit's patch sources."""
    merged: list[Ontology] = []
    seen: set[str] = set()
    for ontology in (
        *_document_supplemental_ontologies(context),
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


def _resolve_critic_visits(unit_state: UnitFactsState | UnitOntologyState) -> int:
    """Critic attempts allowed per render attempt.

    Unset means the legacy coupling to ``max_visits_per_node``: the inner
    critic loop shares the outer render loop's bound, so the worst case is
    ``max_visits ** 2`` billed critic calls. Setting it decouples the two.
    """
    override = unit_state.max_critic_visits_per_node
    if override is None:
        return unit_state.max_visits_per_node
    return max(1, override)


def _collect_facts_findings(
    unit_state: UnitFactsState,
    atomic: AtomicToolBox | None = None,
) -> list[FactsUnitFinding]:
    """Run the deterministic per-unit validator against the current graph.

    The toolbox supplies the deployment's namespace exemptions and quantity
    fallback vocabulary; ``None`` (tests) means no exemptions.
    """
    return collect_unit_findings(
        graph=unit_state.content_unit.graph,
        ontology_graph=unit_state.ontology_snapshot.graph,
        quarantined=unit_state.quarantined_literal_triples,
        extraction_text=unit_state.content_unit.extraction_text,
        fact_namespaces=[DEFAULT_IRI, str(unit_state.content_unit.doc_iri)],
        # Citation numerics (pages, years, volume numbers) are not extractable
        # quantities — never push coverage repair on bibliography units.
        coverage_limit=0 if unit_state.content_unit.is_citation_metadata else 30,
        policy=atomic.validation_policy if atomic is not None else None,
    )


def _record_facts_attempt(
    unit_state: UnitFactsState,
    *,
    kind: Literal["render", "critic", "llm_repair"],
    render_attempt: int,
    critic_attempt: int = 0,
    n_findings: int = 0,
    n_mandatory: int = 0,
    repair_failed: bool = False,
    repair_delete_only: bool = False,
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
            repair_delete_only=repair_delete_only,
            failure_stage=(str(unit_state.failure_stage) if repair_failed else None),
            failure_reason=unit_state.failure_reason if repair_failed else None,
            triple_count=len(graph),
        )
    )


async def _run_finding_driven_repair(
    unit_state: UnitFactsState,
    atomic,
    supplemental: list[Ontology],
    *,
    render_attempt: int,
    critic_fixes: Sequence[TripleFix] = (),
) -> UnitFactsState:
    """Repair machine-found violations with bounded render-update visits.

    Only the *trigger* here is deterministic: each visit is a paid
    ``render_facts_update`` call fed with the findings. Machine-applied,
    LLM-free rewrites are a different thing entirely and run at parse time
    (``agent/render_facts.py::_normalize_and_repair_graph``) and at the
    post-merge gate (``tool/facts_validation::apply_shacl_repairs``).

    Runs after the final render (where the LLM critic is skipped): mandatory
    findings — quarantined literals, unknown/near-miss terms — drive the loop.
    Advisory findings (numeric coverage) ride along in the prompt when a repair
    does run, but never trigger one on their own: they fire on nearly every
    unit of numeric prose, so gating on them cost an extra render per unit.
    A failed repair leaves the pre-repair graph intact (the patch path applies
    only parsed operations) and is recorded rather than erased.
    """
    repair_visits = atomic.facts_llm_repair_visits
    # Critic fixes join the deterministic findings for the first pass only.
    # They are consumed by the render that reads them, exactly like
    # `state.suggestions`; re-adding them after each pass would make a fix the
    # renderer declined an unresolvable finding and burn the whole budget.
    findings = [
        *_collect_facts_findings(unit_state, atomic),
        *critic_fixes_to_findings(critic_fixes, atomic.acceptance_policy),
    ]
    for repair_attempt in range(1, repair_visits + 1):
        mandatory = [finding for finding in findings if finding.mandatory]
        if not mandatory:
            break
        logger.info(
            "Finding-driven facts repair render %s/%s: %d finding(s) (%d mandatory)",
            repair_attempt,
            repair_visits,
            len(findings),
            len(mandatory),
        )
        unit_state.deterministic_findings = findings
        graph_before = unit_state.content_unit.graph.copy()
        triples_before = len(unit_state.content_unit.graph)
        mandatory_before = len(mandatory)
        unit_state = await render_facts(
            unit_state, atomic, supplemental_ontologies=supplemental
        )
        repair_failed = unit_state.status != Status.SUCCESS
        repair_delete_only = False
        if not repair_failed:
            findings = _collect_facts_findings(unit_state, atomic)
            mandatory_after = sum(1 for finding in findings if finding.mandatory)
            triples_after = len(unit_state.content_unit.graph)
            # The findings prompt orders every mandatory item to be fixed by
            # rewriting the offending term *in place*, so a genuine repair
            # always writes something back. A render that only removed triples
            # answered a repair order with a deletion, and the fact that the
            # finding is gone afterwards is the deletion working, not a fix.
            #
            # Keying on the finding count alone cannot see this: deleting the
            # flagged statement drops `mandatory_after` below
            # `mandatory_before`, so the dominant failure mode of the 2026-08
            # matsci runs -- 25 of 58 cached repair responses deleting valid
            # values outright -- scored as a successful repair. Comparing the
            # written triples catches it, and stays quiet for a rewrite that
            # happens to shrink the graph by collapsing a duplicate.
            wrote_nothing = not (unit_state.content_unit.graph - graph_before)
            deleted_something = bool(graph_before - unit_state.content_unit.graph)
            no_progress = (
                triples_after < triples_before and mandatory_after >= mandatory_before
            )
            if (deleted_something and wrote_nothing) or no_progress:
                # Roll back rather than keep the shrunken graph: the pre-repair
                # graph holds strictly more data, and its mandatory findings are
                # the ones the next attempt (or the residual metric) should see.
                logger.warning(
                    "Finding-driven repair deleted %d triple(s) and wrote %d "
                    "(%d -> %d triples, mandatory %d -> %d) — rolling back the "
                    "delete-only repair",
                    len(graph_before - unit_state.content_unit.graph),
                    len(unit_state.content_unit.graph - graph_before),
                    triples_before,
                    triples_after,
                    mandatory_before,
                    mandatory_after,
                )
                unit_state.content_unit.graph = graph_before
                findings = _collect_facts_findings(unit_state, atomic)
                repair_delete_only = True
        # Recorded counts are the residual AFTER this repair render (on failure
        # the graph is unchanged, so the pre-render findings still describe it).
        # This is what `facts_findings_residual` sums document-level; recording
        # the pre-render counts here measured what the repair was asked to fix,
        # not what survived it.
        _record_facts_attempt(
            unit_state,
            kind="llm_repair",
            render_attempt=render_attempt,
            critic_attempt=repair_attempt,
            n_findings=len(findings),
            n_mandatory=sum(1 for finding in findings if finding.mandatory),
            repair_failed=repair_failed,
            repair_delete_only=repair_delete_only,
        )
        if repair_failed:
            # The pre-repair graph is intact, so the unit is still usable and
            # the loop reports SUCCESS. The diagnosis is copied onto the attempt
            # record *before* the state is cleared -- clearing it outright left
            # `repair_failed=True` with no stage or reason, so a provider
            # timeout and an unparseable response looked identical.
            logger.warning(
                "Finding-driven facts repair render failed (%s: %s); keeping graph",
                unit_state.failure_stage,
                unit_state.failure_reason,
            )
            unit_state.clear_failure()
            unit_state.status = Status.SUCCESS
            break

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
    """Point unit state at the assembled context (snapshot + writable + sources).

    The snapshot is shared by reference, not copied. Every consumer in both unit
    loops treats it as read-only schema -- the facts loop mutates only the
    rendered facts graph, and the ontology loop edits ``working_graph``, keeping
    the snapshot as its pristine baseline for ``working_graph_changed()``.
    Deep-copying it per unit cost a full rdflib graph copy each time, on the
    event loop, for a value that is identical across the whole fan-out.
    """
    unit_state.ontology_snapshot = ctx.snapshot
    unit_state.ontology_patch_sources = list(ctx.patch_sources)
    unit_state.writable_iris = list(ctx.writable_iris)
    unit_state.assembly_anchor_iri = ctx.primary_writable_iri
    unit_state.assembly_mode_used = ctx.assembly_mode


async def _apply_facts_ontology_context(
    unit_state: UnitFactsState,
    context: UnitLoopContext,
    tools: ToolBox,
) -> UnitFactsState:
    """Set ontology_snapshot for facts from the per-unit context resolver.

    Only reached when the caller has no merged document context to hand down
    (single-unit pipelines, or facts-only runs with no ontology stage).
    """
    ctx = await resolve_unit_ontology_context(context, tools, unit_state.content_unit)
    logger.info(
        "Ontology context for mode %s: sources=%s writable=%s",
        context.ontology_context_mode,
        ctx.patch_sources,
        ctx.writable_iris,
    )
    _apply_unit_ontology_context(unit_state, ctx)
    return unit_state


async def facts_loop(
    state: UnitFactsState,
    tools: ToolBox,
    document_context: UnitLoopContext,
    max_visits_per_node: int | None = None,
    pre_resolved_context: UnitOntologyContext | None = None,
) -> UnitFactsState:
    """Run facts render/critic loop for one content unit.

    Args:
        state: Unit facts state to run the loop over.
        tools: Tool container.
        document_context: Document-level inputs, shared read-only.
        max_visits_per_node: Override for the render/critic bound.
        pre_resolved_context: Ontology context resolved once by the caller.
            The merged document ontology depends only on document-level state,
            so the fan-out builds it once and hands the *same object* to every
            unit; resolving it here instead cost one full rdflib merge and two
            graph copies per unit. Falls back to per-unit resolution when None.
    """
    atomic = tools.get_atomic_tools()
    unit_state = state.model_copy(deep=True)
    # Charge resolver LLM calls (e.g. ontology selection) to this unit's
    # tracker — the copy that survives the loop and is merged by the caller.
    # Shallow copy: retrieval_metrics stays shared with the caller's context.
    document_context = document_context.model_copy(
        update={"budget_tracker": unit_state.budget_tracker}
    )
    # The stage the loop is currently in, so an unhandled exception is
    # attributed to where it happened. Hardcoding the critique stage reported
    # a render or context-resolution crash as a failed critique.
    stage = FailureStage.GENERATE_GRAPH_UPDATE_FOR_FACTS
    try:
        if pre_resolved_context is not None:
            _apply_unit_ontology_context(unit_state, pre_resolved_context)
        else:
            unit_state = await _apply_facts_ontology_context(
                unit_state, document_context, tools
            )
        max_visits = _resolve_max_visits_limit(
            unit_state.max_visits_per_node, max_visits_per_node
        )
        unit_state.max_visits_per_node = max_visits

        for render_attempt in range(1, max_visits + 1):
            stage = FailureStage.GENERATE_GRAPH_UPDATE_FOR_FACTS
            unit_state.node_visits[WorkflowNode.TEXT_TO_FACTS] += 1
            _reset_node_evidence_context(unit_state, WorkflowNode.TEXT_TO_FACTS)
            supplemental = _supplemental_ontologies_for_unit(
                document_context, unit_state, tools
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
                    "(skipping LLM critic; finding-driven repair renders may "
                    "still run)",
                    render_attempt,
                    max_visits,
                )
                return await _run_finding_driven_repair(
                    unit_state,
                    atomic,
                    supplemental,
                    render_attempt=render_attempt,
                )

            stage = FailureStage.FACTS_CRITIQUE
            for critic_attempt in range(1, _resolve_critic_visits(unit_state) + 1):
                unit_state.node_visits[WorkflowNode.CRITICISE_FACTS] += 1
                _reset_node_evidence_context(unit_state, WorkflowNode.CRITICISE_FACTS)
                unit_state.deterministic_findings = _collect_facts_findings(
                    unit_state, atomic
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
                    return await _run_finding_driven_repair(
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
                        "Unit facts critic rejected at render %s/%s critic %s/%s "
                        "without search request; repairing in place",
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
                    return await _run_finding_driven_repair(
                        unit_state,
                        atomic,
                        supplemental,
                        render_attempt=render_attempt,
                    )

            # A rejecting critic no longer escalates to another full render.
            # It used to fall through to the next `render_attempt`, which
            # re-extracted the unit from scratch under a prompt that invited
            # unrequested rewriting -- the expensive, open-ended answer to a
            # signal that is now a list of specific defects. Its blocking fixes
            # go through the same bounded rewrite-in-place pass the
            # deterministic findings use. The outer loop therefore retries only
            # on *render failure*, and a unit's worst-case call count no longer
            # grows with MAX_VISITS.
            return await _run_finding_driven_repair(
                unit_state,
                atomic,
                supplemental,
                render_attempt=render_attempt,
                critic_fixes=unit_state.suggestions.actionable_fixes,
            )

        logger.info("Unit facts loop exhausted retries")
        return unit_state
    except Exception as exc:
        logger.exception("Unhandled exception in facts_loop")
        unit_state.set_failure(stage, str(exc))
        return unit_state


async def ontology_loop(
    state: UnitOntologyState,
    tools: ToolBox,
    document_context: UnitLoopContext,
    max_visits_per_node: int | None = None,
) -> UnitOntologyState:
    """Run ontology render/critic loop for one content unit.

    Per-unit ontology context is assembled via ``resolve_unit_ontology_context``
    before the first render.
    """
    atomic = tools.get_atomic_tools()
    unit_state = state.model_copy(deep=True)
    # Charge resolver LLM calls to this unit's surviving tracker; shallow copy
    # keeps retrieval_metrics shared with the caller's context.
    document_context = document_context.model_copy(
        update={"budget_tracker": unit_state.budget_tracker}
    )
    # See facts_loop: the stage an unhandled exception is attributed to tracks
    # where the loop actually is, rather than always naming the critique.
    stage = FailureStage.GENERATE_GRAPH_UPDATE_FOR_ONTOLOGY
    try:
        ctx = await resolve_unit_ontology_context(
            document_context, tools, unit_state.content_unit
        )
        _apply_unit_ontology_context(unit_state, ctx)
        working_copy_start = time.perf_counter()
        unit_state.working_graph = unit_state.ontology_snapshot.graph.copy()
        unit_state.budget_tracker.add_duration(
            "ctx/working_graph_copy", time.perf_counter() - working_copy_start
        )

        max_visits = _resolve_max_visits_limit(
            unit_state.max_visits_per_node, max_visits_per_node
        )
        unit_state.max_visits_per_node = max_visits

        for render_attempt in range(1, max_visits + 1):
            stage = FailureStage.GENERATE_GRAPH_UPDATE_FOR_ONTOLOGY
            unit_state.node_visits[WorkflowNode.TEXT_TO_ONTOLOGY] += 1
            _reset_node_evidence_context(unit_state, WorkflowNode.TEXT_TO_ONTOLOGY)
            supplemental = _supplemental_ontologies_for_unit(
                document_context, unit_state, tools
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

            stage = FailureStage.ONTOLOGY_CRITIQUE
            for critic_attempt in range(1, _resolve_critic_visits(unit_state) + 1):
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
        unit_state.set_failure(stage, str(exc))
        return unit_state
