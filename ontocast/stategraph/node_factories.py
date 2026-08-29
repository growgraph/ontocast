import asyncio
import logging
import time
from collections.abc import AsyncIterator, Coroutine, Sequence
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from rdflib import RDFS, Literal, URIRef

from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.agent.render_ontology import render_ontology_update
from ontocast.agent.summarize_chunks import ensure_unit_summary
from ontocast.onto.content_unit import ContentUnit, OutputType, SourceUnit
from ontocast.onto.enum import (
    OntologyAssemblyMode,
    OntologyContextMode,
    RetrievalMetric,
    Status,
    WorkflowNode,
)
from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.model import (
    UnitFailure,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.ontology_apply import (
    OntologyDelta,
    apply_partitioned_updates,
    partition_triples_by_namespace,
)
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import UNIT_SUM_SUFFIX, AgentState, BudgetTracker
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.atomic import facts_loop, ontology_loop
from ontocast.stategraph.context_resolver import (
    aggregate_writable_metrics,
    build_merged_document_ontology_context,
)
from ontocast.stategraph.facts_gate import run_facts_gate
from ontocast.stategraph.helpers import (
    all_unit_patch_source_iris,
    build_document_excerpt,
    enforce_redeclared_deletes,
    merge_unit_deltas,
    reconcile_fresh_ontologies,
)
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.ontology_validation import (
    apply_minted_duplicate_rewrites,
    detect_minted_duplicates,
)
from ontocast.tool.validate import RDFGraphConnectivityValidator
from ontocast.toolbox import ToolBox
from ontocast.util.loop_lag import loop_lag

logger = logging.getLogger(__name__)

T = TypeVar("T")


@asynccontextmanager
async def _unit_slot(semaphore: asyncio.Semaphore) -> AsyncIterator[float]:
    """Hold a unit-worker slot, yielding how long acquiring it took.

    The wait is yielded rather than recorded, because the unit loops deep-copy
    the state they are handed -- the tracker that survives is the one on the
    *returned* state, so the caller must charge it there.

    A non-zero ``"<node>/worker_wait"`` means units queued behind
    ``PARALLEL_WORKERS``, so widening the fan-out would help. Near zero means
    the configured width is not what limits the stage.
    """
    wait_start = time.perf_counter()
    async with semaphore:
        yield time.perf_counter() - wait_start


async def _gather_units(
    node: WorkflowNode,
    state: AgentState,
    tasks: Sequence[Coroutine[Any, Any, T]],
) -> tuple[list[T], int]:
    """Run per-unit tasks concurrently, recording the stage's event-loop stall.

    Awaited provider calls yield and produce no lag, so ``"<node>/loop_lag_total"``
    isolates synchronous CPU work that blocked every other unit -- the thing that
    makes a nominally N-way fan-out behave like a serial loop. Read it together
    with :meth:`~ontocast.onto.state.BudgetTracker.parallel_efficiency`.

    Failures are isolated to their own unit. The unit loops already catch their
    own errors, but the code around them (state construction, context
    projection) does not, and a bare ``gather`` would let one such error abort
    the node while its siblings kept running as orphans -- their provider spend
    billed and then discarded.

    Args:
        node: Fan-out node the tasks belong to; namespaces the metric keys.
        state: Document state whose tracker the stage metrics land on.
        tasks: Per-unit coroutines to run concurrently.

    Returns:
        tuple: Successful results in submission order, and the number of units
        that raised.
    """
    async with loop_lag() as lag:
        raw = await asyncio.gather(*tasks, return_exceptions=True)
    state.budget_tracker.add_duration(f"{node}/loop_lag_total", lag.total)
    state.budget_tracker.add_duration(f"{node}/loop_lag_max", lag.peak)

    results: list[T] = []
    failures = 0
    for index, item in enumerate(raw):
        if isinstance(item, BaseException):
            failures += 1
            state.budget_tracker.incr(f"{node}/unit_errors")
            logger.exception(
                "Unit %s raised during %s: %s", index, node, item, exc_info=item
            )
            continue
        results.append(item)
    return results, failures


def _index_ontologies_by_anchor(artifacts: list[Ontology]) -> dict[str, Ontology]:
    return {ontology.iri: ontology for ontology in artifacts if ontology.iri}


def _empty_unit_snapshot() -> OntologySnapshot:
    return OntologySnapshot.empty(
        title="Pending context resolve",
        description="Placeholder until resolve_unit_ontology_context runs.",
    )


def _map_stage_status(failed_without_output: int, total_units: int) -> Status:
    """Status for a completed map stage.

    A stage where *every* unit failed is a failure, not a success with empty
    output. Both map nodes previously ended with an unconditional
    ``Status.SUCCESS``, so a document whose units all died -- or one whose
    conversion failed upstream -- still returned HTTP 200 with empty facts.
    Partial failure stays SUCCESS: the surviving units produced real output,
    and the casualties are recorded in ``state.unit_failures``.
    """
    if total_units and failed_without_output >= total_units:
        return Status.FAILED
    return Status.SUCCESS


def make_render_ontology_node(tools: ToolBox):
    async def render_ontology_updates(state: AgentState) -> AgentState:
        if not state.content_units:
            state.ontology_units = []
            # Do not overwrite an upstream failure: conversion and chunking set
            # FAILED and the graph edge into this node is unconditional, so
            # clobbering it here turned a failed conversion into HTTP 200.
            if state.status != Status.FAILED:
                state.status = Status.SUCCESS
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        async def process_unit(
            unit_index: int,
        ) -> tuple[int, UnitOntologyState, str, list[str], OntologyAssemblyMode]:
            async with _unit_slot(semaphore) as worker_wait:
                unit_budget = BudgetTracker()
                # Before building the unit state: the loop deep-copies the
                # content unit, so a later write would not be visible to it.
                await ensure_unit_summary(state, unit_index, tools, unit_budget)
                unit_context = UnitLoopContext.from_agent_state(state, unit_budget)
                ontology_state = UnitOntologyState(
                    content_unit=state.content_units[unit_index],
                    ontology_snapshot=_empty_unit_snapshot(),
                    ontology_patch_sources=[],
                    ontology_user_instruction=state.ontology_user_instruction,
                    budget_tracker=unit_budget,
                    max_visits_per_node=state.max_visits,
                    max_critic_visits_per_node=(
                        tools.config.server.max_critic_visits_per_node
                    ),
                    current_domain=state.current_domain,
                    ontology_max_triples=tools.config.server.ontology_max_triples,
                    llm_graph_format=state.llm_graph_format,
                    ontology_context_max_triples=tools.config.server.ontology_context_max_triples,
                )
                loop_start = time.perf_counter()
                result = await ontology_loop(ontology_state, tools, unit_context)
                result.budget_tracker.add_duration(
                    f"{WorkflowNode.RENDER_ONTOLOGY_UPDATE}{UNIT_SUM_SUFFIX}",
                    time.perf_counter() - loop_start,
                )
                result.budget_tracker.add_duration(
                    f"{WorkflowNode.RENDER_ONTOLOGY_UPDATE}/worker_wait", worker_wait
                )
                # Per-unit resolver metrics previously landed on a discarded
                # deep copy; fold them back (last writer wins on shared keys).
                state.retrieval_metrics.update(unit_context.retrieval_metrics)
                return (
                    unit_index,
                    result,
                    result.assembly_anchor_iri,
                    list(result.writable_iris or result.ontology_patch_sources),
                    result.assembly_mode_used,
                )

        tasks = [process_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results, unit_errors = await _gather_units(
            WorkflowNode.RENDER_ONTOLOGY_UPDATE, state, tasks
        )
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        ontology_units: list[ContentUnit] = []
        unit_deltas: list[OntologyDelta] = []
        fresh_ontologies: list[Ontology] = []
        failed_without_output_count = unit_errors
        salvaged_failed_count = 0
        unit_contexts: dict[int, tuple[str, list[str], OntologyAssemblyMode]] = {}
        all_writable: list[str] = []
        seen_writable: set[str] = set()

        # Accumulated over *every* unit (see the facts reduce below): the
        # residual's denominator is "units", not "units that ran a critic".
        ontology_findings_residual = 0
        ontology_mandatory_residual = 0
        for (
            unit_index,
            result,
            primary_iri,
            writable_iris,
            assembly_mode,
        ) in ordered_results:
            state.budget_tracker.merge_from(result.budget_tracker)
            ontology_findings_residual += len(result.deterministic_findings)
            ontology_mandatory_residual += sum(
                1 for finding in result.deterministic_findings if finding.mandatory
            )
            if result.attempt_log:
                state.ontology_loop_telemetry[unit_index] = list(result.attempt_log)
            unit_contexts[unit_index] = (
                primary_iri,
                list(result.ontology_patch_sources),
                assembly_mode,
            )
            for iri in writable_iris:
                if iri and iri not in seen_writable:
                    seen_writable.add(iri)
                    all_writable.append(iri)

            has_output = bool(result.all_updates) or result.working_graph_changed()
            if (
                result.fresh_ontology is not None
                and not result.fresh_ontology.is_null()
            ):
                fresh_ontologies.append(result.fresh_ontology)
                has_output = True

            if not has_output:
                failed_without_output_count += 1
                state.unit_failures.append(
                    UnitFailure(
                        unit_index=unit_index,
                        phase="ontology",
                        stage=(
                            result.failure_stage.value
                            if result.failure_stage is not None
                            else None
                        ),
                        reason=result.failure_reason,
                    )
                )
                continue

            content_unit = result.content_unit
            delta = result.build_delta()
            if not delta.is_empty():
                unit_deltas.append(delta)
            if len(delta.inserts) > 0:
                ontology_units.append(
                    ContentUnit(
                        text=content_unit.text,
                        index=content_unit.index,
                        doc_iri=content_unit.doc_iri,
                        graph=delta.inserts,
                        type=OutputType.ONTOLOGIES,
                    )
                )
            if result.status != Status.SUCCESS:
                salvaged_failed_count += 1

        if failed_without_output_count:
            logger.warning(
                "Parallel ontology map failed without usable output for "
                f"{failed_without_output_count}/{len(state.content_units)} unit(s)"
            )
        if salvaged_failed_count:
            logger.warning(
                "Parallel ontology map salvaged output from non-converged loop(s): "
                f"{salvaged_failed_count}/{len(state.content_units)} unit(s)"
            )

        _, state.unit_patch_sources, _, primary_counts = aggregate_writable_metrics(
            unit_contexts
        )
        state.retrieval_metrics[RetrievalMetric.ONTOLOGY_WRITABLE_COUNT] = len(
            seen_writable
        )
        state.retrieval_metrics[RetrievalMetric.ONTOLOGY_PRIMARY_UNITS] = sum(
            primary_counts.values()
        )
        state.retrieval_metrics[RetrievalMetric.ONTOLOGY_FINDINGS_RESIDUAL] = (
            ontology_findings_residual
        )
        state.retrieval_metrics[RetrievalMetric.ONTOLOGY_MANDATORY_RESIDUAL] = (
            ontology_mandatory_residual
        )
        # The ontology critic's own ledger, mirroring the facts block: without
        # it, whether the critic ran at all (it never does at MAX_VISITS=1) and
        # how often `success or score > 90` accepted are unrecoverable from a
        # run's artifacts.
        ontology_critic_attempts = [
            attempt
            for attempts in state.ontology_loop_telemetry.values()
            for attempt in attempts
            if attempt.kind == "critic"
        ]
        state.retrieval_metrics[RetrievalMetric.ONTOLOGY_CRITIC_CALLS] = len(
            ontology_critic_attempts
        )
        state.retrieval_metrics[RetrievalMetric.ONTOLOGY_CRITIC_ACCEPTED] = sum(
            1 for attempt in ontology_critic_attempts if attempt.success
        )

        # Document-level insert/delete consensus + namespace apply onto catalog bases.
        merged_delta = merge_unit_deltas(unit_deltas)

        # Single-ontology modes are untouched by the delete policy: there the
        # model saw the whole graph, so its deletes were judged on full
        # evidence.
        if (
            state.ontology_context_mode
            == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
        ):
            state.ontology_reduce_metrics["deletes_dropped_unredeclared"] = (
                enforce_redeclared_deletes(merged_delta)
            )

        # Minted-duplicate reconciliation against the FULL terminals. The
        # per-unit label-collision check indexes the snapshot — under
        # vector retrieval that is exactly the part of the catalog where the
        # duplicate is not, so a term the retrieval failed to surface gets
        # re-minted under a fresh IRI and nothing else on the write path would
        # ever notice. 'detect' (default) only measures; 'rewrite' substitutes
        # after a sampling run has shown the matches are true duplicates.
        reconcile_mode = (
            tools.config.get_tool_config().ontology_validation.reconcile_minted_terms
        )
        if reconcile_mode != "off" and len(merged_delta.inserts) > 0 and all_writable:
            terminal_graphs = {
                iri: terminal.graph
                for iri in all_writable
                if (
                    terminal
                    := tools.ontology_manager.get_freshest_terminal_ontology_by_iri(iri)
                )
                is not None
                and not terminal.is_null()
            }
            duplicates = detect_minted_duplicates(merged_delta.inserts, terminal_graphs)
            state.ontology_reduce_metrics["minted_duplicates"] = len(duplicates)
            if duplicates:
                state.ontology_reduce_metrics["minted_duplicate_pairs"] = [
                    duplicate.model_dump() for duplicate in duplicates
                ]
                for duplicate in duplicates:
                    logger.warning(
                        "Minted term <%s> duplicates catalog term <%s> "
                        "(surface %r, role %s)%s",
                        duplicate.minted_iri,
                        duplicate.catalog_iri,
                        duplicate.surface,
                        duplicate.role,
                        " — rewriting" if reconcile_mode == "rewrite" else "",
                    )
                if reconcile_mode == "rewrite":
                    state.ontology_reduce_metrics["minted_duplicates_rewritten"] = (
                        apply_minted_duplicate_rewrites(
                            merged_delta.inserts, duplicates
                        )
                    )

        fresh_ontologies, fresh_metrics = reconcile_fresh_ontologies(fresh_ontologies)
        state.ontology_reduce_metrics.update(fresh_metrics)

        artifacts: list[Ontology] = list(fresh_ontologies)
        if not merged_delta.is_empty() and all_writable:
            partitioned_inserts, unattributed = partition_triples_by_namespace(
                merged_delta.inserts,
                writable_iris=all_writable,
                ontology_manager=tools.ontology_manager,
            )
            state.ontology_reduce_metrics["unattributed_insert_triples"] = unattributed
            partitioned_deletes, unattributed_deletes = partition_triples_by_namespace(
                merged_delta.deletes,
                writable_iris=all_writable,
                ontology_manager=tools.ontology_manager,
            )
            state.ontology_reduce_metrics["unattributed_delete_triples"] = (
                unattributed_deletes
            )
            applied, apply_metrics, applied_updates = apply_partitioned_updates(
                partitioned_inserts,
                ontology_manager=tools.ontology_manager,
                normalize_units_fn=normalize_ontology_units,
                tools=tools,
                partitioned_deletes=partitioned_deletes,
            )
            artifacts.extend(applied)
            state.ontology_updates_applied.extend(applied_updates)
            state.ontology_reduce_metrics.update(apply_metrics)
        elif not merged_delta.is_empty() and not all_writable:
            logger.warning(
                "Ontology map produced %s complement / %s delete triples but no "
                "writable catalog IRIs; skipping catalog apply",
                len(merged_delta.inserts),
                len(merged_delta.deletes),
            )

        state.ontology_artifacts = artifacts
        state.reduced_ontology_artifacts = list(artifacts)
        state.reduced_ontology_by_anchor = _index_ontologies_by_anchor(artifacts)
        state.ontology_reduce_metrics["reduced_artifact_count"] = len(artifacts)
        state.ontology_units = ontology_units
        state.status = _map_stage_status(
            failed_without_output_count, len(state.content_units)
        )
        return state

    return render_ontology_updates


def make_normalize_ontology_node(tools: ToolBox):
    """Normalize is largely handled in the map stage via namespace apply.

    Kept as a no-op success node when artifacts already carry catalog lineage,
    so the graph topology (map → normalize → …) stays stable.
    """

    def normalize_ontology_updates(state: AgentState) -> AgentState:
        if (
            not state.ontology_units
            and not document_ontology_access(state).reduced_artifacts()
        ):
            state.ontology_provenance_artifact = RDFGraph()
            state.status = Status.SUCCESS
            return state

        # Artifacts from map already applied onto catalog bases. Ensure indexes.
        artifacts = document_ontology_access(state).reduced_artifacts()
        state.reduced_ontology_by_anchor = _index_ontologies_by_anchor(artifacts)
        state.ontology_provenance_artifact = (
            state.ontology_provenance_artifact or RDFGraph()
        )
        state.ontology_reduce_metrics["normalized_ontology_updates"] = len(artifacts)
        state.status = Status.SUCCESS
        return state

    return normalize_ontology_updates


def make_consolidate_ontology_node(tools: ToolBox):
    atomic_tools = tools.get_atomic_tools()

    async def consolidate_ontology(state: AgentState) -> AgentState:
        """Optional post-normalization ontology consolidation pass."""
        if not tools.config.server.enable_ontology_consolidation:
            logger.info(
                "Skipping ontology consolidation: enable_ontology_consolidation is false"
            )
            state.status = Status.SUCCESS
            return state
        doc_onto = document_ontology_access(state)
        artifacts = doc_onto.reduced_artifacts()
        if not state.render_ontology or len(artifacts) != 1 or artifacts[0].is_null():
            logger.info(
                "Skipping ontology consolidation: requires exactly one rendered ontology artifact"
            )
            state.status = Status.SUCCESS
            return state

        excerpt = build_document_excerpt(state).strip()
        if not excerpt:
            logger.info(
                "Skipping ontology consolidation: no usable document excerpt was produced"
            )
            state.status = Status.SUCCESS
            return state

        consolidation_unit = SourceUnit(
            text=excerpt,
            index=0,
            doc_iri=state.doc_iri,
            type=OutputType.ONTOLOGIES,
        )
        consolidation_instruction = (
            "Consolidation pass: keep ontology IRI, ontology_id, and prefix unchanged. "
            "Harmonize duplicated or semantically overlapping classes/properties, "
            "normalize naming consistency, and improve hierarchy coherence."
        )
        ontology_user_instruction = (
            f"{state.ontology_user_instruction}\n\n{consolidation_instruction}".strip()
        )
        primary = artifacts[0]
        snap = OntologySnapshot.from_ontology(
            primary,
            assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
            title="Consolidation snapshot",
        )
        consolidation_state = UnitOntologyState(
            content_unit=consolidation_unit,
            ontology_snapshot=snap,
            ontology_patch_sources=all_unit_patch_source_iris(state),
            writable_iris=[primary.iri] if primary.iri else [],
            ontology_user_instruction=ontology_user_instruction,
            budget_tracker=state.budget_tracker,
            max_visits_per_node=1,
            current_domain=state.current_domain,
            ontology_max_triples=tools.config.server.ontology_max_triples,
            llm_graph_format=state.llm_graph_format,
            ontology_context_max_triples=tools.config.server.ontology_context_max_triples,
            working_graph=snap.graph.copy(),
            assembly_anchor_iri=primary.iri or "",
        )
        result = await render_ontology_update(consolidation_state, atomic_tools)
        if result.status == Status.SUCCESS and result.working_graph_changed():
            delta = result.build_delta()
            if not delta.is_empty() and primary.iri:
                # The consolidation delta is a complement of `primary` (the
                # map-stage artifact), so it must be applied on top of exactly
                # that artifact — not the pre-run catalog terminal, which would
                # silently drop the map-stage additions.
                base_overrides = {primary.iri: primary}
                partitioned_inserts, _unattr = partition_triples_by_namespace(
                    delta.inserts,
                    writable_iris=[primary.iri],
                    ontology_manager=tools.ontology_manager,
                    base_overrides=base_overrides,
                )
                partitioned_deletes, _unattr_del = partition_triples_by_namespace(
                    delta.deletes,
                    writable_iris=[primary.iri],
                    ontology_manager=tools.ontology_manager,
                    base_overrides=base_overrides,
                )
                applied, _metrics, applied_updates = apply_partitioned_updates(
                    partitioned_inserts,
                    ontology_manager=tools.ontology_manager,
                    normalize_units_fn=normalize_ontology_units,
                    tools=tools,
                    partitioned_deletes=partitioned_deletes,
                    base_overrides=base_overrides,
                )
                if applied:
                    state.reduced_ontology_artifacts = applied
                    state.reduced_ontology_by_anchor = _index_ontologies_by_anchor(
                        applied
                    )
                    state.ontology_artifacts = applied
                    state.ontology_updates_applied.extend(applied_updates)
                    logger.info(
                        "Ontology consolidation applied %s update operation(s).",
                        len(applied_updates),
                    )
                else:
                    logger.warning(
                        "Ontology consolidation produced deltas but catalog apply "
                        "returned no artifacts; keeping the map-stage artifact."
                    )
            else:
                logger.warning(
                    "Ontology consolidation was enabled but no complement triples "
                    "were produced."
                )
        else:
            logger.warning(
                "Ontology consolidation was enabled but no update was applied."
            )
        state.status = Status.SUCCESS
        return state

    return consolidate_ontology


def make_render_facts_node(tools: ToolBox):
    async def render_facts(state: AgentState) -> AgentState:
        if not state.content_units:
            state.facts_units = []
            if state.status != Status.FAILED:
                state.status = Status.SUCCESS
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        # Built once for the whole document, not once per unit. It reads only
        # reduced_ontology_artifacts, which the ontology stage froze upstream,
        # so every unit would otherwise pay an identical full rdflib merge plus
        # two graph copies -- synchronously, on the event loop, stalling every
        # other unit's in-flight provider call. None means no ontology stage ran
        # (facts-only mode); units then resolve their own context as before.
        merged_context = build_merged_document_ontology_context(
            UnitLoopContext.from_agent_state(state)
        )
        if merged_context is not None:
            # Hand the same graph to merge/validate downstream instead of
            # letting each rebuild it.
            state.facts_ontology_context = merged_context.snapshot.graph

        async def process_unit(
            unit_index: int,
        ) -> tuple[int, UnitFactsState, str, list[str], OntologyAssemblyMode]:
            async with _unit_slot(semaphore) as worker_wait:
                unit_budget = BudgetTracker()
                # No-op when the ontology fan-out already summarised this unit;
                # does the work when facts run without an ontology stage.
                await ensure_unit_summary(state, unit_index, tools, unit_budget)
                unit_context = UnitLoopContext.from_agent_state(state, unit_budget)
                facts_state = UnitFactsState(
                    content_unit=state.content_units[unit_index],
                    ontology_snapshot=_empty_unit_snapshot(),
                    ontology_patch_sources=[],
                    facts_user_instruction=state.facts_user_instruction,
                    budget_tracker=unit_budget,
                    max_visits_per_node=state.max_visits,
                    max_critic_visits_per_node=(
                        tools.config.server.max_critic_visits_per_node
                    ),
                    llm_graph_format=state.llm_graph_format,
                    ontology_context_max_triples=tools.config.server.ontology_context_max_triples,
                )
                loop_start = time.perf_counter()
                result = await facts_loop(
                    facts_state,
                    tools,
                    unit_context,
                    pre_resolved_context=merged_context,
                )
                result.budget_tracker.add_duration(
                    f"{WorkflowNode.RENDER_FACTS}{UNIT_SUM_SUFFIX}",
                    time.perf_counter() - loop_start,
                )
                result.budget_tracker.add_duration(
                    f"{WorkflowNode.RENDER_FACTS}/worker_wait", worker_wait
                )
                # Per-unit resolver metrics previously landed on a discarded
                # deep copy; fold them back (last writer wins on shared keys).
                state.retrieval_metrics.update(unit_context.retrieval_metrics)
                return (
                    unit_index,
                    result,
                    result.assembly_anchor_iri,
                    list(result.ontology_patch_sources),
                    result.assembly_mode_used,
                )

        tasks = [process_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results, unit_errors = await _gather_units(
            WorkflowNode.RENDER_FACTS, state, tasks
        )
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        facts_units: list[ContentUnit] = []
        failed_without_output_count = unit_errors
        salvaged_failed_count = 0
        # Accumulated over *every* unit, whether or not it ran a repair render,
        # so the residual has "units" as its denominator.
        findings_residual = 0
        mandatory_residual = 0
        unit_contexts: dict[int, tuple[str, list[str], OntologyAssemblyMode]] = {}
        for (
            unit_index,
            result,
            anchor_iri,
            patch_sources,
            assembly_mode,
        ) in ordered_results:
            state.budget_tracker.merge_from(result.budget_tracker)
            findings_residual += len(result.deterministic_findings)
            mandatory_residual += sum(
                1 for finding in result.deterministic_findings if finding.mandatory
            )
            if result.attempt_log:
                state.facts_loop_telemetry[unit_index] = list(result.attempt_log)
            if result.applied_repairs:
                state.facts_repairs_applied[unit_index] = list(result.applied_repairs)
            unit_contexts[unit_index] = (anchor_iri, patch_sources, assembly_mode)
            has_output = len(result.content_unit.graph) > 0
            if not has_output:
                failed_without_output_count += 1
                state.unit_failures.append(
                    UnitFailure(
                        unit_index=unit_index,
                        phase="facts",
                        stage=(
                            result.failure_stage.value
                            if result.failure_stage is not None
                            else None
                        ),
                        reason=result.failure_reason,
                    )
                )
                continue

            facts_units.append(result.content_unit)
            if result.status != Status.SUCCESS:
                salvaged_failed_count += 1

        if failed_without_output_count:
            logger.warning(
                "Parallel facts map failed without usable output for "
                f"{failed_without_output_count}/{len(state.content_units)} unit(s)"
            )
        if salvaged_failed_count:
            logger.warning(
                "Parallel facts map salvaged output from non-converged loop(s): "
                f"{salvaged_failed_count}/{len(state.content_units)} unit(s)"
            )

        _, state.unit_patch_sources, _, anchor_counts = aggregate_writable_metrics(
            unit_contexts
        )
        state.retrieval_metrics[RetrievalMetric.FACTS_ANCHOR_COUNT] = len(anchor_counts)
        state.retrieval_metrics[RetrievalMetric.FACTS_ANCHOR_UNITS] = sum(
            anchor_counts.values()
        )
        all_attempts = [
            attempt
            for attempts in state.facts_loop_telemetry.values()
            for attempt in attempts
        ]
        state.retrieval_metrics[RetrievalMetric.FACTS_LLM_REPAIR_RENDERS_TOTAL] = sum(
            1 for attempt in all_attempts if attempt.kind == "llm_repair"
        )
        # A repair render that itself fails leaves the pre-repair graph intact
        # and the unit reports SUCCESS, so without this the crash is recorded on
        # the attempt log and observed nowhere.
        state.retrieval_metrics[RetrievalMetric.FACTS_LLM_REPAIR_RENDERS_FAILED] = sum(
            1 for attempt in all_attempts if attempt.repair_failed
        )
        # Repair renders rolled back for answering the findings prompt with
        # deletions. Non-zero means the prompt or the validator is provoking
        # data-destroying responses -- the failure mode that cost the 2026-08
        # matsci arms 38-64% of their value nodes while logging nothing a run
        # could be judged by.
        state.retrieval_metrics[RetrievalMetric.FACTS_REPAIR_DELETE_ONLY] = sum(
            1 for attempt in all_attempts if attempt.repair_delete_only
        )
        # Residual is read off each unit's final findings, not off the attempt
        # log. Summing `attempts[-1]` where `kind == "llm_repair"` silently
        # contributed 0 for every unit that never ran a repair render -- the
        # clean ones and the ones whose loop exhausted its retries -- so the
        # metric's denominator was "units that needed repair", not "units", and
        # a change that made *fewer* units enter repair read as a drop in
        # residual findings. It also summed total findings, so advisory
        # NUMERIC_COVERAGE (which fires on nearly every unit of numeric prose)
        # dominated the number that was supposed to track mandatory defects.
        state.retrieval_metrics[RetrievalMetric.FACTS_FINDINGS_RESIDUAL] = (
            findings_residual
        )
        state.retrieval_metrics[RetrievalMetric.FACTS_MANDATORY_RESIDUAL] = (
            mandatory_residual
        )
        # The critic's own ledger. `node_visits` counting CRITICISE_FACTS lived
        # on the per-unit state copy and died with it, so the number of critic
        # calls a run bought was not recoverable from its own artifacts.
        critic_attempts = [a for a in all_attempts if a.kind == "critic"]
        state.retrieval_metrics[RetrievalMetric.FACTS_CRITIC_CALLS] = len(
            critic_attempts
        )
        state.retrieval_metrics[RetrievalMetric.FACTS_CRITIC_ACCEPTED] = sum(
            1 for attempt in critic_attempts if attempt.success
        )
        state.facts_units = facts_units
        state.status = _map_stage_status(
            failed_without_output_count, len(state.content_units)
        )
        return state

    return render_facts


def _facts_aggregation_inputs(state: AgentState) -> tuple[RDFGraph, dict]:
    """Ontology context and document metadata shared by merge and validate.

    Reuses the graph the facts fan-out already merged. Only falls back to
    merging here when the fan-out did not run (facts-only entry points), which
    is also why the fallback is not cached: there is nothing to reuse it from.
    """
    ontology_graph = RDFGraph()
    if len(state.facts_ontology_context) > 0:
        ontology_graph = state.facts_ontology_context
    else:
        merged_context = build_merged_document_ontology_context(
            UnitLoopContext.from_agent_state(state)
        )
        if merged_context is not None and len(merged_context.snapshot.graph) > 0:
            ontology_graph = merged_context.snapshot.graph
            state.facts_ontology_context = ontology_graph
    document_metadata = dict(state.document_metadata)
    if (
        state.source_url
        and "source_url" not in document_metadata
        and "source_uri" not in document_metadata
    ):
        document_metadata["source_url"] = state.source_url
    return ontology_graph, document_metadata


def make_merge_facts_node(tools: ToolBox):
    def merge_facts(state: AgentState) -> AgentState:
        if not state.facts_units:
            state.aggregated_facts = RDFGraph()
            if state.status != Status.FAILED:
                state.status = Status.SUCCESS
            return state

        ontology_graph, document_metadata = _facts_aggregation_inputs(state)
        result = tools.aggregator.postprocess_facts_units(
            units=state.facts_units,
            ontology_graph=ontology_graph,
            doc_iri=state.doc_iri,
            document_metadata=document_metadata,
            doc_namespace=state.doc_namespace,
        )
        state.aggregated_facts = result.graph
        state.aggregation_clusters = result.merged_clusters
        state.aggregation_key_clusters = result.key_supported_clusters
        state.retrieval_metrics[RetrievalMetric.FACTS_REJECTED_MERGES] = (
            result.rejected_merge_count
        )
        if len(state.aggregated_facts) == 0:
            logger.warning(
                "Facts aggregation produced an empty graph from "
                f"{len(state.facts_units)} successful unit(s)."
            )
        state.status = Status.SUCCESS
        return state

    return merge_facts


# Finding kinds whose signature IS a bad identity merge: two things that were
# not the same got one IRI. Un-merging them is a plausible repair.
#
# Deliberately excludes SHACL. A constraint violation says a node is
# under-specified or mistyped against a shape -- a missing required property,
# a datatype mismatch -- which is orthogonal to identity. Feeding it to the
# repair dissolved legitimate clusters whenever the focus node happened to be
# merged, and let SHACL dominate the loop's accept test (violations must
# strictly decrease), so un-merging was scored on constraints it cannot fix.
def make_validate_facts_node(tools: ToolBox):
    def validate_facts(state: AgentState) -> AgentState:
        """Post-aggregation invariant gate with two LLM-free repair stages.

        Delegates to :func:`~ontocast.stategraph.facts_gate.run_facts_gate`,
        which the single-unit entry path shares. The document path enables the
        un-merge repair: it has many units to re-aggregate against each other.
        """
        if not state.facts_units or len(state.aggregated_facts) == 0:
            if state.status != Status.FAILED:
                state.status = Status.SUCCESS
            return state

        ontology_graph, document_metadata = _facts_aggregation_inputs(state)
        run_facts_gate(
            state,
            ontology_graph,
            tools,
            merge_repair=True,
            document_metadata=document_metadata,
        )
        state.status = Status.SUCCESS
        return state

    return validate_facts


def make_structural_check_node(tools: ToolBox):
    del tools

    def structural_check(state: AgentState) -> AgentState:
        """Run lightweight structural checks over the stitched ontology before the final critic."""
        doc_onto = document_ontology_access(state)
        artifacts = doc_onto.reduced_artifacts()
        if artifacts:
            component_counts: list[int] = []
            for ontology in artifacts:
                if ontology.is_null() or len(ontology.graph) == 0:
                    continue
                ontology_validation = RDFGraphConnectivityValidator(
                    ontology.graph
                ).validate_connectivity()
                component_counts.append(ontology_validation.num_components)
                if not ontology_validation.is_fully_connected:
                    state.improvements_suggestions.append(
                        f"Structural check ({ontology.iri}): ontology has disconnected components; "
                        "prefer linking classes/properties explicitly."
                    )
                if ontology_validation.missing_labels:
                    state.improvements_suggestions.append(
                        f"Structural check ({ontology.iri}): ontology predicates missing labels were detected."
                    )
            if component_counts:
                state.retrieval_metrics[
                    RetrievalMetric.STRUCTURAL_ONTOLOGY_COMPONENTS_MAX
                ] = max(component_counts)
        state.status = Status.SUCCESS
        return state

    return structural_check


def _extract_consistency_queries(graph: RDFGraph, max_terms: int = 8) -> list[str]:
    labels: list[str] = []
    for _, _, obj in graph.triples((None, RDFS.label, None)):
        if isinstance(obj, Literal):
            value = str(obj).strip()
            if value:
                labels.append(value)
    for subject, _, _ in graph:
        if isinstance(subject, URIRef):
            _, local_name = split_namespace_local(str(subject))
            if local_name and local_name not in labels:
                labels.append(local_name.replace("_", " "))
        if len(labels) >= max_terms:
            break
    return labels[:max_terms]


def make_consistency_critic_node(tools: ToolBox):
    def consistency_critic(state: AgentState) -> AgentState:
        """Global consistency critic over candidate ontology atoms using vector re-query."""
        doc_onto = document_ontology_access(state)
        artifacts = [
            ontology
            for ontology in doc_onto.reduced_artifacts()
            if not ontology.is_null() and len(ontology.graph) > 0
        ]
        if (
            state.ontology_context_mode
            != OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
            or tools.vector_store is None
            or not artifacts
        ):
            state.status = Status.SUCCESS
            return state

        merged_graph = RDFGraph()
        for ontology in artifacts:
            merged_graph += ontology.graph
        query_terms = _extract_consistency_queries(merged_graph)
        if not query_terms:
            state.status = Status.SUCCESS
            return state

        allowed_sources = set(all_unit_patch_source_iris(state))
        for ontology in artifacts:
            if ontology.iri:
                allowed_sources.add(ontology.iri)
        # ``search_patch_hits`` returns rank-fused scores, not raw similarities.
        threshold = (
            tools.config.tool_config.vector_store.consistency_critic_min_fused_score
        )
        conflicts: list[str] = []
        for query in query_terms:
            hits = tools.vector_store.search_patch_hits(query=query, top_k=3)
            for hit in hits:
                if (
                    hit.score >= threshold
                    and hit.atom.ontology_iri
                    and hit.atom.ontology_iri not in allowed_sources
                ):
                    conflicts.append(
                        f"Potential cross-ontology conflict for '{query}' with "
                        f"source {hit.atom.ontology_iri} (score={hit.score:.2f})."
                    )
            if len(conflicts) >= 5:
                break

        if conflicts:
            state.improvements_suggestions.extend(conflicts[:5])
            logger.warning(
                "Consistency critic detected %s potential cross-ontology conflicts",
                len(conflicts),
            )
        state.retrieval_metrics[RetrievalMetric.CONSISTENCY_CONFLICTS] = len(conflicts)
        state.status = Status.SUCCESS
        return state

    return consistency_critic
