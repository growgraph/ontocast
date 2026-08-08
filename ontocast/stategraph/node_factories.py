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
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType, SourceUnit
from ontocast.onto.enum import (
    OntologyAssemblyMode,
    OntologyContextMode,
    Status,
    WorkflowNode,
)
from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.model import FactsValidationFinding, UnitFailure
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
from ontocast.stategraph.helpers import (
    all_unit_patch_source_iris,
    build_document_excerpt,
    build_ontology_delta_graph,
    merge_unit_deltas,
)
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.facts_invariants import (
    collect_shacl_shapes,
    validate_aggregated_facts,
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
                    current_domain=state.current_domain,
                    ontology_max_triples=tools.config.server.ontology_max_triples,
                    llm_graph_format=state.llm_graph_format,
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

        for (
            unit_index,
            result,
            primary_iri,
            writable_iris,
            assembly_mode,
        ) in ordered_results:
            state.budget_tracker.merge_from(result.budget_tracker)
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
            delta = build_ontology_delta_graph(result)
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

        (
            state.unit_anchor_assignment,
            state.unit_patch_sources,
            state.unit_context_mode_used,
            primary_counts,
        ) = aggregate_writable_metrics(unit_contexts)
        state.candidate_anchor_iris = sorted(seen_writable | set(primary_counts))
        state.retrieval_metrics["ontology_writable_count"] = len(seen_writable)
        state.retrieval_metrics["ontology_primary_units"] = sum(primary_counts.values())

        # Document-level insert/delete consensus + namespace apply onto catalog bases.
        merged_delta = merge_unit_deltas(unit_deltas)

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
        state.ontology_reduce_provenance = state.ontology_provenance_artifact
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
            working_graph=snap.graph.copy(),
            assembly_anchor_iri=primary.iri or "",
        )
        result = await render_ontology_update(consolidation_state, atomic_tools)
        if result.status == Status.SUCCESS and result.working_graph_changed():
            delta = build_ontology_delta_graph(result)
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
                    llm_graph_format=state.llm_graph_format,
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
        unit_contexts: dict[int, tuple[str, list[str], OntologyAssemblyMode]] = {}
        for (
            unit_index,
            result,
            anchor_iri,
            patch_sources,
            assembly_mode,
        ) in ordered_results:
            state.budget_tracker.merge_from(result.budget_tracker)
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

        (
            state.unit_anchor_assignment,
            state.unit_patch_sources,
            state.unit_context_mode_used,
            anchor_counts,
        ) = aggregate_writable_metrics(unit_contexts)
        state.candidate_anchor_iris = sorted(anchor_counts.keys())
        state.retrieval_metrics["facts_anchor_count"] = len(anchor_counts)
        state.retrieval_metrics["facts_anchor_units"] = sum(anchor_counts.values())
        all_attempts = [
            attempt
            for attempts in state.facts_loop_telemetry.values()
            for attempt in attempts
        ]
        state.retrieval_metrics["facts_repair_visits_total"] = sum(
            1 for attempt in all_attempts if attempt.kind == "repair"
        )
        state.retrieval_metrics["facts_findings_residual"] = sum(
            attempts[-1].n_deterministic_findings
            for attempts in state.facts_loop_telemetry.values()
            if attempts and attempts[-1].kind == "repair"
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
        state.retrieval_metrics["facts_rejected_merges"] = result.rejected_merge_count
        if len(state.aggregated_facts) == 0:
            logger.warning(
                "Facts aggregation produced an empty graph from "
                f"{len(state.facts_units)} successful unit(s)."
            )
        state.status = Status.SUCCESS
        return state

    return merge_facts


def _vetoes_from_findings(
    findings: list[FactsValidationFinding],
    clusters: dict[str, list[str]],
) -> set[frozenset[URIRef]]:
    """Full-cluster pair vetoes for error findings on merged entities.

    Both the finding's subject and its IRI-valued objects are candidate merge
    victims. DEGENERATE_COREFERENCE reports the *pointing* node as subject and
    the over-merged endpoint in ``values`` (``range1 hasLowerBound v1 ;
    hasUpperBound v1`` -- ``v1`` is the collapsed cluster, ``range1`` usually
    is not merged at all), so a subject-only lookup could never repair it.
    The same holds for the IRI-object branch of SUSPECT_MULTI_VALUE.
    """
    vetoes: set[frozenset[URIRef]] = set()
    for finding in findings:
        candidates = [finding.subject, *finding.values]
        for candidate in candidates:
            members = clusters.get(candidate, [])
            if len(members) < 2:
                continue
            refs = [URIRef(member) for member in members]
            for index, left in enumerate(refs):
                for right in refs[index + 1 :]:
                    vetoes.add(frozenset((left, right)))
    return vetoes


def make_validate_facts_node(tools: ToolBox):
    facts_validation = tools.config.get_tool_config().facts_validation

    def validate_facts(state: AgentState) -> AgentState:
        """Post-aggregation invariant gate with deterministic un-merge repair.

        Error findings whose subject resulted from an identity merge turn the
        offending cluster into pair vetoes; the retained facts units are then
        re-aggregated, up to ``FACTS_MERGE_REPAIR_PASSES`` times. Residual
        findings stay on the state as telemetry.
        """
        if not state.facts_units or len(state.aggregated_facts) == 0:
            if state.status != Status.FAILED:
                state.status = Status.SUCCESS
            return state

        ontology_graph, document_metadata = _facts_aggregation_inputs(state)
        if not len(ontology_graph):
            # Facts were extracted with no catalog vocabulary in front of the
            # model. The per-term non-catalog check cannot see this -- with no
            # context there is nothing to compare against -- so it is reported
            # here, where an empty context is known to be unexpected.
            reason = state.retrieval_metrics.get(
                "empty_snapshot_reason", "no ontology context was assembled"
            )
            logger.warning(
                "Validating facts against an empty ontology context (%s); every "
                "extracted term is outside the catalog.",
                reason,
            )
            state.retrieval_metrics["validated_without_ontology_context"] = True

        shapes_graph = collect_shacl_shapes(ontology_graph, facts_validation.shapes_dir)
        fact_namespaces = [DEFAULT_IRI, str(state.doc_iri), state.doc_namespace or ""]

        def run_validation():
            return validate_aggregated_facts(
                state.aggregated_facts,
                ontology_graph,
                shapes_graph=shapes_graph,
                fact_namespaces=fact_namespaces,
                suspect_multi_value_severity=(
                    facts_validation.suspect_multi_value_severity
                ),
                functional_min_single_support=(
                    facts_validation.functional_min_single_support
                ),
                quantity_fallback_vocabulary=(
                    facts_validation.quantity_fallback_vocabulary
                ),
            )

        report = run_validation()
        vetoes: set[frozenset[URIRef]] = set()
        repair_passes = 0
        rejected_repairs = 0
        while (
            report.error_findings
            and repair_passes < facts_validation.merge_repair_passes
        ):
            new_vetoes = _vetoes_from_findings(
                report.error_findings, state.aggregation_clusters
            )
            if not (new_vetoes - vetoes):
                break
            vetoes |= new_vetoes
            logger.info(
                "Facts validation gate: %d error finding(s), re-aggregating "
                "with %d merge veto pair(s)",
                len(report.error_findings),
                len(vetoes),
            )
            result = tools.aggregator.postprocess_facts_units(
                units=state.facts_units,
                ontology_graph=ontology_graph,
                doc_iri=state.doc_iri,
                document_metadata=document_metadata,
                doc_namespace=state.doc_namespace,
                merge_vetoes=vetoes,
            )
            # Un-merging is destructive: a veto dissolves a whole cluster, so a
            # pass that does not strictly reduce the error count has traded real
            # coreference for nothing and must not be kept.
            previous_graph = state.aggregated_facts
            previous_clusters = state.aggregation_clusters
            previous_errors = len(report.error_findings)
            state.aggregated_facts = result.graph
            state.aggregation_clusters = result.merged_clusters
            candidate_report = run_validation()
            if len(candidate_report.error_findings) >= previous_errors:
                logger.warning(
                    "Facts validation gate: repair pass %d did not reduce errors "
                    "(%d -> %d); reverting to the pre-repair graph",
                    repair_passes + 1,
                    previous_errors,
                    len(candidate_report.error_findings),
                )
                state.aggregated_facts = previous_graph
                state.aggregation_clusters = previous_clusters
                rejected_repairs += 1
                break
            repair_passes += 1
            report = candidate_report

        state.facts_validation_findings = report.findings
        state.retrieval_metrics["facts_validation_findings"] = len(report.findings)
        state.retrieval_metrics["facts_validation_errors"] = len(report.error_findings)
        state.retrieval_metrics["facts_merge_repair_passes"] = repair_passes
        state.retrieval_metrics["facts_merge_vetoes"] = len(vetoes)
        state.retrieval_metrics["facts_merge_repairs_rejected"] = rejected_repairs
        if repair_passes:
            # merge_facts recorded this against the pre-repair aggregation.
            state.retrieval_metrics["facts_rejected_merges"] = len(vetoes)
        if report.error_findings:
            logger.warning(
                "Facts validation gate: %d error finding(s) remain after "
                "%d repair pass(es)",
                len(report.error_findings),
                repair_passes,
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
                state.retrieval_metrics["structural_ontology_components_max"] = max(
                    component_counts
                )
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
        state.retrieval_metrics["consistency_conflicts"] = len(conflicts)
        state.status = Status.SUCCESS
        return state

    return consistency_critic
