import asyncio
import logging
import time
from collections import Counter
from collections.abc import AsyncGenerator, Collection, Coroutine, Sequence
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from rdflib import RDF, RDFS, Literal, URIRef

from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.agent.resolve_coreference import judge_coreference
from ontocast.agent.summarize_chunks import ensure_unit_summary
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.enum import (
    OntologyAssemblyMode,
    OntologyContextMode,
    OntologyContextScope,
    RetrievalMetric,
    Status,
    WorkflowNode,
)
from ontocast.onto.identity import is_owned_ontology
from ontocast.onto.iri_policy import normalize_namespace_iri, split_namespace_local
from ontocast.onto.model import (
    FactsUnitFindingKind,
    OntologyUnitFinding,
    UnitFailure,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.ontology_apply import (
    OntologyDelta,
    apply_partitioned_updates,
    partition_triples_by_namespace,
    refile_fresh_statements,
)
from ontocast.onto.ontology_header import mint_ontology
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import OntologyContextConfigError
from ontocast.onto.state import UNIT_SUM_SUFFIX, AgentState, BudgetTracker
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.atomic import facts_loop, ontology_loop
from ontocast.stategraph.context_resolver import (
    admit_facts_context_terms,
    aggregate_writable_metrics,
    build_merged_document_ontology_context,
    build_unioned_document_ontology_context,
)
from ontocast.stategraph.facts_gate import run_facts_gate, stored_pairs
from ontocast.stategraph.helpers import (
    all_unit_patch_source_iris,
    drop_all_deletes,
    enforce_redeclared_deletes,
    merge_unit_deltas,
    reconcile_fresh_ontologies,
)
from ontocast.stategraph.term_decisions import (
    decide_minted_terms,
)
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.agg.aggregate import AggregationResult
from ontocast.tool.agg.coreference import coreference_candidates, verdict_pairs
from ontocast.tool.llm import RUN_STOPPING_ERRORS
from ontocast.tool.ontology_validation import (
    apply_minted_duplicate_rewrites,
    detect_minted_duplicates,
)
from ontocast.tool.ontology_validation.axiom_hygiene import (
    drop_example_namespace_triples,
    drop_external_subjects,
    drop_non_term_subjects,
    drop_role_crossing_parents,
    drop_terms_minted_outside_home,
    drop_vacuous_upper_terms,
    normalize_datatype_ranges,
)
from ontocast.tool.ontology_validation.reconcile import MintedDuplicate
from ontocast.tool.ontology_validation.shape import (
    CatalogSurfaceIndex,
    minted_class_iris,
)
from ontocast.tool.validate import RDFGraphConnectivityValidator
from ontocast.toolbox import ToolBox
from ontocast.util.loop_lag import loop_lag

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Failures that describe the run rather than the content unit: an ontology
#: context the run cannot resolve, a request the provider refuses outright, and a
#: call the run meter refuses. Each is identical for every sibling unit, so
#: _gather_units re-raises them instead of counting N of them as unit errors.
_DEPLOYMENT_FAULTS = (OntologyContextConfigError, *RUN_STOPPING_ERRORS)


def _insert_only(tools: ToolBox) -> bool:
    """Whether ``ONTOLOGY_UPDATE_MODE=insert_only`` is in force."""
    return (
        tools.config.get_tool_config().ontology_validation.update_mode == "insert_only"
    )


def _count_insert_only_drops(state: AgentState, dropped: int) -> None:
    if dropped:
        state.budget_tracker.incr("ontology/deletes_dropped_insert_only", dropped)


def external_namespaces(tools: ToolBox, current_domain: str) -> list[str]:
    """Namespace stems of the catalog ontologies outside ``current_domain``.

    Each external ontology contributes its IRI and its namespace, without a
    trailing ``/`` or ``#``; prefixes merely bound in its graph do not count.
    """
    stems: set[str] = set()
    for head in tools.ontology_manager.ontologies:
        if head.is_null() or not head.iri:
            continue
        if is_owned_ontology(head.iri, current_domain):
            continue
        stems.add(head.iri.rstrip("/#"))
        stems.add(normalize_namespace_iri(head.iri, context="ontology").rstrip("/#"))
        if head.namespace:
            stems.add(head.namespace.rstrip("/#"))
    return sorted(stems)


def clean_ontology_delta(
    graphs: Sequence[RDFGraph],
    *,
    external: Sequence[str],
    current_domain: str,
    catalog_index: CatalogSurfaceIndex,
    ontology_iris: Collection[str],
    home_namespace: str | None,
) -> dict[str, int]:
    """Apply the deterministic clean-up rules to each graph; the counts removed.

    ``catalog_index`` names the terms every catalog head declares, and which
    are classes and properties; ``ontology_iris`` the heads' (and the home's) IRIs: statements about them
    are not undeclared. Without a home namespace, terms are not checked for
    where they were minted.
    """
    counts = {
        "external_subject_refused": 0,
        "example_namespace_dropped": 0,
        "minted_outside_home_dropped": 0,
        "non_term_subjects_dropped": 0,
        "datatype_ranges_normalized": 0,
        "vacuous_upper_terms_dropped": 0,
        "role_crossing_parents_dropped": 0,
    }
    catalog_terms = catalog_index.terms
    declared = {*catalog_terms, *ontology_iris}
    for graph in graphs:
        counts["external_subject_refused"] += drop_external_subjects(graph, external)
        counts["example_namespace_dropped"] += drop_example_namespace_triples(
            graph, current_domain
        )
        if home_namespace:
            counts["minted_outside_home_dropped"] += drop_terms_minted_outside_home(
                graph,
                home_namespace=home_namespace,
                catalog_terms=catalog_terms,
                current_domain=current_domain,
            )
        counts["non_term_subjects_dropped"] += drop_non_term_subjects(
            graph, declared=declared, facts_namespace=DEFAULT_IRI
        )
        counts["datatype_ranges_normalized"] += normalize_datatype_ranges(graph)
        counts["vacuous_upper_terms_dropped"] += drop_vacuous_upper_terms(graph)
        counts["role_crossing_parents_dropped"] += drop_role_crossing_parents(
            graph,
            catalog_classes=catalog_index.classes,
            catalog_properties=catalog_index.properties,
        )
    return counts


def home_base_overrides(
    state: AgentState, tools: ToolBox, fresh_ontologies: Sequence[Ontology]
) -> dict[str, Ontology]:
    """A base to apply the home ontology's delta onto when it is not stored yet.

    Under vector retrieval a unit renders an update against catalog patches
    even when the document's home ontology is new; without a base its terms
    would have no ontology to land in. A fresh render of the home is that
    base; otherwise an empty root under the home identity is.
    """
    home = state.home_ontology
    if home is None:
        return {}
    for fresh in fresh_ontologies:
        if fresh.iri == home.iri:
            return {home.iri: fresh}
    head = tools.ontology_manager.get_head_by_iri(home.iri)
    if head is not None and not head.is_null():
        return {}
    return {home.iri: mint_ontology(home, RDFGraph())}


@asynccontextmanager
async def _unit_slot(semaphore: asyncio.Semaphore) -> AsyncGenerator[float, None]:
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
    *,
    index_offset: int = 0,
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
    made and then discarded.

    Three classes of failure are deliberately not isolated -- ``_DEPLOYMENT_FAULTS``.
    :class:`~ontocast.onto.retrieval_capabilities.OntologyContextConfigError`,
    :class:`~ontocast.tool.llm.LLMConfigurationError` and
    :class:`~ontocast.onto.run_meter.BudgetExceededError` describe the run,
    not the unit, so they are identical for every sibling and
    isolating them turns a configuration fault into N unit failures and a
    successful, empty run. They are re-raised once the gather has drained, which
    orphans nothing -- every sibling has already been awaited. Note that
    subclassing ``BaseException`` would not achieve this: ``return_exceptions``
    captures those as values too.

    Args:
        node: Fan-out node the tasks belong to; namespaces the metric keys.
        state: Document state whose tracker the stage metrics land on.
        tasks: Per-unit coroutines to run concurrently.
        index_offset: Index of ``tasks[0]`` among the document's units. A
            warm-up splits the fan-out into two gathers, and without this the
            second batch would name its failures by position within the batch --
            reporting unit 0 for a unit that is not unit 0.

    Returns:
        tuple: Successful results in submission order, and the number of units
        that raised.
    """
    async with loop_lag() as lag:
        raw = await asyncio.gather(*tasks, return_exceptions=True)
    state.budget_tracker.add_duration(f"{node}/loop_lag_total", lag.total)
    state.budget_tracker.add_duration(f"{node}/loop_lag_max", lag.peak)

    for item in raw:
        if isinstance(item, _DEPLOYMENT_FAULTS):
            raise item

    results: list[T] = []
    failures = 0
    for index, item in enumerate(raw):
        if isinstance(item, BaseException):
            failures += 1
            state.budget_tracker.incr(f"{node}/unit_errors")
            logger.exception(
                "Unit %s raised during %s: %s",
                index + index_offset,
                node,
                item,
                exc_info=item,
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
                    home_ontology=state.home_ontology,
                    ontology_plan=state.ontology_plan,
                    ontology_max_triples=tools.config.server.ontology_max_triples,
                    llm_graph_format=state.llm_graph_format,
                    llm_output_layout=tools.config.server.llm_output_layout,
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
        # Deletes of units whose context was a retrieved subset: kept only
        # where the document redeclares their subject.
        deletes_on_partial_view = RDFGraph()
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
        # For consolidation: the findings left on each unit, and which units
        # minted each term (their text is the term's evidence).
        residual_findings: list[OntologyUnitFinding] = []
        term_units: dict[str, set[int]] = {}
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
            residual_findings.extend(result.deterministic_findings)
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
            unit_graphs = [delta.inserts]
            if result.fresh_ontology is not None:
                unit_graphs.append(result.fresh_ontology.graph)
            for graph in unit_graphs:
                for subject in set(graph.subjects()):
                    if isinstance(subject, URIRef):
                        term_units.setdefault(str(subject), set()).add(unit_index)
            if not delta.is_empty():
                unit_deltas.append(delta)
            if result.ontology_snapshot.partial:
                deletes_on_partial_view += delta.deletes
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
        state.unit_context_sources = {
            **state.unit_context_sources,
            "ontology": dict(state.unit_patch_sources),
        }
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
        # it, whether the critic ran (ONTOLOGY_CRITIC_PASSES) and what it
        # accepted are unrecoverable from a run's artifacts.
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
        state.retrieval_metrics[RetrievalMetric.ONTOLOGY_CRITIC_UNITS_SKIPPED] = sum(
            1
            for attempts in state.ontology_loop_telemetry.values()
            if any(attempt.kind == "critic_skipped" for attempt in attempts)
        )
        ontology_patch_attempts = [
            attempt
            for attempts in state.ontology_loop_telemetry.values()
            for attempt in attempts
            if attempt.kind == "critic_patch"
        ]
        state.retrieval_metrics[RetrievalMetric.ONTOLOGY_CRITIC_FIXES_APPLIED] = sum(
            attempt.n_fixes_applied for attempt in ontology_patch_attempts
        )
        state.retrieval_metrics[RetrievalMetric.ONTOLOGY_CRITIC_FIXES_NOOP] = sum(
            attempt.n_fixes_noop for attempt in ontology_patch_attempts
        )
        state.retrieval_metrics[RetrievalMetric.ONTOLOGY_CRITIC_PATCHES_ROLLED_BACK] = (
            sum(1 for attempt in ontology_patch_attempts if attempt.patch_rolled_back)
        )

        # Document-level insert/delete consensus + namespace apply onto catalog bases.
        merged_delta = merge_unit_deltas(unit_deltas)
        if _insert_only(tools):
            _count_insert_only_drops(state, drop_all_deletes(merged_delta))

        # A unit that saw its ontologies whole judged its deletes on full
        # evidence; only deletes judged on a partial view need a redeclaration.
        state.ontology_reduce_metrics["deletes_dropped_unredeclared"] = (
            enforce_redeclared_deletes(merged_delta, deletes_on_partial_view)
        )

        # Clean-up without a model, before anything reviews the delta:
        # consolidation and duplicate detection should not spend a decision on
        # a triple that is never written.
        home = state.home_ontology
        hygiene = clean_ontology_delta(
            [
                merged_delta.inserts,
                merged_delta.deletes,
                *(o.graph for o in fresh_ontologies),
            ],
            external=external_namespaces(tools, state.current_domain),
            current_domain=state.current_domain,
            catalog_index=tools.ontology_manager.catalog_surface_index(),
            ontology_iris=[
                *(head.iri for head in tools.ontology_manager.ontologies if head.iri),
                *([home.iri] if home is not None else []),
            ],
            home_namespace=home.namespace if home is not None else None,
        )
        state.ontology_reduce_metrics.update(
            {key: value for key, value in hygiene.items() if value}
        )

        # Minted-duplicate reconciliation against the whole catalog, not the
        # units' snapshots: under vector retrieval a term the retrieval failed
        # to surface gets re-minted under a fresh IRI. 'detect' (default) only
        # measures; 'rewrite' substitutes once the matches are shown to be
        # true duplicates.
        reconcile_mode = (
            tools.config.get_tool_config().ontology_validation.reconcile_minted_terms
        )
        duplicates: list[MintedDuplicate] = []
        if reconcile_mode != "off" and len(merged_delta.inserts) > 0:
            duplicates = detect_minted_duplicates(
                merged_delta.inserts, tools.ontology_manager.catalog_surface_index()
            )
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

        # The document-level decision on every minted term: candidates the
        # plan did not name are held back, consolidation (when on) decides
        # each term, and the decisions are compiled into both carriers.
        await decide_minted_terms(
            state,
            tools,
            merged_inserts=merged_delta.inserts,
            fresh_ontologies=fresh_ontologies,
            findings=residual_findings,
            term_units=term_units,
            duplicates=duplicates,
        )
        fresh_ontologies = [o for o in fresh_ontologies if len(o.graph) > 0]

        base_overrides = home_base_overrides(state, tools, fresh_ontologies)
        if state.home_ontology is not None and state.home_ontology.iri not in (
            seen_writable
        ):
            seen_writable.add(state.home_ontology.iri)
            all_writable.append(state.home_ontology.iri)

        # A fresh render is filed by subject like an update: what it states
        # about another ontology's terms joins the delta for that ontology.
        refiled = unowned = 0
        for fresh in fresh_ontologies:
            moved, dropped = refile_fresh_statements(
                fresh,
                writable_iris=all_writable,
                ontology_manager=tools.ontology_manager,
                current_domain=state.current_domain,
                base_overrides=base_overrides,
            )
            for triple in moved:
                merged_delta.inserts.add(triple)
            refiled += len(moved)
            unowned += dropped
        if refiled:
            state.ontology_reduce_metrics["fresh_triples_refiled"] = refiled
        if unowned:
            state.ontology_reduce_metrics["fresh_triples_unattributed"] = unowned

        catalog_index = tools.ontology_manager.catalog_surface_index()
        catalog_subjects = catalog_index.terms
        state.ontology_minted_classes = sorted(
            minted_class_iris(
                merged_delta.inserts, *(o.graph for o in fresh_ontologies)
            )
            - catalog_subjects
        )
        state.ontology_reduce_metrics["minted_classes"] = len(
            state.ontology_minted_classes
        )

        # A fresh render whose every statement was filed elsewhere is not an
        # ontology the document wrote.
        artifacts: list[Ontology] = [
            o for o in fresh_ontologies if set(o.graph.subjects()) - {URIRef(o.iri)}
        ]
        if not merged_delta.is_empty() and all_writable:
            partitioned_inserts, unattributed = partition_triples_by_namespace(
                merged_delta.inserts,
                writable_iris=all_writable,
                ontology_manager=tools.ontology_manager,
                current_domain=state.current_domain,
                base_overrides=base_overrides,
            )
            state.ontology_reduce_metrics["unattributed_insert_triples"] = unattributed
            partitioned_deletes, unattributed_deletes = partition_triples_by_namespace(
                merged_delta.deletes,
                writable_iris=all_writable,
                ontology_manager=tools.ontology_manager,
                current_domain=state.current_domain,
                base_overrides=base_overrides,
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
                base_overrides=base_overrides,
            )
            # A fresh home the delta was applied onto is superseded by the result.
            applied_iris = {artifact.iri for artifact in applied}
            artifacts = [a for a in artifacts if a.iri not in applied_iris]
            artifacts.extend(applied)
            for iri, updates in applied_updates.items():
                state.ontology_updates_by_iri.setdefault(iri, []).extend(updates)
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
        if (
            merged_context is None
            and tools.config.server.ontology_context_scope
            == OntologyContextScope.DOCUMENT
        ):
            # No ontology stage ran, so there is nothing to merge -- but the
            # deployment has asked for one chapter per document rather than one
            # per unit. Resolve every unit's context up front and union it, so
            # the fan-out shares a prompt prefix instead of N distinct ones.
            merged_context = await build_unioned_document_ontology_context(
                UnitLoopContext.from_agent_state(state), tools, state.content_units
            )
        if merged_context is not None:
            merged_context = admit_facts_context_terms(
                merged_context,
                state.ontology_plan,
                tools,
                budget_tracker=state.budget_tracker,
            )
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
                conformance_chapter, contract_terms, selection_pending = (
                    tools.shapes_prompt_contract()
                )
                facts_state = UnitFactsState(
                    content_unit=state.content_units[unit_index],
                    ontology_snapshot=_empty_unit_snapshot(),
                    ontology_patch_sources=[],
                    facts_user_instruction=state.facts_user_instruction,
                    conformance_chapter=conformance_chapter,
                    shapes_contract_terms=contract_terms,
                    conformance_selection_pending=selection_pending,
                    budget_tracker=unit_budget,
                    max_visits_per_node=state.max_visits,
                    llm_graph_format=state.llm_graph_format,
                    llm_output_layout=tools.config.server.llm_output_layout,
                    ontology_context_max_triples=tools.config.server.ontology_context_max_triples,
                    ontology_chapter_format=tools.config.server.ontology_chapter_format,
                    ontology_text_caps=tools.config.server.ontology_text_caps,
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

        # A provider's prefix cache is populated by a request that has already
        # completed, so a fan-out that issues every call at once has all of them
        # miss a prefix they all share. Running the first unit alone turns the
        # rest into hits -- but only where the chapter is shared, which is why
        # this is a separate knob rather than an implication of the scope.
        warmup = min(
            tools.config.server.fanout_warmup_units,
            len(state.content_units),
        )
        raw_results: list = []
        unit_errors = 0
        if warmup:
            warm_results, warm_errors = await _gather_units(
                WorkflowNode.RENDER_FACTS,
                state,
                [process_unit(i) for i in range(warmup)],
            )
            raw_results.extend(warm_results)
            unit_errors += warm_errors
        rest = [process_unit(i) for i in range(warmup, len(state.content_units))]
        if rest:
            more_results, more_errors = await _gather_units(
                WorkflowNode.RENDER_FACTS, state, rest, index_offset=warmup
            )
            raw_results.extend(more_results)
            unit_errors += more_errors
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        facts_units: list[ContentUnit] = []
        failed_without_output_count = unit_errors
        salvaged_failed_count = 0
        # Accumulated over *every* unit, whether or not it ran a repair render,
        # so the residual has "units" as its denominator.
        findings_residual = 0
        mandatory_residual = 0
        mandatory_by_kind: Counter[str] = Counter()
        subject_gaps_rendered = 0
        subject_gaps_left = 0
        critic_fixes_applied = 0
        critic_fixes_residual = 0
        critic_fixes_noop = 0
        critic_fixes_rolled_back = 0
        critic_fixes_junk_refused = 0
        critic_fixes_unresolved_prefix = 0
        critic_units_unreviewed = 0
        critic_units_skipped = 0
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
            mandatory_by_kind.update(
                str(finding.kind)
                for finding in result.deterministic_findings
                if finding.mandatory
            )
            subject_gaps_rendered += result.statements_without_subject_rendered
            subject_gaps_left += sum(
                1
                for finding in result.deterministic_findings
                if finding.kind == FactsUnitFindingKind.STATEMENT_WITHOUT_SUBJECT
            )
            critic_fixes_applied += result.critic_fixes_applied
            critic_fixes_residual += result.critic_fixes_residual
            critic_fixes_noop += result.critic_fixes_noop
            critic_fixes_rolled_back += result.critic_fixes_rolled_back
            critic_fixes_junk_refused += result.critic_fixes_junk_refused
            critic_fixes_unresolved_prefix += result.critic_fixes_unresolved_prefix
            # A unit whose critic call failed used to leave the loop SUCCESS,
            # indistinguishable from one the critic accepted; a unit the loop
            # never sent to the critic looked the same. Both are counted.
            critic_units_unreviewed += result.critic_outcome == "unavailable"
            critic_units_skipped += result.critic_outcome == "skipped"
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

        if merged_context is None and (
            tools.config.get_tool_config().facts_validation.context_from_units
        ):
            # No ontology stage ran, so build_merged_document_ontology_context
            # had nothing to merge. Without this both MERGE_FACTS and
            # VALIDATE_FACTS run against an empty ontology graph although every
            # unit rendered against a real one.
            unit_ontology_context = _union_unit_ontology_context(ordered_results)
            if len(unit_ontology_context) > 0:
                state.facts_ontology_context = unit_ontology_context
                state.retrieval_metrics[RetrievalMetric.ONTOLOGY_SNAPSHOT_TRIPLES] = (
                    len(unit_ontology_context)
                )

        _, state.unit_patch_sources, _, anchor_counts = aggregate_writable_metrics(
            unit_contexts
        )
        state.unit_context_sources = {
            **state.unit_context_sources,
            "facts": dict(state.unit_patch_sources),
        }
        state.retrieval_metrics[RetrievalMetric.FACTS_ANCHOR_COUNT] = len(anchor_counts)
        state.retrieval_metrics[RetrievalMetric.FACTS_ANCHOR_UNITS] = sum(
            anchor_counts.values()
        )
        all_attempts = [
            attempt
            for attempts in state.facts_loop_telemetry.values()
            for attempt in attempts
        ]
        # Patch passes undone for leaving the unit worse -- deleting without
        # writing, shrinking without resolving, or manufacturing new mandatory
        # findings. Non-zero means the critique is provoking data-destroying
        # responses, the failure mode that has cost runs a large share of their
        # value nodes while logging nothing a run could be judged by.
        state.retrieval_metrics[RetrievalMetric.FACTS_CRITIC_PATCHES_ROLLED_BACK] = sum(
            1 for attempt in all_attempts if attempt.patch_rolled_back
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
        state.retrieval_metrics[RetrievalMetric.FACTS_MANDATORY_RESIDUAL_BY_KIND] = (
            dict(sorted(mandatory_by_kind.items()))
        )
        state.retrieval_metrics[
            RetrievalMetric.FACTS_STATEMENTS_WITHOUT_SUBJECT_RENDERED
        ] = subject_gaps_rendered
        state.retrieval_metrics[RetrievalMetric.FACTS_STATEMENTS_WITHOUT_SUBJECT] = (
            subject_gaps_left
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
        # What the critique actually produced, split by outcome. The
        # loop used to discard every fix on an accepted render, so the honest
        # value of these two together was once "far less than the call count
        # suggests" -- and nothing recorded it.
        state.retrieval_metrics[RetrievalMetric.FACTS_CRITIC_FIXES_APPLIED] = (
            critic_fixes_applied
        )
        state.retrieval_metrics[RetrievalMetric.FACTS_CRITIC_FIXES_RESIDUAL] = (
            critic_fixes_residual
        )
        # Fixes that removed exactly what they re-added. A critique made mostly
        # of these is a critic producing motion rather than corrections, which
        # nothing could see before: they used to be counted as fixes that
        # landed.
        state.retrieval_metrics[RetrievalMetric.FACTS_CRITIC_FIXES_NOOP] = (
            critic_fixes_noop
        )
        state.retrieval_metrics[RetrievalMetric.FACTS_CRITIC_FIXES_ROLLED_BACK] = (
            critic_fixes_rolled_back
        )
        state.retrieval_metrics[RetrievalMetric.FACTS_CRITIC_FIXES_JUNK_REFUSED] = (
            critic_fixes_junk_refused
        )
        state.retrieval_metrics[
            RetrievalMetric.FACTS_CRITIC_FIXES_UNRESOLVED_PREFIX
        ] = critic_fixes_unresolved_prefix
        state.retrieval_metrics[RetrievalMetric.FACTS_CRITIC_UNITS_UNREVIEWED] = (
            critic_units_unreviewed
        )
        state.retrieval_metrics[RetrievalMetric.FACTS_CRITIC_UNITS_SKIPPED] = (
            critic_units_skipped
        )
        # The insert-only completion pass, read off the attempt log like the
        # critic's own ledger: calls made, inserts that stayed, and missed
        # measurements the inventory stopped listing.
        completion_attempts = [a for a in all_attempts if a.kind == "completion"]
        state.retrieval_metrics[RetrievalMetric.FACTS_COMPLETION_CALLS] = len(
            completion_attempts
        )
        state.retrieval_metrics[RetrievalMetric.FACTS_COMPLETION_TRIPLES_INSERTED] = (
            sum(a.n_triples_inserted for a in completion_attempts)
        )
        state.retrieval_metrics[
            RetrievalMetric.FACTS_COMPLETION_MEASUREMENTS_RECOVERED
        ] = sum(a.n_measurements_recovered for a in completion_attempts)
        state.facts_units = facts_units
        state.status = _map_stage_status(
            failed_without_output_count, len(state.content_units)
        )
        return state

    return render_facts


def _union_unit_ontology_context(
    results: Sequence[tuple[int, UnitFactsState, str, list[str], OntologyAssemblyMode]],
) -> RDFGraph:
    """Union the ontology snapshots the facts units actually resolved.

    Only used when no ontology stage ran: there are then no reduced artifacts
    to merge, so the document-level context would be empty even though every
    unit rendered against a real one.

    Snapshots are deduplicated by their contributing catalog IRIs. In
    single-select mode every unit resolves the same catalog ontology into its
    own snapshot object, so unioning them unfiltered would pay a full rdflib
    merge per unit for a graph already present.

    Args:
        results: The fan-out's per-unit results, in unit order.

    Returns:
        The union of the distinct unit snapshots; empty when none carried one.
    """
    union = RDFGraph()
    seen_sources: set[tuple[str, ...]] = set()
    seen_objects: set[int] = set()
    for _, result, _, _, _ in results:
        snapshot = result.ontology_snapshot
        if snapshot is None or len(snapshot.graph) == 0:
            continue
        key = tuple(sorted(snapshot.source_iris))
        if key:
            if key in seen_sources:
                continue
            seen_sources.add(key)
        elif id(snapshot) in seen_objects:
            continue
        seen_objects.add(id(snapshot))
        union += snapshot.graph
    if len(union) > 0:
        union.sanitize_prefixes_namespaces()
    return union


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
        _store_aggregation(state, result)
        aggregator = tools.aggregator
        if aggregator.llm_coreference and len(state.facts_units) > 1:
            state.coreference_candidates = coreference_candidates(
                unconfirmed_merges=result.unconfirmed_merges,
                lexical_near_misses=result.lexical_near_misses,
                units=state.facts_units,
                final_uri={
                    str(entity): str(decision.final_uri)
                    for entity, decision in result.decisions.items()
                    if decision.final_uri is not None
                },
                max_pairs=aggregator.llm_coreference_max_pairs,
            )
        if len(state.aggregated_facts) == 0:
            logger.warning(
                "Facts aggregation produced an empty graph from "
                f"{len(state.facts_units)} successful unit(s)."
            )
        state.status = Status.SUCCESS
        return state

    return merge_facts


def _store_aggregation(state: AgentState, result: AggregationResult) -> None:
    """Make ``result`` the aggregation the later nodes read."""
    state.aggregated_facts = result.graph
    state.aggregation_clusters = result.merged_clusters
    state.aggregation_key_clusters = result.key_supported_clusters
    state.aggregation_cross_unit_pairs = result.cross_unit_object_pairs
    state.retrieval_metrics[RetrievalMetric.FACTS_REJECTED_MERGES] = (
        result.rejected_merge_count
    )
    state.retrieval_metrics[RetrievalMetric.FACTS_MERGE_REJECTIONS_BY_REASON] = dict(
        result.rejected_by_reason
    )


def make_resolve_coreference_node(tools: ToolBox):
    async def resolve_coreference(state: AgentState) -> AgentState:
        """Put the pairs aggregation could not settle to the model, then re-aggregate.

        ``same`` verdicts merge like a shared key and ``different`` ones veto;
        the aggregator's guards still apply. The verdicts are kept on the state
        so the facts gate's own re-aggregation honours them.
        """
        candidates = state.coreference_candidates
        if not candidates:
            return state
        verdicts = await judge_coreference(state, tools, candidates)
        metrics = state.retrieval_metrics
        metrics[RetrievalMetric.FACTS_COREFERENCE_PAIRS] = len(candidates)
        metrics[RetrievalMetric.FACTS_COREFERENCE_SAME] = sum(
            1 for v in verdicts.values() if v == "same"
        )
        metrics[RetrievalMetric.FACTS_COREFERENCE_DIFFERENT] = sum(
            1 for v in verdicts.values() if v == "different"
        )
        vetoes, confirmations = verdict_pairs(candidates, verdicts)
        if not vetoes and not confirmations:
            return state
        state.coreference_vetoes = stored_pairs(vetoes)
        state.coreference_confirmations = stored_pairs(confirmations)
        ontology_graph, document_metadata = _facts_aggregation_inputs(state)
        result = tools.aggregator.postprocess_facts_units(
            units=state.facts_units,
            ontology_graph=ontology_graph,
            doc_iri=state.doc_iri,
            document_metadata=document_metadata,
            doc_namespace=state.doc_namespace,
            merge_vetoes=vetoes,
            confirmed_pairs=confirmations,
        )
        _store_aggregation(state, result)
        return state

    return resolve_coreference


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
            record_unused_minted_classes(state)
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
        record_unused_minted_classes(state)
        state.status = Status.SUCCESS
        return state

    return validate_facts


def record_unused_minted_classes(state: AgentState) -> None:
    """Count the classes this document minted that none of its facts types.

    Recorded only when the run rendered both lanes, so a class can have been
    instantiated. A minted class no fact of its own document uses is the
    clearest sign the class names that document rather than its domain.
    """
    if not (state.render_ontology and state.render_facts):
        return
    if not state.ontology_minted_classes:
        return
    used = {
        str(obj)
        for obj in state.aggregated_facts.objects(None, RDF.type)
        if isinstance(obj, URIRef)
    }
    unused = [iri for iri in state.ontology_minted_classes if iri not in used]
    state.ontology_reduce_metrics["unused_minted_classes"] = len(unused)
    state.ontology_reduce_metrics["unused_minted_class_iris"] = unused


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
