import asyncio
import logging

from rdflib import RDFS, Literal, URIRef

from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.agent.render_ontology import render_ontology_update
from ontocast.agent.summarize_chunks import should_summarize_unit, summarize_chunk
from ontocast.onto.content_unit import ContentUnit, OutputType, SourceUnit
from ontocast.onto.enum import (
    OntologyAssemblyMode,
    OntologyContextMode,
    Status,
    WorkflowNode,
)
from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.ontology_apply import (
    apply_partitioned_inserts,
    partition_inserts_by_namespace,
)
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState, BudgetTracker
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
)
from ontocast.tool.validate import RDFGraphConnectivityValidator
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def _index_ontologies_by_anchor(artifacts: list[Ontology]) -> dict[str, Ontology]:
    return {ontology.iri: ontology for ontology in artifacts if ontology.iri}


def _empty_unit_snapshot() -> OntologySnapshot:
    return OntologySnapshot.empty(
        title="Pending context resolve",
        description="Placeholder until resolve_unit_ontology_context runs.",
    )


def make_render_ontology_node(tools: ToolBox):
    async def render_ontology_updates(state: AgentState) -> AgentState:
        if not state.content_units:
            state.ontology_units = []
            state.status = Status.SUCCESS
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        async def process_unit(
            unit_index: int,
        ) -> tuple[int, UnitOntologyState, str, list[str], OntologyAssemblyMode]:
            async with semaphore:
                base_state = state.model_copy(deep=True)
                unit_budget = BudgetTracker()
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
                result = await ontology_loop(ontology_state, tools, base_state)
                return (
                    unit_index,
                    result,
                    result.assembly_anchor_iri,
                    list(result.writable_iris or result.ontology_patch_sources),
                    result.assembly_mode_used,
                )

        tasks = [process_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results = await asyncio.gather(*tasks)
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        ontology_units: list[ContentUnit] = []
        fresh_ontologies: list[Ontology] = []
        failed_without_output_count = 0
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
                continue

            content_unit = result.content_unit
            delta_graph = build_ontology_delta_graph(result)
            if len(delta_graph) > 0:
                ontology_units.append(
                    ContentUnit(
                        text=content_unit.text,
                        index=content_unit.index,
                        doc_iri=content_unit.doc_iri,
                        graph=delta_graph,
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

        # Document-level complement bag + namespace apply onto catalog bases.
        merged_delta = RDFGraph()
        for unit in ontology_units:
            for triple in unit.graph:
                merged_delta.add(triple)
            for prefix, namespace in unit.graph.namespaces():
                if prefix:
                    merged_delta.bind(prefix, namespace)

        artifacts: list[Ontology] = list(fresh_ontologies)
        if len(merged_delta) > 0 and all_writable:
            partitioned, unattributed = partition_inserts_by_namespace(
                merged_delta,
                writable_iris=all_writable,
                ontology_manager=tools.ontology_manager,
            )
            state.ontology_reduce_metrics["unattributed_insert_triples"] = unattributed
            applied, apply_metrics = apply_partitioned_inserts(
                partitioned,
                ontology_manager=tools.ontology_manager,
                normalize_units_fn=normalize_ontology_units,
                tools=tools,
            )
            artifacts.extend(applied)
            state.ontology_reduce_metrics.update(apply_metrics)
        elif len(merged_delta) > 0 and not all_writable:
            logger.warning(
                "Ontology map produced %s complement triples but no writable catalog "
                "IRIs; skipping catalog apply",
                len(merged_delta),
            )

        state.ontology_artifacts = artifacts
        state.reduced_ontology_artifacts = list(artifacts)
        state.reduced_ontology_by_anchor = _index_ontologies_by_anchor(artifacts)
        state.ontology_reduce_metrics["reduced_artifact_count"] = len(artifacts)
        state.ontology_units = ontology_units
        state.status = Status.SUCCESS
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
            if len(delta) > 0 and primary.iri:
                partitioned, _unattr = partition_inserts_by_namespace(
                    delta,
                    writable_iris=[primary.iri],
                    ontology_manager=tools.ontology_manager,
                )
                applied, _metrics = apply_partitioned_inserts(
                    partitioned,
                    ontology_manager=tools.ontology_manager,
                    normalize_units_fn=normalize_ontology_units,
                    tools=tools,
                )
                if applied:
                    state.reduced_ontology_artifacts = applied
                    state.reduced_ontology_by_anchor = _index_ontologies_by_anchor(
                        applied
                    )
                    state.ontology_artifacts = applied
                    state.ontology_updates_applied.extend(
                        result.ontology_updates_applied
                    )
                    logger.info(
                        "Ontology consolidation applied %s update operation(s).",
                        len(result.ontology_updates_applied),
                    )
                else:
                    logger.warning(
                        "Ontology consolidation produced deltas but catalog apply "
                        "returned no artifacts."
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
            state.status = Status.SUCCESS
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        async def process_unit(
            unit_index: int,
        ) -> tuple[int, UnitFactsState, str, list[str], OntologyAssemblyMode]:
            async with semaphore:
                base_state = state.model_copy(deep=True)
                unit_budget = BudgetTracker()
                facts_state = UnitFactsState(
                    content_unit=state.content_units[unit_index],
                    ontology_snapshot=_empty_unit_snapshot(),
                    ontology_patch_sources=[],
                    facts_user_instruction=state.facts_user_instruction,
                    budget_tracker=unit_budget,
                    max_visits_per_node=state.max_visits,
                    llm_graph_format=state.llm_graph_format,
                )
                result = await facts_loop(
                    facts_state,
                    tools,
                    base_state,
                )
                return (
                    unit_index,
                    result,
                    result.assembly_anchor_iri,
                    list(result.ontology_patch_sources),
                    result.assembly_mode_used,
                )

        tasks = [process_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results = await asyncio.gather(*tasks)
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        facts_units: list[ContentUnit] = []
        failed_without_output_count = 0
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
            unit_contexts[unit_index] = (anchor_iri, patch_sources, assembly_mode)
            has_output = len(result.content_unit.graph) > 0
            if not has_output:
                failed_without_output_count += 1
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
        state.facts_units = facts_units
        state.status = Status.SUCCESS
        return state

    return render_facts


def make_merge_facts_node(tools: ToolBox):
    def merge_facts(state: AgentState) -> AgentState:
        if not state.facts_units:
            state.aggregated_facts = RDFGraph()
            state.status = Status.SUCCESS
            return state

        ontology_graph = RDFGraph()
        merged_context = build_merged_document_ontology_context(state)
        if merged_context is not None and len(merged_context.snapshot.graph) > 0:
            ontology_graph = merged_context.snapshot.graph
        document_metadata = dict(state.document_metadata)
        if (
            state.source_url
            and "source_url" not in document_metadata
            and "source_uri" not in document_metadata
        ):
            document_metadata["source_url"] = state.source_url
        state.aggregated_facts = tools.aggregator.postprocess_facts_units(
            units=state.facts_units,
            ontology_graph=ontology_graph,
            doc_iri=state.doc_iri,
            document_metadata=document_metadata,
            doc_namespace=state.doc_namespace,
        )
        if len(state.aggregated_facts) == 0:
            logger.warning(
                "Facts aggregation produced an empty graph from "
                f"{len(state.facts_units)} successful unit(s)."
            )
        state.status = Status.SUCCESS
        return state

    return merge_facts


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


def make_summarize_chunks_node(tools: ToolBox):
    async def summarize_chunks(state: AgentState) -> AgentState:
        if not state.content_units or not state.use_summarization:
            state.status = Status.SUCCESS
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        async def process_unit(unit_index: int) -> tuple[int, str | None]:
            async with semaphore:
                unit = state.content_units[unit_index]
                if not should_summarize_unit(unit, state.summarize_sections):
                    return unit_index, None
                try:
                    summary = await summarize_chunk(
                        unit,
                        tools,
                        max_sentences=state.summary_max_sentences,
                    )
                    return unit_index, summary
                except Exception as exc:
                    logger.warning(
                        "Summarization failed for unit %s: %s",
                        unit_index,
                        exc,
                    )
                    return unit_index, None

        tasks = [process_unit(i) for i in range(len(state.content_units))]
        raw_results = await asyncio.gather(*tasks)
        summarized_count = 0
        for unit_index, summary in sorted(raw_results, key=lambda item: item[0]):
            if summary is None:
                continue
            state.content_units[unit_index].summary = summary
            summarized_count += 1

        logger.info(
            "Summarized %s/%s content unit(s)",
            summarized_count,
            len(state.content_units),
        )
        state.set_node_status(WorkflowNode.SUMMARIZE_CHUNKS, Status.SUCCESS)
        state.status = Status.SUCCESS
        return state

    return summarize_chunks
