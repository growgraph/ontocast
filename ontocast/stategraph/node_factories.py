import asyncio
import logging
from collections import defaultdict

from rdflib import DCTERMS, RDFS, Literal, URIRef

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
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState, BudgetTracker
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.atomic import facts_loop, ontology_loop
from ontocast.stategraph.context_resolver import (
    aggregate_anchor_metrics,
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
                    ontology_snapshot=NULL_ONTOLOGY,
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
                    list(result.ontology_patch_sources),
                    result.assembly_mode_used,
                )

        tasks = [process_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results = await asyncio.gather(*tasks)
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        ontology_units: list[ContentUnit] = []
        failed_without_output_count = 0
        salvaged_failed_count = 0
        unit_contexts: dict[int, tuple[str, list[str], OntologyAssemblyMode]] = {}
        anchor_delta_graphs: dict[str, RDFGraph] = defaultdict(RDFGraph)
        for (
            unit_index,
            result,
            anchor_iri,
            patch_sources,
            assembly_mode,
        ) in ordered_results:
            state.budget_tracker.merge_from(result.budget_tracker)
            unit_contexts[unit_index] = (anchor_iri, patch_sources, assembly_mode)
            has_output = bool(result.all_updates) or (
                result.current_ontology.hash != result.ontology_snapshot.hash
            )
            if not has_output:
                failed_without_output_count += 1
                continue

            content_unit = result.content_unit
            delta_graph = build_ontology_delta_graph(result)
            anchor_delta_graphs[anchor_iri] += delta_graph
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

        artifacts: list[Ontology] = []
        for anchor_iri, graph in anchor_delta_graphs.items():
            if len(graph) == 0:
                continue
            artifacts.append(
                Ontology(
                    ontology_id=None,
                    title=f"Anchor artifact {anchor_iri}",
                    description=(
                        "Per-anchor ontology artifact produced from unit-level map updates."
                    ),
                    graph=graph,
                    iri=anchor_iri,
                    current_domain=state.current_domain,
                )
            )

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
            anchor_counts,
        ) = aggregate_anchor_metrics(unit_contexts)
        state.candidate_anchor_iris = sorted(anchor_counts.keys())
        state.retrieval_metrics["ontology_anchor_count"] = len(anchor_counts)
        state.retrieval_metrics["ontology_anchor_units"] = sum(anchor_counts.values())
        state.ontology_artifacts = artifacts
        state.reduced_ontology_artifacts = list(artifacts)
        state.reduced_ontology_by_anchor = _index_ontologies_by_anchor(artifacts)
        state.ontology_reduce_metrics["reduced_artifact_count"] = len(artifacts)
        state.ontology_units = ontology_units
        state.status = Status.SUCCESS
        return state

    return render_ontology_updates


def make_normalize_ontology_node(tools: ToolBox):
    # Design note — two-stage merge responsibility:
    #
    # Stage A (make_render_ontology_node, above):
    #   Each unit's delta graph is aggregated per anchor into a lightweight
    #   Ontology artifact (insert-only, no base ontology applied yet).  This
    #   keeps the map phase stateless and parallelisable.
    #
    # Stage B (this node — normalize):
    #   For single-anchor documents the unit deltas are re-merged from
    #   state.ontology_units and applied *on top of* the pre-existing base
    #   ontology (from OntologyManager), producing a versioned Ontology with
    #   correct parent_hashes lineage.  Provenance/alignment triples are then
    #   stripped to a side artifact.  This stage is intentionally skipped for
    #   multi-anchor documents because each anchor's ontology evolves
    #   independently and cross-anchor collapse is not yet implemented.
    #
    # The apparent "double merge" is therefore not redundant: Stage A produces
    # a delta artifact; Stage B produces the final versioned ontology with base
    # lineage and provenance separation.  Removing either stage would break
    # version tracking or provenance isolation.
    def normalize_ontology_updates(state: AgentState) -> AgentState:
        if not state.ontology_units:
            state.ontology_provenance_artifact = RDFGraph()
            state.status = Status.SUCCESS
            return state

        doc_onto = document_ontology_access(state)
        artifacts = doc_onto.reduced_artifacts()
        # Multi-anchor documents evolve ontologies independently; cross-anchor
        # collapse is not yet implemented, so we skip global normalization and
        # leave each anchor's artifact unchanged.
        if len(artifacts) != 1:
            logger.warning(
                "normalize_ontology_updates: skipping global normalization — "
                "%d anchor artifact(s) found (expected exactly 1). "
                "Per-anchor provenance stripping and base-ontology versioning "
                "will not be applied for this document.",
                len(artifacts),
            )
            state.ontology_provenance_artifact = RDFGraph()
            state.ontology_reduce_provenance = RDFGraph()
            state.ontology_reduce_metrics["normalized_ontology_updates"] = 0
            state.status = Status.SUCCESS
            return state
        primary = artifacts[0]
        ontology, applied_updates, provenance_artifact = normalize_ontology_units(
            units=state.ontology_units,
            tools=tools,
            base_ontology=primary if not primary.is_null() else None,
            require_base=True,
        )
        state.reduced_ontology_artifacts = [ontology]
        state.reduced_ontology_by_anchor = _index_ontologies_by_anchor([ontology])
        state.ontology_artifacts = [ontology]
        state.ontology_updates_applied = applied_updates
        state.ontology_provenance_artifact = provenance_artifact
        state.ontology_reduce_provenance = provenance_artifact
        state.ontology_reduce_metrics["normalized_ontology_updates"] = len(
            applied_updates
        )
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
        consolidation_state = UnitOntologyState(
            content_unit=consolidation_unit,
            ontology_snapshot=artifacts[0],
            ontology_patch_sources=all_unit_patch_source_iris(state),
            ontology_user_instruction=ontology_user_instruction,
            budget_tracker=state.budget_tracker,
            max_visits_per_node=1,
            current_domain=state.current_domain,
            ontology_max_triples=tools.config.server.ontology_max_triples,
            llm_graph_format=state.llm_graph_format,
        )
        result = await render_ontology_update(consolidation_state, atomic_tools)
        if result.status == Status.SUCCESS and not result.current_ontology.is_null():
            state.reduced_ontology_artifacts = [result.current_ontology]
            state.reduced_ontology_by_anchor = _index_ontologies_by_anchor(
                [result.current_ontology]
            )
            state.ontology_artifacts = [result.current_ontology]
            state.ontology_updates_applied.extend(result.ontology_updates_applied)
            logger.info(
                f"Ontology consolidation applied {len(result.ontology_updates_applied)} "
                "update operation(s)."
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
                    ontology_snapshot=NULL_ONTOLOGY,
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
        ) = aggregate_anchor_metrics(unit_contexts)
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
        if (
            merged_context is not None
            and len(merged_context.ontology_snapshot.graph) > 0
        ):
            ontology_graph = merged_context.ontology_snapshot.graph
        state.aggregated_facts = tools.aggregator.postprocess_facts_units(
            units=state.facts_units,
            ontology_graph=ontology_graph,
        )
        if len(state.aggregated_facts) == 0:
            logger.warning(
                "Facts aggregation produced an empty graph from "
                f"{len(state.facts_units)} successful unit(s)."
            )
        if state.source_url and state.doc_namespace:
            state.aggregated_facts.add(
                (URIRef(state.doc_namespace), DCTERMS.source, URIRef(state.source_url))
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
        threshold = tools.config.tool_config.vector_store.consistency_critic_similarity_threshold
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
