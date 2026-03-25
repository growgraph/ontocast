import asyncio
import logging

from rdflib import DCTERMS, RDFS, Literal, URIRef

from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.agent.render_ontology import render_ontology_update
from ontocast.onto.content_unit import ContentUnit, OutputType, SourceUnit
from ontocast.onto.enum import OntologyContextMode, Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.atomic import facts_loop, ontology_loop
from ontocast.stategraph.helpers import (
    build_document_excerpt,
    build_ontology_delta_graph,
)
from ontocast.tool.validate import RDFGraphConnectivityValidator
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def make_bootstrap_ontology_node(tools: ToolBox):
    atomic_tools = tools.get_atomic_tools()

    async def bootstrap_ontology(state: AgentState) -> AgentState:
        """Create one seed ontology for null-selection flow."""
        if not state.render_ontology or not state.current_ontology.is_null():
            state.status = Status.SUCCESS
            return state
        if not state.content_units:
            state.status = Status.SUCCESS
            return state

        excerpt = build_document_excerpt(state).strip()
        if not excerpt:
            logger.warning(
                "Skipping ontology bootstrap: no usable excerpt was produced from content units."
            )
            state.status = Status.SUCCESS
            return state

        bootstrap_unit = SourceUnit(
            text=excerpt,
            index=0,
            doc_iri=URIRef(state.doc_iri),
            type=OutputType.ONTOLOGIES,
        )
        bootstrap_state = UnitOntologyState(
            content_unit=bootstrap_unit,
            ontology_snapshot=Ontology(),
            ontology_patch_sources=state.ontology_patch_sources,
            ontology_user_instruction=state.ontology_user_instruction,
            budget_tracker=state.budget_tracker,
            max_visits_per_node=tools.config.server.max_visits_per_node,
            current_domain=state.current_domain,
            ontology_max_triples=tools.config.server.ontology_max_triples,
        )
        result = await ontology_loop(bootstrap_state, atomic_tools)
        if result.status == Status.SUCCESS and not result.current_ontology.is_null():
            state.current_ontology = result.current_ontology
            logger.info(
                f"Bootstrapped ontology anchor: {state.current_ontology.iri} "
                f"({len(state.current_ontology.graph)} triples)"
            )
        else:
            logger.warning(
                "Ontology bootstrap did not yield a usable seed ontology; "
                "continuing with fallback normalization behavior."
            )
        state.status = Status.SUCCESS
        return state

    return bootstrap_ontology


def make_render_ontology_node(tools: ToolBox):
    atomic_tools = tools.get_atomic_tools()

    async def render_ontology_updates(state: AgentState) -> AgentState:
        if not state.content_units:
            state.ontology_units = []
            state.status = Status.SUCCESS
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        async def process_unit(unit_index: int) -> tuple[int, UnitOntologyState]:
            async with semaphore:
                base_state = state.model_copy(deep=True)
                ontology_state = UnitOntologyState(
                    content_unit=state.content_units[unit_index],
                    ontology_snapshot=state.current_ontology,
                    ontology_patch_sources=state.ontology_patch_sources,
                    ontology_user_instruction=state.ontology_user_instruction,
                    budget_tracker=base_state.budget_tracker,
                    max_visits_per_node=tools.config.server.max_visits_per_node,
                    current_domain=state.current_domain,
                    ontology_max_triples=tools.config.server.ontology_max_triples,
                )
                result = await ontology_loop(ontology_state, atomic_tools)
                return unit_index, result

        tasks = [process_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results = await asyncio.gather(*tasks)
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        ontology_units: list[ContentUnit] = []
        failed_without_output_count = 0
        salvaged_failed_count = 0
        for _, result in ordered_results:
            has_output = bool(result.all_updates) or (
                result.current_ontology.hash != result.ontology_snapshot.hash
            )
            if not has_output:
                failed_without_output_count += 1
                continue

            content_unit = result.content_unit
            delta_graph = build_ontology_delta_graph(result)
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

        state.ontology_units = ontology_units
        state.status = Status.SUCCESS
        return state

    return render_ontology_updates


def make_normalize_ontology_node(tools: ToolBox):
    def normalize_ontology_updates(state: AgentState) -> AgentState:
        if not state.ontology_units:
            state.ontology_provenance_artifact = RDFGraph()
            state.status = Status.SUCCESS
            return state

        ontology, applied_updates, provenance_artifact = normalize_ontology_units(
            units=state.ontology_units,
            tools=tools,
            base_ontology=state.current_ontology
            if not state.current_ontology.is_null()
            else None,
            require_base=True,
        )
        state.current_ontology = ontology
        state.ontology_updates_applied = applied_updates
        state.ontology_provenance_artifact = provenance_artifact
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
        if not state.render_ontology or state.current_ontology.is_null():
            logger.info(
                "Skipping ontology consolidation: no rendered ontology snapshot available"
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
            ontology_snapshot=state.current_ontology,
            ontology_patch_sources=state.ontology_patch_sources,
            ontology_user_instruction=ontology_user_instruction,
            budget_tracker=state.budget_tracker,
            max_visits_per_node=1,
            current_domain=state.current_domain,
            ontology_max_triples=tools.config.server.ontology_max_triples,
        )
        result = await render_ontology_update(consolidation_state, atomic_tools)
        if result.status == Status.SUCCESS and not result.current_ontology.is_null():
            state.current_ontology = result.current_ontology
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
    atomic_tools = tools.get_atomic_tools()

    async def render_facts(state: AgentState) -> AgentState:
        if not state.content_units:
            state.facts_units = []
            state.status = Status.SUCCESS
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        async def process_unit(unit_index: int) -> tuple[int, UnitFactsState]:
            async with semaphore:
                base_state = state.model_copy(deep=True)
                facts_state = UnitFactsState(
                    content_unit=state.content_units[unit_index],
                    ontology_snapshot=state.current_ontology,
                    facts_user_instruction=state.facts_user_instruction,
                    budget_tracker=base_state.budget_tracker,
                    max_visits_per_node=tools.config.server.max_visits_per_node,
                )
                result = await facts_loop(facts_state, atomic_tools)
                return unit_index, result

        tasks = [process_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results = await asyncio.gather(*tasks)
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        facts_units: list[ContentUnit] = []
        failed_without_output_count = 0
        salvaged_failed_count = 0
        for _, result in ordered_results:
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

        for unit in state.facts_units:
            unit.sanitize()
        state.aggregated_facts = tools.aggregator.aggregate_graphs(
            units=state.facts_units,
            ontology_graph=state.current_ontology.graph
            if not state.current_ontology.is_null()
            else None,
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
        if (
            not state.current_ontology.is_null()
            and len(state.current_ontology.graph) > 0
        ):
            ontology_validation = RDFGraphConnectivityValidator(
                state.current_ontology.graph
            ).validate_connectivity()
            state.retrieval_metrics["structural_ontology_components"] = (
                ontology_validation.num_components
            )
            if not ontology_validation.is_fully_connected:
                state.improvements_suggestions.append(
                    "Structural check: ontology has disconnected components; "
                    "prefer linking classes/properties explicitly."
                )
            if ontology_validation.missing_labels:
                state.improvements_suggestions.append(
                    "Structural check: ontology predicates missing labels were detected."
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
            local_name = str(subject).rstrip("/").split("/")[-1].split("#")[-1]
            if local_name and local_name not in labels:
                labels.append(local_name.replace("_", " "))
        if len(labels) >= max_terms:
            break
    return labels[:max_terms]


def make_consistency_critic_node(tools: ToolBox):
    def consistency_critic(state: AgentState) -> AgentState:
        """Global consistency critic over candidate ontology atoms using vector re-query."""
        if (
            state.ontology_context_mode != OntologyContextMode.RETRIEVED_INDUCED_GRAPH
            or tools.vector_store is None
            or state.current_ontology.is_null()
            or len(state.current_ontology.graph) == 0
        ):
            state.status = Status.SUCCESS
            return state

        query_terms = _extract_consistency_queries(state.current_ontology.graph)
        if not query_terms:
            state.status = Status.SUCCESS
            return state

        allowed_sources = set(state.ontology_patch_sources)
        if state.current_ontology.iri:
            allowed_sources.add(state.current_ontology.iri)
        threshold = (
            tools.config.tool_config.qdrant.consistency_critic_similarity_threshold
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
