import asyncio
import logging

from rdflib import DCTERMS, URIRef

from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.agent.render_ontology import render_ontology_update
from ontocast.onto.content_unit import ContentUnit, OutputType, SourceUnit
from ontocast.onto.enum import Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.atomic import facts_loop, ontology_loop
from ontocast.stategraph.helpers import (
    build_document_excerpt,
    build_ontology_delta_graph,
)
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
            state.status = Status.SUCCESS
            return state

        ontology, applied_updates = normalize_ontology_units(
            units=state.ontology_units,
            tools=tools,
            base_ontology=state.current_ontology
            if not state.current_ontology.is_null()
            else None,
            require_base=True,
        )
        state.current_ontology = ontology
        state.ontology_updates_applied = applied_updates
        state.status = Status.SUCCESS
        return state

    return normalize_ontology_updates


def make_consolidate_ontology_node(tools: ToolBox):
    atomic_tools = tools.get_atomic_tools()

    async def consolidate_ontology(state: AgentState) -> AgentState:
        """Optional post-normalization ontology consolidation pass."""
        if not tools.config.server.enable_ontology_consolidation:
            state.status = Status.SUCCESS
            return state
        if not state.render_ontology or state.current_ontology.is_null():
            state.status = Status.SUCCESS
            return state

        excerpt = build_document_excerpt(state).strip()
        if not excerpt:
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
            state.parallel_facts_units = []
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

        state.parallel_facts_units = facts_units
        state.status = Status.SUCCESS
        return state

    return render_facts


def make_merge_facts_node(tools: ToolBox):
    def merge_facts(state: AgentState) -> AgentState:
        if not state.parallel_facts_units:
            state.aggregated_facts = RDFGraph()
            state.status = Status.SUCCESS
            return state

        for unit in state.parallel_facts_units:
            unit.sanitize()
        state.aggregated_facts = tools.aggregator.aggregate_graphs(
            units=state.parallel_facts_units,
            ontology_graph=state.current_ontology.graph
            if not state.current_ontology.is_null()
            else None,
        )
        if len(state.aggregated_facts) == 0:
            logger.warning(
                "Facts aggregation produced an empty graph from "
                f"{len(state.parallel_facts_units)} successful unit(s)."
            )
        if state.source_url and state.doc_namespace:
            state.aggregated_facts.add(
                (URIRef(state.doc_namespace), DCTERMS.source, URIRef(state.source_url))
            )
        state.status = Status.SUCCESS
        return state

    return merge_facts
