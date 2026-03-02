"""Parallel map/reduce workflow graph for OntoCast."""

import asyncio
import logging
from functools import partial

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from rdflib import DCTERMS, URIRef

from ontocast.agent import chunk_text, convert_document, select_ontology
from ontocast.agent.aggregate_serialize import serialize
from ontocast.agent.reduce_results import reduce_ontology_updates
from ontocast.agent.unit_loops import run_unit_facts_loop, run_unit_ontology_loop
from ontocast.onto.enum import Status, WorkflowNode
from ontocast.onto.parallel_state import UnitFactsState, UnitOntologyState
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def create_parallel_agent_graph(tools: ToolBox) -> CompiledStateGraph:
    """Create the experimental parallel map/reduce agent graph."""
    workflow = StateGraph(AgentState)

    convert_document_ = partial(convert_document, tools=tools)
    chunk_text_ = partial(chunk_text, tools=tools)
    select_ontology_ = partial(select_ontology, tools=tools)
    serialize_ = partial(serialize, tools=tools)

    async def bootstrap_ontology(state: AgentState) -> AgentState:
        if state.skip_ontology_development:
            return state
        if not state.current_ontology.is_null():
            return state
        if not state.content_units:
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        async def process_bootstrap_unit(
            unit_idx: int,
        ) -> tuple[int, UnitOntologyState]:
            async with semaphore:
                base_state = state.model_copy(deep=True)
                ontology_state = UnitOntologyState(
                    content_unit=state.content_units[unit_idx],
                    ontology_snapshot=state.current_ontology,
                    ontology_user_instruction=state.ontology_user_instruction,
                    budget_tracker=base_state.budget_tracker,
                    max_retries=tools.config.server.parallel_ontology_retries,
                )
                result = await run_unit_ontology_loop(ontology_state, tools)
                return unit_idx, result

        tasks = [process_bootstrap_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results = await asyncio.gather(*tasks)
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        failed_chunks = 0
        aggregated_updates = []
        for _, result in ordered_results:
            if result.status != Status.SUCCESS:
                failed_chunks += 1
            aggregated_updates.extend(result.output_updates)

        if failed_chunks:
            logger.warning(
                "Parallel ontology bootstrap failed for "
                f"{failed_chunks}/{len(state.content_units)} unit(s)"
            )

        state.current_ontology = reduce_ontology_updates(
            base_ontology=state.current_ontology,
            updates=aggregated_updates,
            ontology_max_triples=state.ontology_max_triples,
        )
        if not state.current_ontology.is_null():
            state.clear_failure()
        else:
            logger.warning(
                "Parallel pipeline bootstrap could not produce ontology; "
                "continuing with selected ontology snapshot."
            )
        return state

    async def map_units_parallel(state: AgentState) -> AgentState:
        if not state.content_units:
            state.parallel_facts_units = []
            state.parallel_ontology_updates = []
            state.status = Status.SUCCESS
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        async def process_one_unit(unit_idx: int):
            async with semaphore:
                base_state = state.model_copy(deep=True)
                facts_state = UnitFactsState(
                    content_unit=state.content_units[unit_idx],
                    ontology_snapshot=state.current_ontology,
                    facts_user_instruction=state.facts_user_instruction,
                    budget_tracker=base_state.budget_tracker,
                    max_retries=tools.config.server.parallel_facts_retries,
                )
                facts_task = run_unit_facts_loop(facts_state, tools)
                if state.skip_ontology_development:
                    ontology_result = None
                    facts_result = await facts_task
                else:
                    ontology_state = UnitOntologyState(
                        content_unit=state.content_units[unit_idx],
                        ontology_snapshot=state.current_ontology,
                        ontology_user_instruction=state.ontology_user_instruction,
                        budget_tracker=base_state.budget_tracker,
                        max_retries=tools.config.server.parallel_ontology_retries,
                    )
                    ontology_task = run_unit_ontology_loop(ontology_state, tools)
                    facts_result, ontology_result = await asyncio.gather(
                        facts_task, ontology_task
                    )
                return unit_idx, facts_result, ontology_result

        tasks = [process_one_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results = await asyncio.gather(*tasks)
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        facts_units = []
        ontology_updates = []
        failed_facts = 0

        for _, facts_result, ontology_result in ordered_results:
            if (
                facts_result.status == Status.SUCCESS
                and facts_result.output_unit is not None
            ):
                facts_units.append(facts_result.output_unit)
            else:
                failed_facts += 1

            if ontology_result is None:
                continue
            ontology_updates.extend(ontology_result.output_updates)

        if failed_facts:
            logger.warning(
                f"Parallel facts map failed for {failed_facts}/{len(state.content_units)} unit(s)"
            )

        state.parallel_facts_units = facts_units
        state.parallel_ontology_updates = ontology_updates
        state.status = Status.SUCCESS
        return state

    def reduce_ontology(state: AgentState) -> AgentState:
        if state.skip_ontology_development:
            state.status = Status.SUCCESS
            return state

        state.current_ontology = reduce_ontology_updates(
            base_ontology=state.current_ontology,
            updates=state.parallel_ontology_updates,
            ontology_max_triples=state.ontology_max_triples,
        )
        state.status = Status.SUCCESS
        return state

    def reduce_facts(state: AgentState) -> AgentState:
        if not state.parallel_facts_units:
            state.aggregated_facts = state.aggregated_facts.__class__()
            state.status = Status.SUCCESS
            return state

        for unit in state.parallel_facts_units:
            unit.sanitize()
        state.aggregated_facts = tools.aggregator.aggregate_graphs(
            units=state.parallel_facts_units
        )
        if state.source_url and state.doc_namespace:
            state.aggregated_facts.add(
                (URIRef(state.doc_namespace), DCTERMS.source, URIRef(state.source_url))
            )
        state.status = Status.SUCCESS
        return state

    workflow.add_node(WorkflowNode.CONVERT_TO_MD, convert_document_)
    workflow.add_node(WorkflowNode.CHUNK, chunk_text_)
    workflow.add_node(WorkflowNode.SELECT_ONTOLOGY, select_ontology_)
    workflow.add_node(WorkflowNode.BOOTSTRAP_ONTOLOGY, bootstrap_ontology)
    workflow.add_node(WorkflowNode.PARALLEL_MAP_UNITS, map_units_parallel)
    workflow.add_node(WorkflowNode.REDUCE_ONTOLOGY, reduce_ontology)
    workflow.add_node(WorkflowNode.REDUCE_FACTS, reduce_facts)
    workflow.add_node(WorkflowNode.SERIALIZE, serialize_)

    workflow.add_edge(START, WorkflowNode.CONVERT_TO_MD)
    workflow.add_edge(WorkflowNode.CONVERT_TO_MD, WorkflowNode.CHUNK)
    workflow.add_edge(WorkflowNode.CHUNK, WorkflowNode.SELECT_ONTOLOGY)
    workflow.add_edge(WorkflowNode.SELECT_ONTOLOGY, WorkflowNode.BOOTSTRAP_ONTOLOGY)
    workflow.add_edge(WorkflowNode.BOOTSTRAP_ONTOLOGY, WorkflowNode.PARALLEL_MAP_UNITS)
    workflow.add_edge(WorkflowNode.PARALLEL_MAP_UNITS, WorkflowNode.REDUCE_ONTOLOGY)
    workflow.add_edge(WorkflowNode.REDUCE_ONTOLOGY, WorkflowNode.REDUCE_FACTS)
    workflow.add_edge(WorkflowNode.REDUCE_FACTS, WorkflowNode.SERIALIZE)
    workflow.add_edge(WorkflowNode.SERIALIZE, END)

    return workflow.compile()
