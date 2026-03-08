import asyncio
import logging
from functools import partial

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from rdflib import DCTERMS, URIRef

from ontocast.agent import chunk_text, convert_document, select_ontology
from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.agent.render_ontology import render_ontology_update
from ontocast.agent.serialize import serialize
from ontocast.onto.content_unit import ContentUnit, OutputType, SourceUnit
from ontocast.onto.enum import Status, WorkflowNode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.unit_loops import unit_facts_loop, unit_ontology_loop
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def _delta_graph_from_ontology_result(result: UnitOntologyState) -> RDFGraph:
    """Extract delta graph from unit ontology result for aggregation.

    If all_updates is non-empty, union insert triples from each GraphUpdate.
    If all_updates is empty, use the full current_ontology graph as the delta.
    """
    if result.all_updates:
        delta = RDFGraph()
        for gu in result.all_updates:
            insert_graph = gu.extract_insert_graph()
            for triple in insert_graph:
                delta.add(triple)
            for prefix, uri in insert_graph.namespaces():
                if prefix:
                    delta.bind(prefix, uri)
        return delta
    return result.current_ontology.graph.copy()


def _route_after_select_ontology(state: AgentState) -> str:
    """Route after ontology selection."""
    if not state.render_ontology:
        return WorkflowNode.RENDER_FACTS
    if state.current_ontology.is_null():
        return WorkflowNode.BOOTSTRAP_ONTOLOGY
    return WorkflowNode.RENDER_ONTOLOGY_UPDATE


def _route_after_reduce_ontology(state: AgentState) -> str:
    """Route after ontology stage: facts map if needed, else serialize."""
    if state.render_facts:
        return WorkflowNode.RENDER_FACTS
    return WorkflowNode.SERIALIZE


def _create_document_excerpt(state: AgentState) -> str:
    """Create a representative excerpt from sampled source units."""
    excerpt_parts: list[str] = []

    if state.content_units:
        num_chunks = len(state.content_units)
        if num_chunks == 1:
            indices = [0]
        elif num_chunks == 2:
            indices = [0, 1]
        else:
            indices = [0, 1, num_chunks // 2, num_chunks - 1]

        seen: set[int] = set()
        for idx in indices:
            if idx in seen or idx < 0 or idx >= num_chunks:
                continue
            seen.add(idx)
            chunk_text = state.content_units[idx].text.strip()
            if not chunk_text:
                continue
            excerpt_parts.append(chunk_text)

    if excerpt_parts:
        return "\n\n[...]\n\n".join(excerpt_parts)
    if state.input_text:
        return state.input_text
    return ""


def create_agent_graph(tools: ToolBox) -> CompiledStateGraph:
    """Create the parallel map/reduce agent graph.

    Flow: CONVERT -> CHUNK -> (conditional)
          - ontology null: SELECT_ONTOLOGY -> (ontology or facts map)
          - ontology set: PARALLEL_ONTOLOGY_MAP or PARALLEL_FACTS_MAP
          - render_ontology: PARALLEL_ONTOLOGY_MAP -> REDUCE_ONTOLOGY ->
            [PARALLEL_FACTS_MAP -> REDUCE_FACTS]? -> SERIALIZE
          - render_facts only: PARALLEL_FACTS_MAP -> REDUCE_FACTS -> SERIALIZE

    One ontology is selected per document in the main workflow (SELECT_ONTOLOGY).
    """
    workflow = StateGraph(AgentState)

    convert_document_ = partial(convert_document, tools=tools)
    chunk_text_ = partial(chunk_text, tools=tools)
    select_ontology_ = partial(select_ontology, tools=tools)
    serialize_ = partial(serialize, tools=tools)

    async def bootstrap_ontology(state: AgentState) -> AgentState:
        """Create one seed ontology for null-selection flow."""
        if not state.render_ontology or not state.current_ontology.is_null():
            state.status = Status.SUCCESS
            return state
        if not state.content_units:
            state.status = Status.SUCCESS
            return state

        excerpt = _create_document_excerpt(state).strip()
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
            max_retries=tools.config.server.parallel_ontology_retries,
            current_domain=state.current_domain,
            ontology_max_triples=tools.config.server.ontology_max_triples,
        )
        result = await unit_ontology_loop(bootstrap_state, tools)
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

    async def render_ontology(state: AgentState) -> AgentState:
        if not state.content_units:
            state.ontology_units = []
            state.status = Status.SUCCESS
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        async def process_unit(unit_idx: int) -> tuple[int, UnitOntologyState]:
            async with semaphore:
                base_state = state.model_copy(deep=True)
                ontology_state = UnitOntologyState(
                    content_unit=state.content_units[unit_idx],
                    ontology_snapshot=state.current_ontology,
                    ontology_user_instruction=state.ontology_user_instruction,
                    budget_tracker=base_state.budget_tracker,
                    max_retries=tools.config.server.parallel_ontology_retries,
                    current_domain=state.current_domain,
                    ontology_max_triples=tools.config.server.ontology_max_triples,
                )
                result = await unit_ontology_loop(ontology_state, tools)
                return unit_idx, result

        tasks = [process_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results = await asyncio.gather(*tasks)
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        ontology_units: list[ContentUnit] = []
        failed = 0
        for _, result in ordered_results:
            if result.status == Status.SUCCESS:
                cu = result.content_unit
                delta_graph = _delta_graph_from_ontology_result(result)
                ontology_units.append(
                    ContentUnit(
                        text=cu.text,
                        index=cu.index,
                        doc_iri=cu.doc_iri,
                        graph=delta_graph,
                        type=OutputType.ONTOLOGIES,
                    )
                )
            else:
                failed += 1

        if failed:
            logger.warning(
                f"Parallel ontology map failed for {failed}/{len(state.content_units)} unit(s)"
            )

        state.ontology_units = ontology_units
        state.status = Status.SUCCESS
        return state

    def normalize_ontology(state: AgentState) -> AgentState:
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

    async def consolidate_ontology(state: AgentState) -> AgentState:
        """Optional post-normalization ontology consolidation pass."""
        if not tools.config.server.enable_ontology_consolidation:
            state.status = Status.SUCCESS
            return state
        if not state.render_ontology or state.current_ontology.is_null():
            state.status = Status.SUCCESS
            return state

        excerpt = _create_document_excerpt(state).strip()
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
            max_retries=1,
            current_domain=state.current_domain,
            ontology_max_triples=tools.config.server.ontology_max_triples,
        )
        result = await render_ontology_update(consolidation_state, tools)
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

    async def render_facts(state: AgentState) -> AgentState:
        if not state.content_units:
            state.parallel_facts_units = []
            state.status = Status.SUCCESS
            return state

        worker_limit = max(1, tools.config.server.parallel_workers)
        semaphore = asyncio.Semaphore(worker_limit)

        async def process_unit(unit_idx: int) -> tuple[int, UnitFactsState]:
            async with semaphore:
                base_state = state.model_copy(deep=True)
                facts_state = UnitFactsState(
                    content_unit=state.content_units[unit_idx],
                    ontology_snapshot=state.current_ontology,
                    facts_user_instruction=state.facts_user_instruction,
                    budget_tracker=base_state.budget_tracker,
                    max_retries=tools.config.server.parallel_facts_retries,
                )
                result = await unit_facts_loop(facts_state, tools)
                return unit_idx, result

        tasks = [process_unit(i) for i, _ in enumerate(state.content_units)]
        raw_results = await asyncio.gather(*tasks)
        ordered_results = sorted(raw_results, key=lambda item: item[0])

        facts_units: list[ContentUnit] = []
        failed = 0
        for _, result in ordered_results:
            if result.status == Status.SUCCESS:
                facts_units.append(result.content_unit)
            else:
                failed += 1

        if failed:
            logger.warning(
                f"Parallel facts map failed for {failed}/{len(state.content_units)} unit(s)"
            )

        state.parallel_facts_units = facts_units
        state.status = Status.SUCCESS
        return state

    def merge_facts(state: AgentState) -> AgentState:
        if not state.parallel_facts_units:
            state.aggregated_facts = RDFGraph()
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
    workflow.add_node(WorkflowNode.RENDER_ONTOLOGY_UPDATE, render_ontology)
    workflow.add_node(WorkflowNode.NORMALIZE_ONTOLOGY_UPDATES, normalize_ontology)
    workflow.add_node(WorkflowNode.CONSOLIDATE_ONTOLOGY, consolidate_ontology)
    workflow.add_node(WorkflowNode.RENDER_FACTS, render_facts)
    workflow.add_node(WorkflowNode.MERGE_FACTS, merge_facts)
    workflow.add_node(WorkflowNode.SERIALIZE, serialize_)
    workflow.add_edge(WorkflowNode.CHUNK, WorkflowNode.SELECT_ONTOLOGY)
    workflow.add_conditional_edges(
        WorkflowNode.SELECT_ONTOLOGY,
        _route_after_select_ontology,
        {
            WorkflowNode.BOOTSTRAP_ONTOLOGY: WorkflowNode.BOOTSTRAP_ONTOLOGY,
            WorkflowNode.RENDER_ONTOLOGY_UPDATE: WorkflowNode.RENDER_ONTOLOGY_UPDATE,
            WorkflowNode.RENDER_FACTS: WorkflowNode.RENDER_FACTS,
        },
    )
    workflow.add_edge(
        WorkflowNode.BOOTSTRAP_ONTOLOGY, WorkflowNode.RENDER_ONTOLOGY_UPDATE
    )
    workflow.add_edge(START, WorkflowNode.CONVERT_TO_MD)
    workflow.add_edge(WorkflowNode.CONVERT_TO_MD, WorkflowNode.CHUNK)
    workflow.add_edge(
        WorkflowNode.RENDER_ONTOLOGY_UPDATE, WorkflowNode.NORMALIZE_ONTOLOGY_UPDATES
    )
    workflow.add_edge(
        WorkflowNode.NORMALIZE_ONTOLOGY_UPDATES, WorkflowNode.CONSOLIDATE_ONTOLOGY
    )
    workflow.add_conditional_edges(
        WorkflowNode.CONSOLIDATE_ONTOLOGY,
        _route_after_reduce_ontology,
        {
            WorkflowNode.RENDER_FACTS: WorkflowNode.RENDER_FACTS,
            WorkflowNode.SERIALIZE: WorkflowNode.SERIALIZE,
        },
    )
    workflow.add_edge(WorkflowNode.RENDER_FACTS, WorkflowNode.MERGE_FACTS)
    workflow.add_edge(WorkflowNode.MERGE_FACTS, WorkflowNode.SERIALIZE)
    workflow.add_edge(WorkflowNode.SERIALIZE, END)

    return workflow.compile()
