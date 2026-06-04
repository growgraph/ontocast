from functools import partial

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from ontocast.agent import chunk_text, convert_document
from ontocast.agent.serialize import serialize
from ontocast.onto.enum import WorkflowNode
from ontocast.onto.state import AgentState
from ontocast.stategraph.node_factories import (
    make_consistency_critic_node,
    make_consolidate_ontology_node,
    make_merge_facts_node,
    make_normalize_ontology_node,
    make_render_facts_node,
    make_render_ontology_node,
    make_structural_check_node,
    make_summarize_chunks_node,
)
from ontocast.stategraph.routing import (
    route_after_chunk,
    route_after_tag_or_chunk,
)
from ontocast.toolbox import ToolBox


def create_agent_graph(tools: ToolBox) -> CompiledStateGraph:
    """Create the parallel map/reduce agent graph.

    Flow: CONVERT -> CHUNK (prepare: segment, tag, filter, size) ->
          [SUMMARIZE_CHUNKS] -> (conditional extraction)

    Per-unit ontology context is assembled inside ``ontology_loop`` (not at a
    document-level select node). For ``ONTOLOGY_AND_FACTS``, the full ontology
    block completes before the facts map runs; facts use the merged document
    ontology from ``AgentState``.
    """
    workflow = StateGraph(AgentState)

    convert_document_node = partial(convert_document, tools=tools)
    chunk_text_node = partial(chunk_text, tools=tools)
    serialize_node = partial(serialize, tools=tools)

    summarize_chunks_node = make_summarize_chunks_node(tools)
    render_ontology_node = make_render_ontology_node(tools)
    normalize_ontology_node = make_normalize_ontology_node(tools)
    consolidate_ontology_node = make_consolidate_ontology_node(tools)
    render_facts_node = make_render_facts_node(tools)
    merge_facts_node = make_merge_facts_node(tools)
    structural_check_node = make_structural_check_node(tools)
    consistency_critic_node = make_consistency_critic_node(tools)

    workflow.add_node(WorkflowNode.CONVERT_TO_TEXT, convert_document_node)
    workflow.add_node(WorkflowNode.CHUNK, chunk_text_node)
    workflow.add_node(WorkflowNode.SUMMARIZE_CHUNKS, summarize_chunks_node)
    workflow.add_node(WorkflowNode.RENDER_ONTOLOGY_UPDATE, render_ontology_node)
    workflow.add_node(WorkflowNode.NORMALIZE_ONTOLOGY_UPDATES, normalize_ontology_node)
    workflow.add_node(WorkflowNode.CONSOLIDATE_ONTOLOGY, consolidate_ontology_node)
    workflow.add_node(WorkflowNode.RENDER_FACTS, render_facts_node)
    workflow.add_node(WorkflowNode.MERGE_FACTS, merge_facts_node)
    workflow.add_node(WorkflowNode.STRUCTURAL_CHECK, structural_check_node)
    workflow.add_node(WorkflowNode.CONSISTENCY_CRITIC, consistency_critic_node)
    workflow.add_node(WorkflowNode.SERIALIZE, serialize_node)
    workflow.add_edge(START, WorkflowNode.CONVERT_TO_TEXT)
    workflow.add_edge(WorkflowNode.CONVERT_TO_TEXT, WorkflowNode.CHUNK)
    workflow.add_conditional_edges(
        WorkflowNode.CHUNK,
        route_after_chunk,
        {
            WorkflowNode.SUMMARIZE_CHUNKS: WorkflowNode.SUMMARIZE_CHUNKS,
            WorkflowNode.RENDER_ONTOLOGY_UPDATE: WorkflowNode.RENDER_ONTOLOGY_UPDATE,
            WorkflowNode.RENDER_FACTS: WorkflowNode.RENDER_FACTS,
        },
    )
    workflow.add_conditional_edges(
        WorkflowNode.SUMMARIZE_CHUNKS,
        route_after_tag_or_chunk,
        {
            WorkflowNode.RENDER_ONTOLOGY_UPDATE: WorkflowNode.RENDER_ONTOLOGY_UPDATE,
            WorkflowNode.RENDER_FACTS: WorkflowNode.RENDER_FACTS,
        },
    )
    workflow.add_edge(
        WorkflowNode.RENDER_ONTOLOGY_UPDATE, WorkflowNode.NORMALIZE_ONTOLOGY_UPDATES
    )
    workflow.add_edge(
        WorkflowNode.NORMALIZE_ONTOLOGY_UPDATES, WorkflowNode.CONSOLIDATE_ONTOLOGY
    )
    workflow.add_edge(WorkflowNode.CONSOLIDATE_ONTOLOGY, WorkflowNode.STRUCTURAL_CHECK)
    workflow.add_edge(WorkflowNode.RENDER_FACTS, WorkflowNode.MERGE_FACTS)

    workflow.add_edge(WorkflowNode.STRUCTURAL_CHECK, WorkflowNode.CONSISTENCY_CRITIC)
    workflow.add_edge(WorkflowNode.MERGE_FACTS, WorkflowNode.SERIALIZE)

    def route_after_consistency_critic(state: AgentState) -> str:
        if state.render_facts:
            return WorkflowNode.RENDER_FACTS
        return WorkflowNode.SERIALIZE

    workflow.add_conditional_edges(
        WorkflowNode.CONSISTENCY_CRITIC,
        route_after_consistency_critic,
        {
            WorkflowNode.RENDER_FACTS: WorkflowNode.RENDER_FACTS,
            WorkflowNode.SERIALIZE: WorkflowNode.SERIALIZE,
        },
    )
    workflow.add_edge(WorkflowNode.SERIALIZE, END)

    return workflow.compile()
