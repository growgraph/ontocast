from functools import partial

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from ontocast.agent import chunk_text, convert_document, select_ontology
from ontocast.agent.serialize import serialize
from ontocast.onto.enum import WorkflowNode
from ontocast.onto.state import AgentState
from ontocast.stategraph.node_factories import (
    make_bootstrap_ontology_node,
    make_consistency_critic_node,
    make_consolidate_ontology_node,
    make_merge_facts_node,
    make_normalize_ontology_node,
    make_render_facts_node,
    make_render_ontology_node,
    make_structural_check_node,
)
from ontocast.stategraph.routing import (
    route_after_ontology_selection,
)
from ontocast.toolbox import ToolBox


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

    convert_document_node = partial(convert_document, tools=tools)
    chunk_text_node = partial(chunk_text, tools=tools)
    select_ontology_node = partial(select_ontology, tools=tools)
    serialize_node = partial(serialize, tools=tools)

    bootstrap_ontology_node = make_bootstrap_ontology_node(tools)
    render_ontology_node = make_render_ontology_node(tools)
    normalize_ontology_node = make_normalize_ontology_node(tools)
    consolidate_ontology_node = make_consolidate_ontology_node(tools)
    render_facts_node = make_render_facts_node(tools)
    merge_facts_node = make_merge_facts_node(tools)
    structural_check_node = make_structural_check_node(tools)
    consistency_critic_node = make_consistency_critic_node(tools)

    workflow.add_node(WorkflowNode.CONVERT_TO_MD, convert_document_node)
    workflow.add_node(WorkflowNode.CHUNK, chunk_text_node)
    workflow.add_node(WorkflowNode.SELECT_ONTOLOGY, select_ontology_node)
    workflow.add_node(WorkflowNode.BOOTSTRAP_ONTOLOGY, bootstrap_ontology_node)
    workflow.add_node(WorkflowNode.RENDER_ONTOLOGY_UPDATE, render_ontology_node)
    workflow.add_node(WorkflowNode.NORMALIZE_ONTOLOGY_UPDATES, normalize_ontology_node)
    workflow.add_node(WorkflowNode.CONSOLIDATE_ONTOLOGY, consolidate_ontology_node)
    workflow.add_node(WorkflowNode.RENDER_FACTS, render_facts_node)
    workflow.add_node(WorkflowNode.MERGE_FACTS, merge_facts_node)
    workflow.add_node(WorkflowNode.STRUCTURAL_CHECK, structural_check_node)
    workflow.add_node(WorkflowNode.CONSISTENCY_CRITIC, consistency_critic_node)
    workflow.add_node(WorkflowNode.SERIALIZE, serialize_node)
    workflow.add_edge(WorkflowNode.CHUNK, WorkflowNode.SELECT_ONTOLOGY)
    workflow.add_conditional_edges(
        WorkflowNode.SELECT_ONTOLOGY,
        route_after_ontology_selection,
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
    workflow.add_edge(WorkflowNode.CONSOLIDATE_ONTOLOGY, WorkflowNode.STRUCTURAL_CHECK)
    workflow.add_edge(WorkflowNode.RENDER_FACTS, WorkflowNode.MERGE_FACTS)

    # Facts-only mode: SELECT_ONTOLOGY routes directly to RENDER_FACTS.
    # Ontology path: CONSISTENCY_CRITIC routes to RENDER_FACTS (optional) or SERIALIZE.
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
