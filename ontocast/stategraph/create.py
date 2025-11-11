from functools import partial

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from ontocast.agent import (
    check_chunks_empty,
    chunk_text,
    convert_document,
    select_ontology,
    sublimate_ontology,
)
from ontocast.agent.aggregate_serialize import aggregate, serialize
from ontocast.agent.criticise_facts import criticise_facts
from ontocast.agent.criticise_ontology import criticise_ontology
from ontocast.agent.render_facts import render_facts
from ontocast.agent.render_ontology import render_ontology
from ontocast.onto.enum import FactsDecision, OntologyDecision, Status, WorkflowNode
from ontocast.onto.state import AgentState
from ontocast.stategraph.util import count_visits_conditional_success, wrap_with
from ontocast.toolbox import ToolBox


def create_agent_graph(tools: ToolBox) -> CompiledStateGraph:
    """Create the agent workflow graph.

    This function constructs a directed graph representing the workflow of the
    ontology-based knowledge graph agent. The graph defines the sequence of
    operations and their dependencies for processing documents and generating
    knowledge graphs.

    Args:
        tools: The ToolBox instance containing all necessary tools for the workflow.

    Returns:
        CompiledStateGraph: A compiled state graph ready for execution.
    """
    workflow = StateGraph(AgentState)

    # Create nodes with partial application of tools
    select_ontology_ = partial(select_ontology, tools=tools)
    convert_document_ = partial(convert_document, tools=tools)
    chunk_text_ = partial(chunk_text, tools=tools)
    check_chunks_empty_ = partial(check_chunks_empty)

    # Use structured hybrid agents with Turtle/SPARQL decision logic
    render_ontology_tuple = wrap_with(
        partial(render_ontology, tools=tools),
        WorkflowNode.TEXT_TO_ONTOLOGY,
        count_visits_conditional_success,
    )
    render_facts_tuple = wrap_with(
        partial(render_facts, tools=tools),
        WorkflowNode.TEXT_TO_FACTS,
        count_visits_conditional_success,
    )
    criticise_ontology_tuple = wrap_with(
        partial(criticise_ontology, tools=tools),
        WorkflowNode.CRITICISE_ONTOLOGY,
        count_visits_conditional_success,
    )
    criticise_facts_tuple = wrap_with(
        partial(criticise_facts, tools=tools),
        WorkflowNode.CRITICISE_FACTS,
        count_visits_conditional_success,
    )
    sublimate_ontology_tuple = partial(sublimate_ontology, tools=tools)
    aggregate_facts_tuple = partial(aggregate, tools=tools)
    serialize_tuple = partial(serialize, tools=tools)

    def ontology_routing(state: AgentState):
        """Custom routing function for TEXT_TO_ONTOLOGY node.

        Routes to CRITICISE_ONTOLOGY if skip_ontology_development is False,
        otherwise routes directly to TEXT_TO_FACTS.

        Args:
            state: The current agent state.

        Returns:
            WorkflowNode: The next node to execute.
        """
        if state.skip_ontology_development:
            if state.status == Status.SUCCESS:
                return OntologyDecision.SKIP_TO_FACTS
            else:
                return OntologyDecision.FAILURE_NO_ONTOLOGY
        else:
            return OntologyDecision.IMPROVE_CREATE_ONTOLOGY

    def skip_facts_routing(state: AgentState):
        """Custom routing function for TEXT_TO_ONTOLOGY node.

        Routes to CRITICISE_ONTOLOGY if skip_ontology_development is False,
        otherwise routes directly to TEXT_TO_FACTS.

        Args:
            state: The current agent state.

        Returns:
            WorkflowNode: The next node to execute.
        """
        if state.skip_facts_rendering:
            return FactsDecision.SERIALIZE
        else:
            if state.status == Status.SUCCESS:
                return FactsDecision.TEXT_TO_FACTS
            else:
                return FactsDecision.TEXT_TO_ONTOLOGY

    def simple_routing(state: AgentState):
        """Simple routing function based on state status.

        Args:
            state: The current agent state.

        Returns:
            Status: The current status of the state.
        """
        return state.status

    # Add nodes using string values
    workflow.add_node(WorkflowNode.CONVERT_TO_MD, convert_document_)
    workflow.add_node(WorkflowNode.CHUNK, chunk_text_)
    workflow.add_node(WorkflowNode.SELECT_ONTOLOGY, select_ontology_)
    workflow.add_node(*render_ontology_tuple)
    workflow.add_node(*render_facts_tuple)
    workflow.add_node(WorkflowNode.SUBLIMATE_ONTOLOGY, sublimate_ontology_tuple)
    workflow.add_node(*criticise_ontology_tuple)
    workflow.add_node(*criticise_facts_tuple)
    workflow.add_node(WorkflowNode.CHUNKS_EMPTY, check_chunks_empty_)
    workflow.add_node(WorkflowNode.AGGREGATE_FACTS, aggregate_facts_tuple)
    workflow.add_node(WorkflowNode.SERIALIZE, serialize_tuple)

    # Standard edges using string values
    workflow.add_edge(START, WorkflowNode.CONVERT_TO_MD)
    workflow.add_edge(WorkflowNode.CONVERT_TO_MD, WorkflowNode.CHUNK)
    workflow.add_edge(WorkflowNode.CHUNK, WorkflowNode.CHUNKS_EMPTY)
    workflow.add_edge(WorkflowNode.SUBLIMATE_ONTOLOGY, WorkflowNode.CRITICISE_FACTS)
    workflow.add_edge(WorkflowNode.AGGREGATE_FACTS, WorkflowNode.SERIALIZE)
    workflow.add_edge(WorkflowNode.SERIALIZE, END)

    # Add conditional edges for workflow control
    workflow.add_conditional_edges(
        WorkflowNode.CHUNKS_EMPTY,
        simple_routing,
        {
            Status.SUCCESS: WorkflowNode.AGGREGATE_FACTS,
            Status.FAILED: WorkflowNode.SELECT_ONTOLOGY,
        },
    )

    workflow.add_conditional_edges(
        WorkflowNode.SELECT_ONTOLOGY,
        ontology_routing,
        {
            OntologyDecision.FAILURE_NO_ONTOLOGY: END,
            OntologyDecision.IMPROVE_CREATE_ONTOLOGY: WorkflowNode.TEXT_TO_ONTOLOGY,
            OntologyDecision.SKIP_TO_FACTS: WorkflowNode.TEXT_TO_FACTS,
        },
    )

    workflow.add_conditional_edges(
        WorkflowNode.TEXT_TO_ONTOLOGY,
        simple_routing,
        {
            Status.SUCCESS: WorkflowNode.CRITICISE_ONTOLOGY,
            Status.FAILED: WorkflowNode.TEXT_TO_ONTOLOGY,
        },
    )

    workflow.add_conditional_edges(
        WorkflowNode.CRITICISE_ONTOLOGY,
        skip_facts_routing,
        {
            FactsDecision.TEXT_TO_FACTS: WorkflowNode.TEXT_TO_FACTS,
            FactsDecision.TEXT_TO_ONTOLOGY: WorkflowNode.TEXT_TO_ONTOLOGY,
            FactsDecision.SERIALIZE: WorkflowNode.SERIALIZE,
        },
    )

    workflow.add_conditional_edges(
        WorkflowNode.TEXT_TO_FACTS,
        simple_routing,
        {
            Status.SUCCESS: WorkflowNode.SUBLIMATE_ONTOLOGY,
            Status.FAILED: WorkflowNode.TEXT_TO_FACTS,
        },
    )

    workflow.add_conditional_edges(
        WorkflowNode.CRITICISE_FACTS,
        simple_routing,
        {
            Status.SUCCESS: WorkflowNode.CHUNKS_EMPTY,
            Status.FAILED: WorkflowNode.TEXT_TO_FACTS,
        },
    )

    return workflow.compile()
