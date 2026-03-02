from langgraph.graph.state import CompiledStateGraph

from ontocast.stategraph.create_parallel import create_parallel_agent_graph
from ontocast.toolbox import ToolBox


def create_agent_graph(tools: ToolBox) -> CompiledStateGraph:
    """Create the production agent workflow graph."""
    return create_parallel_agent_graph(tools)
