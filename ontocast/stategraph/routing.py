from ontocast.onto.enum import WorkflowNode
from ontocast.onto.state import AgentState


def route_after_tag_or_chunk(state: AgentState) -> str:
    """Route after chunking: the home-ontology step and ontology map-reduce, or facts only."""
    if not state.render_ontology:
        return WorkflowNode.RENDER_FACTS
    return WorkflowNode.RESOLVE_HOME_ONTOLOGY
