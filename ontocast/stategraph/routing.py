from ontocast.onto.enum import WorkflowNode
from ontocast.onto.state import AgentState


def route_after_tag_or_chunk(state: AgentState) -> str:
    """Route after tagging/summarization: ontology map-reduce vs facts-only."""
    if not state.render_ontology:
        return WorkflowNode.RENDER_FACTS
    return WorkflowNode.RENDER_ONTOLOGY_UPDATE
