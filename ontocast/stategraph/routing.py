from ontocast.onto.enum import WorkflowNode
from ontocast.onto.state import AgentState


def route_after_chunk(state: AgentState) -> str:
    """Route after chunking: ontology map-reduce vs facts-only."""
    if not state.render_ontology:
        return WorkflowNode.RENDER_FACTS
    return WorkflowNode.RENDER_ONTOLOGY_UPDATE


def route_after_ontology_consolidation(state: AgentState) -> str:
    """Route after ontology stage to the ontology-only structural check."""
    return WorkflowNode.STRUCTURAL_CHECK
