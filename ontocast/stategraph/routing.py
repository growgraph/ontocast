from ontocast.onto.enum import WorkflowNode
from ontocast.onto.state import AgentState


def route_after_ontology_selection(state: AgentState) -> str:
    """Route after ontology selection."""
    if not state.render_ontology:
        return WorkflowNode.RENDER_FACTS
    if state.current_ontology.is_null():
        return WorkflowNode.BOOTSTRAP_ONTOLOGY
    return WorkflowNode.RENDER_ONTOLOGY_UPDATE


def route_after_ontology_consolidation(state: AgentState) -> str:
    """Route after ontology stage: facts map if needed, else serialize."""
    if state.render_facts:
        return WorkflowNode.RENDER_FACTS
    return WorkflowNode.SERIALIZE
