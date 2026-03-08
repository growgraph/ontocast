import logging

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitOntologyState

logger = logging.getLogger(__name__)


def build_ontology_delta_graph(result: UnitOntologyState) -> RDFGraph:
    """Build a delta graph from a unit ontology result.

    If update operations exist, only inserted triples are aggregated.
    Otherwise, the current ontology snapshot is used as the delta.
    """
    if result.all_updates:
        delta_graph = RDFGraph()
        for graph_update in result.all_updates:
            insert_graph = graph_update.extract_insert_graph()
            for triple in insert_graph:
                delta_graph.add(triple)
            for prefix, namespace_uri in insert_graph.namespaces():
                if prefix:
                    delta_graph.bind(prefix, namespace_uri)
        return delta_graph

    return result.current_ontology.graph.copy()


def build_document_excerpt(state: AgentState) -> str:
    """Create a representative excerpt from sampled source units."""
    excerpt_parts: list[str] = []

    if state.content_units:
        unit_count = len(state.content_units)
        if unit_count == 1:
            sample_indices = [0]
        elif unit_count == 2:
            sample_indices = [0, 1]
        else:
            sample_indices = [0, 1, unit_count // 2, unit_count - 1]

        visited_indices: set[int] = set()
        for index in sample_indices:
            if index in visited_indices or index < 0 or index >= unit_count:
                continue
            visited_indices.add(index)
            unit_text = state.content_units[index].text.strip()
            if not unit_text:
                continue
            excerpt_parts.append(unit_text)

    if excerpt_parts:
        return "\n\n[...]\n\n".join(excerpt_parts)
    if state.input_text:
        return state.input_text
    return ""
