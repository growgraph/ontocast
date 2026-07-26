import logging

from ontocast.onto.ontology_apply import complement_inserts
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import TripleOp
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitOntologyState

logger = logging.getLogger(__name__)


def all_unit_patch_source_iris(state: AgentState) -> list[str]:
    """Sorted unique ontology IRIs appearing in any unit's patch source list."""
    seen: set[str] = set()
    ordered: list[str] = []
    for sources in state.unit_patch_sources.values():
        for iri in sources:
            if iri not in seen:
                seen.add(iri)
                ordered.append(iri)
    return sorted(ordered)


def build_ontology_delta_graph(result: UnitOntologyState) -> RDFGraph:
    """Build a complement-only insert delta from a unit ontology result.

    Only *insert* triples are retained — delete operations in GraphUpdate are
    intentionally discarded (parallel-unit delete consensus is not implemented).

    Inserts already present in the prompt snapshot are subtracted so reduce
    receives true complements (``U \\ S``), not restated context triples.

    Fresh path (no GraphUpdates, empty seed): returns the full working graph.
    """
    if result.all_updates:
        delta_graph = RDFGraph()
        for graph_update in result.all_updates:
            dropped_deletes = sum(
                len(op.graph)
                for op in graph_update.triple_operations
                if isinstance(op, TripleOp) and op.type == "delete"
            )
            if dropped_deletes:
                logger.warning(
                    "build_ontology_delta_graph: unit produced %d delete triple(s) "
                    "that are dropped during map-reduce — delete operations are not "
                    "propagated through the ontology reduce stage.",
                    dropped_deletes,
                )
            insert_graph = graph_update.extract_insert_graph()
            for triple in insert_graph:
                delta_graph.add(triple)
            for prefix, namespace_uri in insert_graph.namespaces():
                if prefix:
                    delta_graph.bind(prefix, namespace_uri)
        return complement_inserts(delta_graph, result.ontology_snapshot.graph)

    # Fresh generation with no structured updates: emit the working graph only
    # when the seed was empty (true create path).
    if result.ontology_snapshot.is_empty() and len(result.working_graph) > 0:
        return result.working_graph.copy()
    return RDFGraph()


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
    if state.docling_doc is not None:
        return state.docling_doc.export_to_markdown()
    return ""
