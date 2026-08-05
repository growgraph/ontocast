import logging

from ontocast.onto.ontology_apply import OntologyDelta
from ontocast.onto.rdfgraph import RDFGraph
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


def build_ontology_delta_graph(result: UnitOntologyState) -> OntologyDelta:
    """Build the net insert/delete delta from a unit ontology result.

    All GraphUpdates (applied and pending) are replayed in order onto a copy of
    the prompt snapshot, then diffed against it. This honors operation order —
    a triple deleted and later re-inserted nets out — and yields:

    - ``inserts``: true complements (``U \\ S``), never restated context triples;
    - ``deletes``: snapshot triples removed by delete operations, to be
      propagated onto catalog terminals during reduce.

    Fresh path (no GraphUpdates, empty seed): full working graph as inserts.
    """
    if result.all_updates:
        snapshot_graph = result.ontology_snapshot.graph
        final_graph, _ = AgentState.render_updated_graph(
            snapshot_graph, result.all_updates, max_triples=None
        )
        snapshot_set = set(snapshot_graph)
        final_set = set(final_graph)
        inserts = RDFGraph()
        deletes = RDFGraph()
        for prefix, namespace_uri in final_graph.namespaces():
            if prefix:
                inserts.bind(prefix, namespace_uri)
                deletes.bind(prefix, namespace_uri)
        for triple in final_set - snapshot_set:
            inserts.add(triple)
        for triple in snapshot_set - final_set:
            deletes.add(triple)
        if len(deletes) > 0:
            logger.info(
                "build_ontology_delta_graph: unit produced %d delete triple(s) "
                "for catalog propagation.",
                len(deletes),
            )
        return OntologyDelta(inserts=inserts, deletes=deletes)

    # Fresh generation with no structured updates: emit the working graph only
    # when the seed was empty (true create path).
    if result.ontology_snapshot.is_empty() and len(result.working_graph) > 0:
        return OntologyDelta(inserts=result.working_graph.copy())
    return OntologyDelta()


def merge_unit_deltas(deltas: list[OntologyDelta]) -> OntologyDelta:
    """Union per-unit deltas into one document-level delta.

    Delete consensus across parallel units is conservative: a triple inserted
    by any unit wins over another unit's delete of the same triple, so the
    merged delta stays monotone-safe under parallel map/reduce.
    """
    inserts = RDFGraph()
    deletes = RDFGraph()
    for delta in deltas:
        for graph, merged in ((delta.inserts, inserts), (delta.deletes, deletes)):
            for triple in graph:
                merged.add(triple)
            for prefix, namespace_uri in graph.namespaces():
                if prefix:
                    merged.bind(prefix, namespace_uri)
    insert_set = set(inserts)
    reconciled_deletes = RDFGraph()
    for prefix, namespace_uri in deletes.namespaces():
        if prefix:
            reconciled_deletes.bind(prefix, namespace_uri)
    vetoed = 0
    for triple in deletes:
        if triple in insert_set:
            vetoed += 1
            continue
        reconciled_deletes.add(triple)
    if vetoed:
        logger.info(
            "merge_unit_deltas: %d delete triple(s) vetoed by parallel-unit "
            "inserts (insert wins on conflict).",
            vetoed,
        )
    return OntologyDelta(inserts=inserts, deletes=reconciled_deletes)


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
