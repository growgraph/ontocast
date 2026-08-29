import logging

from rdflib import URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_apply import OntologyDelta
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool.ontology_validation import detect_minted_duplicates

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


def enforce_redeclared_deletes(delta: OntologyDelta) -> int:
    """Drop merged deletes whose subject the merged inserts do not redeclare.

    Applied only under vector-retrieval context: each unit judged its deletes
    on a *retrieved subset*, so a delete is trusted only as part of a
    redeclaration — the corrected statement must reappear in the inserts. A
    bare delete of catalog content, judged on partial evidence, would
    propagate onto shared catalog terminals cross-document. A safety invariant
    like the insert-wins veto in :func:`merge_unit_deltas`, not a knob; the
    per-unit ``foreign_delete`` finding teaches the loop, this backstops the
    write.

    Returns:
        Number of delete triples dropped.
    """
    if len(delta.deletes) == 0:
        return 0
    redeclared = {
        subject for subject in delta.inserts.subjects() if isinstance(subject, URIRef)
    }
    dropped = 0
    for triple in list(delta.deletes):
        if triple[0] not in redeclared:
            delta.deletes.remove(triple)
            dropped += 1
    if dropped:
        logger.warning(
            "Dropped %d delete triple(s) whose subject the merged inserts do "
            "not redeclare — under retrieved (partial) context an "
            "unredeclared delete is judged on partial evidence and would "
            "propagate to shared catalog terminals",
            dropped,
        )
    return dropped


def reconcile_fresh_ontologies(
    fresh_ontologies: list[Ontology],
) -> tuple[list[Ontology], dict[str, int]]:
    """Reconcile the fresh-create fan-out's N independent ontologies.

    Units on the fresh path each return a whole ontology. Same-IRI artifacts
    are union-merged (the reduce used to keep whichever indexed last,
    silently dropping the others' content); across *different* fresh IRIs,
    minted duplicates are detected pairwise so the overlap is at least
    visible — no LLM harmonization is attempted.

    Returns:
        The reconciled artifact list and the metrics to record
        (``fresh_ontologies_merged``, ``fresh_minted_duplicates``; keys
        present only when non-zero).
    """
    metrics: dict[str, int] = {}
    if len(fresh_ontologies) <= 1:
        return fresh_ontologies, metrics

    by_iri: dict[str, list[Ontology]] = {}
    for fresh in fresh_ontologies:
        by_iri.setdefault(fresh.iri, []).append(fresh)
    merged: list[Ontology] = []
    same_iri_merges = 0
    for group in by_iri.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        same_iri_merges += len(group) - 1
        logger.warning(
            "%d unit(s) produced fresh ontologies under the same IRI %s; "
            "union-merging instead of keeping the last one",
            len(group),
            group[0].iri,
        )
        merged.append(group[0].union_fresh(group[1:]))
    if same_iri_merges:
        metrics["fresh_ontologies_merged"] = same_iri_merges

    if len(merged) > 1:
        cross_duplicates = 0
        for index, fresh in enumerate(merged):
            other_graphs = {other.iri: other.graph for other in merged[index + 1 :]}
            cross_duplicates += len(detect_minted_duplicates(fresh.graph, other_graphs))
        if cross_duplicates:
            metrics["fresh_minted_duplicates"] = cross_duplicates
            logger.warning(
                "%d term(s) overlap across %d distinct fresh ontologies — "
                "consider a shared seed ontology",
                cross_duplicates,
                len(merged),
            )
    return merged, metrics
