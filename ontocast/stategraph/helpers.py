import logging

from rdflib import URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_apply import OntologyDelta
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState

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


def build_document_text(state: AgentState, max_chars: int) -> str:
    """The document's content units, in order, within ``max_chars``.

    Every unit keeps its opening: once the whole text exceeds the budget, each
    unit is cut to the same share of its length. Units routed away from
    extraction (a reference list, an author block) are left out unless no unit
    is plain content.

    Args:
        state: The document state.
        max_chars: Character budget for the joined text.

    Returns:
        The units' texts joined by blank lines, or the excerpt when no unit
        has text.
    """
    units = [
        unit
        for unit in state.content_units
        if not unit.is_citation_metadata and not unit.is_non_content
    ] or list(state.content_units)
    texts = [unit.text.strip() for unit in units if unit.text.strip()]
    if not texts:
        return build_document_excerpt(state)
    total = sum(len(text) for text in texts)
    if total > max_chars > 0:
        share = max_chars / total
        texts = [
            text[: max(1, int(len(text) * share))].rstrip() + " [...]" for text in texts
        ]
    return "\n\n".join(texts)


def build_document_excerpt(state: AgentState) -> str:
    """Create a representative excerpt from sampled source units.

    Samples by position among the units routed to extraction: a reference
    list or an author block kept under its own route would otherwise stand in
    for the document whenever it is the last unit. Falls back to every unit
    when none is plain content.
    """
    excerpt_parts: list[str] = []

    units = [
        unit
        for unit in state.content_units
        if not unit.is_citation_metadata and not unit.is_non_content
    ] or list(state.content_units)
    if units:
        unit_count = len(units)
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
            unit_text = units[index].text.strip()
            if not unit_text:
                continue
            excerpt_parts.append(unit_text)

    if excerpt_parts:
        return "\n\n[...]\n\n".join(excerpt_parts)
    if state.docling_doc is not None:
        return state.docling_doc.export_to_markdown()
    return ""


def enforce_redeclared_deletes(
    delta: OntologyDelta, judged_on_partial_view: RDFGraph
) -> int:
    """Drop deletes judged on a partial view whose subject is not redeclared.

    A unit whose context was a retrieved subset of the catalog
    (:attr:`OntologySnapshot.partial`) judged its deletes without seeing the
    rest of the ontology, so such a delete is trusted only as part of a
    redeclaration: the corrected statement's subject must reappear in the
    document's merged inserts. A bare delete of catalog content, judged on
    partial evidence, would propagate onto shared catalog terminals across
    documents. Deletes from units that saw their ontologies whole are not
    touched. A unit can never delete a statement it was not shown: its deletes
    are the snapshot triples its updates removed.

    Args:
        delta: The merged document delta, edited in place.
        judged_on_partial_view: The deletes proposed by units with a partial
            view.

    Returns:
        Number of delete triples dropped.
    """
    if len(delta.deletes) == 0 or len(judged_on_partial_view) == 0:
        return 0
    redeclared = {
        subject for subject in delta.inserts.subjects() if isinstance(subject, URIRef)
    }
    dropped = 0
    for triple in list(delta.deletes):
        if triple in judged_on_partial_view and triple[0] not in redeclared:
            delta.deletes.remove(triple)
            dropped += 1
    if dropped:
        logger.warning(
            "Dropped %d delete triple(s) judged on a partial view whose subject "
            "the merged inserts do not redeclare",
            dropped,
        )
    return dropped


def drop_all_deletes(delta: OntologyDelta) -> int:
    """Empty the delete side of ``delta`` (``ONTOLOGY_UPDATE_MODE=insert_only``).

    The render path already drops rendered deletes; this guards the apply
    step against a delete reaching it from any other path.

    Returns:
        Number of delete triples dropped.
    """
    dropped = len(delta.deletes)
    if dropped:
        delta.deletes = RDFGraph()
    return dropped


def reconcile_fresh_ontologies(
    fresh_ontologies: list[Ontology],
) -> tuple[list[Ontology], dict[str, int]]:
    """Union the fresh fan-out's per-unit graphs into one root per IRI.

    Every fresh unit of a document mints under the document's home ontology,
    so there is one IRI and the union is the ontology. A unit that minted
    under any other IRI is a fault in the fresh path and raises.

    Returns:
        The reconciled artifact list and the metrics to record
        (``fresh_ontologies_merged``, present only when non-zero).
    """
    metrics: dict[str, int] = {}
    if len(fresh_ontologies) <= 1:
        return fresh_ontologies, metrics
    iris = {fresh.iri for fresh in fresh_ontologies}
    if len(iris) != 1:
        raise RuntimeError(
            f"fresh units minted under {len(iris)} IRIs ({sorted(iris)}); "
            "a document has one home ontology"
        )
    metrics["fresh_ontologies_merged"] = len(fresh_ontologies) - 1
    return [fresh_ontologies[0].union_fresh(fresh_ontologies[1:])], metrics
