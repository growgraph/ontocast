"""Shared post-parse mechanism for the two graph-update render agents.

``agent/common.py`` is about *calling* the LLM; this module is about what
happens to the patch once it has been parsed. Facts updates
and ontology updates
(:func:`ontocast.agent.render_ontology.render_ontology_update`) consume the same
:class:`GraphUpdateRenderReport`, so the hygiene between "the model answered"
and "the update is applied" belongs in one place rather than being reimplemented
per domain -- the ontology side previously had none of it at all.
"""

import logging
from collections import Counter
from collections.abc import Callable

from rdflib import Literal, URIRef

from ontocast.onto.iri_policy import is_iriref_text
from ontocast.onto.model import (
    FactsGateRepairKind,
    GraphRepairRecord,
    GraphUpdateRenderReport,
)
from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple, finalize_llm_graph
from ontocast.onto.sparql_models import GraphUpdate

logger = logging.getLogger(__name__)

#: Domain-specific repair applied to the INSERT side of a patch only. Returns
#: the repaired graph plus any triples it withheld from the applied graph.
InsertHook = Callable[[RDFGraph], tuple[RDFGraph, list[RejectedLiteralTriple]]]


def _malformed_iri(term: object) -> str | None:
    iri = term.datatype if isinstance(term, Literal) else term
    if isinstance(iri, URIRef) and not is_iriref_text(str(iri)):
        return str(iri)
    return None


def drop_malformed_iris(graph: RDFGraph) -> list[GraphRepairRecord]:
    """Remove, in place, every triple naming an IRI no ``IRIREF`` can hold.

    rdflib reads ``<a b>`` with only a warning. Such a name is garbled, not a
    term: it cannot be written into a SPARQL update or read back from Turtle.

    Returns:
        One record per malformed IRI, with the number of triples removed.
    """
    removed: Counter[str] = Counter()
    for triple in list(graph):
        iris = {iri for term in triple if (iri := _malformed_iri(term))}
        if iris:
            graph.remove(triple)
            removed.update(iris)
    if removed:
        logger.warning("Dropped triples naming malformed IRIs: %s", dict(removed))
    return [
        GraphRepairRecord(
            kind=FactsGateRepairKind.MALFORMED_IRI_DROPPED,
            source=iri,
            target="",
            triple_count=count,
        )
        for iri, count in sorted(removed.items())
    ]


def finalize_update_report(
    report: GraphUpdateRenderReport,
    *,
    insert_hook: InsertHook | None = None,
) -> tuple[GraphUpdate, list[RejectedLiteralTriple], list[GraphRepairRecord]]:
    """Turn a rendered patch into an applied-ready update.

    Rebinds prefixes to their canonical namespaces, drops triples naming a
    malformed IRI, drops XSD typed literals the model got wrong (returning them
    for quarantine), and runs the caller's domain repairs.

    ``insert_hook`` never sees the delete side: a delete has to match the stored
    triple verbatim to remove it, so "repairing" a bad literal there would
    silently turn the deletion into a no-op -- and deleting a bad literal is
    exactly what a delete op is for.

    The report is a throwaway parse result and is cleaned in place.

    Args:
        report: Parsed LLM render report carrying the patch.
        insert_hook: Optional domain repair for insert triples.

    Returns:
        The update, the triples withheld from it for quarantine, and the
        records of triples dropped for naming a malformed IRI.
    """
    report.delete_graph.sanitize_prefixes_namespaces()
    report.insert_graph.sanitize_prefixes_namespaces()
    dropped = drop_malformed_iris(report.delete_graph) + drop_malformed_iris(
        report.insert_graph
    )

    report.delete_graph, rejected_all = finalize_llm_graph(report.delete_graph)
    report.insert_graph, rejected_inserts = finalize_llm_graph(report.insert_graph)
    if insert_hook is not None:
        report.insert_graph, hook_rejected = insert_hook(report.insert_graph)
        rejected_inserts = rejected_inserts + hook_rejected

    return report.to_graph_update(), rejected_all + rejected_inserts, dropped


def log_quarantine(kind: str, rejected: list[RejectedLiteralTriple]) -> None:
    """Warn once per render about triples withheld from the applied graph."""
    if rejected:
        logger.warning(
            "%s update quarantined %d triple(s) with invalid literals",
            kind,
            len(rejected),
        )
