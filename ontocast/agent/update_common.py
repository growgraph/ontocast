"""Shared post-parse mechanism for the two graph-update render agents.

``agent/common.py`` is about *calling* the LLM; this module is about what
happens to the patch once it has been parsed. Facts updates
(:func:`ontocast.agent.render_facts.render_facts_update`) and ontology updates
(:func:`ontocast.agent.render_ontology.render_ontology_update`) consume the same
:class:`GraphUpdateRenderReport`, so the hygiene between "the model answered"
and "the update is applied" belongs in one place rather than being reimplemented
per domain -- the ontology side previously had none of it at all.
"""

import logging
from collections.abc import Callable

from ontocast.onto.model import GraphUpdateRenderReport
from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple, finalize_llm_graph
from ontocast.onto.sparql_models import GraphUpdate

logger = logging.getLogger(__name__)

#: Domain-specific repair applied to the INSERT side of a patch only. Returns
#: the repaired graph plus any triples it withheld from the applied graph.
InsertHook = Callable[[RDFGraph], tuple[RDFGraph, list[RejectedLiteralTriple]]]


def finalize_update_report(
    report: GraphUpdateRenderReport,
    *,
    insert_hook: InsertHook | None = None,
) -> tuple[GraphUpdate, list[RejectedLiteralTriple]]:
    """Turn a rendered patch into an applied-ready update.

    Rebinds prefixes to their canonical namespaces, drops XSD typed literals the
    model got wrong (returning them for quarantine), and runs the caller's
    domain repairs.

    ``insert_hook`` never sees the delete side: a delete has to match the stored
    triple verbatim to remove it, so "repairing" a bad literal there would
    silently turn the deletion into a no-op -- and deleting a bad literal is
    exactly what a delete op is for.

    The report is a throwaway parse result and is cleaned in place.

    Args:
        report: Parsed LLM render report carrying the patch.
        insert_hook: Optional domain repair for insert triples.

    Returns:
        The update, and the triples withheld from it.
    """
    report.delete_graph.sanitize_prefixes_namespaces()
    report.insert_graph.sanitize_prefixes_namespaces()

    report.delete_graph, rejected_all = finalize_llm_graph(report.delete_graph)
    report.insert_graph, rejected_inserts = finalize_llm_graph(report.insert_graph)
    if insert_hook is not None:
        report.insert_graph, hook_rejected = insert_hook(report.insert_graph)
        rejected_inserts = rejected_inserts + hook_rejected

    return report.to_graph_update(), rejected_all + rejected_inserts


def log_quarantine(kind: str, rejected: list[RejectedLiteralTriple]) -> None:
    """Warn once per render about triples withheld from the applied graph."""
    if rejected:
        logger.warning(
            "%s update quarantined %d triple(s) with invalid literals",
            kind,
            len(rejected),
        )
