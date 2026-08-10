"""Best-effort condensing of an ontology graph before it becomes prompt text.

Only the vector-retrieval mode ever bounded how much ontology reached the LLM.
``selected_single_ontology`` (the default) and ``fixed_single_ontology`` serialize
the whole selected catalog ontology into every prompt, and the facts fan-out
serializes the union of every ontology artifact -- all with no cap, no sampling
and no truncation. On a large catalog that is the context blow-up; nothing else
in the pipeline notices.

This module trims a graph toward a triple budget by dropping the
least-load-bearing triples first, in the order established by
:data:`~ontocast.onto.graph_prune.BFS_PREDICATE_PRIORITY`. It is deliberately
**best-effort**: it will never drop labels, types, hierarchy or domain/range to
hit a number. A budget that cannot be met without cutting into those is reported
as a warning and the graph is passed through oversized, because silently
removing the schema the model needed produces a bad extraction that looks like a
bad model -- the most expensive failure mode this pipeline has.
"""

import logging

from pydantic import BaseModel, Field
from rdflib import URIRef

from ontocast.onto.graph_prune import (
    BFS_PREDICATE_PRIORITY,
    NOISY_EXPANSION_PREDICATES,
    prune_degenerate_restriction_bnodes,
    prune_orphaned_bnode_subjects,
    strip_redundant_generic_types,
)
from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)

#: Predicates carrying human-facing description rather than structure. Last tier
#: of the shared priority ordering, and the only tier this module will drop.
GLOSS_PREDICATES: frozenset[URIRef] = BFS_PREDICATE_PRIORITY[-1]

#: Tiers that are never dropped: a term the model cannot name, type, place in the
#: hierarchy, or connect is not usable context -- it is an invitation to invent.
LOAD_BEARING_PREDICATES: frozenset[URIRef] = frozenset().union(
    *BFS_PREDICATE_PRIORITY[:-1]
)


class CondenseReport(BaseModel):
    """What condensing did, for telemetry and for explaining a warning."""

    triples_before: int = Field(description="Triple count on entry")
    triples_after: int = Field(description="Triple count after condensing")
    max_triples: int | None = Field(
        default=None, description="Budget applied; None means no budget"
    )
    dropped_noise: int = Field(default=0, description="Header/list plumbing removed")
    dropped_structural: int = Field(
        default=0, description="Generic types, stub restrictions, orphan bnodes"
    )
    dropped_glosses: int = Field(
        default=0, description="Comments, definitions, scope notes, alt labels"
    )
    over_budget: bool = Field(
        default=False,
        description="Still above budget after condensing; passed through oversized",
    )

    @property
    def changed(self) -> bool:
        """Whether anything was removed at all."""
        return self.triples_after != self.triples_before

    def as_metrics(self) -> dict[str, int | bool | None]:
        """Flat mapping for the retrieval-metrics payload."""
        return {
            "triples_before": self.triples_before,
            "triples_after": self.triples_after,
            "max_triples": self.max_triples,
            "dropped_noise": self.dropped_noise,
            "dropped_structural": self.dropped_structural,
            "dropped_glosses": self.dropped_glosses,
            "over_budget": self.over_budget,
        }


def _drop_predicates(graph: RDFGraph, predicates: frozenset[URIRef]) -> int:
    removed = 0
    for triple in list(graph):
        if triple[1] in predicates:
            graph.remove(triple)
            removed += 1
    return removed


def condense_graph_for_prompt(
    graph: RDFGraph,
    max_triples: int | None,
) -> tuple[RDFGraph, CondenseReport]:
    """Trim ``graph`` toward ``max_triples``, dropping the least useful triples first.

    Passes are applied in increasing order of harm, stopping as soon as the graph
    fits: header/list noise, then structural scaffolding, then glosses. Structure
    that lets the model name and place a term is never dropped.

    Args:
        graph: Ontology graph destined for a prompt. Not mutated.
        max_triples: Triple budget, or ``None`` to disable condensing entirely.

    Returns:
        The condensed graph (or ``graph`` itself when nothing was done) and a
        :class:`CondenseReport` describing what happened.
    """
    before = len(graph)
    report = CondenseReport(
        triples_before=before, triples_after=before, max_triples=max_triples
    )
    if max_triples is None or before <= max_triples:
        return graph, report

    working = graph.copy()

    report.dropped_noise = _drop_predicates(working, NOISY_EXPANSION_PREDICATES)
    if len(working) > max_triples:
        structural_before = len(working)
        strip_redundant_generic_types(working)
        prune_degenerate_restriction_bnodes(working)
        prune_orphaned_bnode_subjects(working)
        report.dropped_structural = structural_before - len(working)

    if len(working) > max_triples:
        report.dropped_glosses = _drop_predicates(working, GLOSS_PREDICATES)

    report.triples_after = len(working)
    report.over_budget = len(working) > max_triples

    if report.over_budget:
        logger.warning(
            "Ontology context still exceeds the prompt budget after condensing "
            "(%d > %d triples; was %d). Passing it through rather than dropping "
            "labels, types, hierarchy or domain/range, which the model needs to "
            "use the schema at all. Reduce the catalog, split it, or switch to "
            "ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology, which "
            "retrieves only the relevant part.",
            report.triples_after,
            max_triples,
            before,
        )
    else:
        logger.info(
            "Condensed ontology context %d -> %d triples for a %d budget "
            "(noise=%d, structural=%d, glosses=%d).",
            before,
            report.triples_after,
            max_triples,
            report.dropped_noise,
            report.dropped_structural,
            report.dropped_glosses,
        )

    return working, report
