"""Render a graph so every statement carries its :class:`TripleIndex` id.

The ids themselves live in :mod:`ontocast.onto.triple_index`; only the prompt
surface is here, because how the ids are *shown* depends on the deployment's
graph format while what they mean does not.
"""

from __future__ import annotations

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.triple_index import RDF_TYPE, TripleIndex, format_term


def render_indexed_turtle(graph: RDFGraph, index: TripleIndex) -> str:
    """Render the graph as Turtle-shaped blocks with an inline ``[n]`` per statement.

    Turtle can carry the id inline for free, which puts it where the critic is
    already reading. JSON-LD cannot -- its output contract demands strictly
    valid JSON -- so that format gets :func:`render_index_table` instead.

    A statement outside the index's scope is listed with a ``[-]`` marker: it is
    context to read, not something the critic may cite or remove.
    """
    nsmgr = graph.namespace_manager
    lines: list[str] = []
    for subject, block in index.order:
        lines.append(format_term(subject, nsmgr))
        for position, entry in enumerate(block):
            _, predicate, obj = entry.triple
            predicate_text = (
                "a" if predicate == RDF_TYPE else format_term(predicate, nsmgr)
            )
            terminator = " ." if position == len(block) - 1 else " ;"
            marker = "-" if entry.triple_id is None else str(entry.triple_id)
            lines.append(
                f"  [{marker}] {predicate_text} {format_term(obj, nsmgr)}{terminator}"
            )
    return "\n".join(lines)


def render_index_table(graph: RDFGraph, index: TripleIndex) -> str:
    """Render the id table appended to a chapter whose body cannot carry ids.

    Only addressable statements appear: an entry the critic cannot cite would be
    an invitation to cite it.
    """
    nsmgr = graph.namespace_manager
    lines = ["TRIPLE INDEX (id | subject | predicate | object)"]
    for triple_id in sorted(index.by_id):
        subject, predicate, obj = index.by_id[triple_id]
        lines.append(
            f"  {triple_id} | {format_term(subject, nsmgr)} | "
            f"{format_term(predicate, nsmgr)} | {format_term(obj, nsmgr)}"
        )
    return "\n".join(lines)
