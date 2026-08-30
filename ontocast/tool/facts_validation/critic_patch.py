"""Compile critic-proposed fixes into a validated graph patch, with no LLM call.

The loop's invariant is that **every mutation is a compiled, validated
``GraphUpdate``** -- that is what keeps the two-namespace contract, IRI policy
and literal repair in one place. It had drifted into a stricter and less
defensible rule: that every mutation must come from a *render call*. The
consequence was that a critic fix naming the exact triple to drop still cost a
full re-extraction of the unit, so the loop only ever spent that on fixes it
considered blocking and dropped the rest.

A fix whose ``incorrect_value`` matches triples that are actually in the graph,
or whose ``correct_value`` parses, is already a patch. Compiling it here costs
nothing and leaves the invariant intact: the result goes through the same
``GraphUpdate`` the renderer's wire compiles to.

Deliberately conservative. A fix compiles only when it parses *and*, for
REMOVE/REPLACE, its ``incorrect_value`` matches triples present in the graph
exactly. Anything else -- a truncated payload, an invented prefix, a
description rather than a triple -- is returned as residual for the scoped
repair render, which is where the model's judgement is actually needed. Real
critic output makes that split matter: ADD payloads parse most of the time,
REPLACE payloads much less often.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from rdflib import Graph

from ontocast.onto.model import TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledFixes:
    """The mechanical half of a critique, split from the half needing a render."""

    #: Delete-then-insert patch, or ``None`` when nothing compiled.
    update: GraphUpdate | None = None
    #: Fixes folded into ``update``.
    applied: list[TripleFix] = field(default_factory=list)
    #: Fixes that need a scoped repair render.
    residual: list[TripleFix] = field(default_factory=list)


def _prefix_header(graph: Graph) -> str:
    """Turtle ``@prefix`` lines for every binding the unit graph carries.

    Critic fixes are written as fragments in the prompt's vocabulary and
    usually omit their prefix declarations, so a bare fragment will not parse
    on its own. Supplying the unit's own bindings is what makes the common case
    readable without inventing namespaces the graph never used.
    """
    lines = []
    for prefix, uri in graph.namespaces():
        if prefix:
            lines.append(f"@prefix {prefix}: <{uri}> .")
    return "\n".join(lines) + "\n"


def _parse_fragment(text: str | None, graph: Graph) -> Graph | None:
    """Parse one fix payload, tolerating the truncations models emit.

    Args:
        text: ``incorrect_value`` or ``correct_value`` as written.
        graph: The unit graph, for prefix bindings.

    Returns:
        Graph | None: The parsed triples, or ``None`` if unparseable/empty.
    """
    body = (text or "").strip()
    if not body:
        return None
    if body[0] in "{[":
        try:
            parsed = Graph()
            parsed.parse(data=body, format="json-ld")
        except Exception:
            return None
        return parsed if len(parsed) else None
    # A fragment lifted out of a Turtle document often keeps its trailing
    # predicate-list separator, or loses its terminator entirely.
    candidates = [body]
    if body.endswith((";", ",")):
        candidates.append(body[:-1] + " .")
    if not body.endswith("."):
        candidates.append(body + " .")
    header = _prefix_header(graph)
    for candidate in candidates:
        try:
            parsed = Graph()
            parsed.parse(data=header + candidate, format="turtle")
        except Exception:
            continue
        if len(parsed):
            return parsed
    return None


def compile_critic_fixes(fixes: Sequence[TripleFix], graph: RDFGraph) -> CompiledFixes:
    """Split a critique into a mechanical patch and the fixes needing a render.

    Args:
        fixes: Fixes from the critique report, in the order proposed.
        graph: The rendered unit graph the fixes refer to.

    Returns:
        CompiledFixes: ``update`` is delete-then-insert over the fixes in
        ``applied``; ``residual`` holds the rest, unchanged and unordered
        relative to each other.
    """
    deletes = RDFGraph()
    inserts = RDFGraph()
    applied: list[TripleFix] = []
    residual: list[TripleFix] = []

    for fix in fixes:
        incorrect = _parse_fragment(fix.incorrect_value, graph)
        correct = _parse_fragment(fix.correct_value, graph)
        # Only triples the graph actually contains may be deleted: a fix that
        # misquotes what it is correcting has misunderstood the graph, and
        # acting on it would delete something the critic never looked at.
        matched = (
            [triple for triple in incorrect if triple in graph] if incorrect else []
        )

        if fix.action == "REMOVE":
            if not matched or len(matched) != len(incorrect or []):
                residual.append(fix)
                continue
            for triple in matched:
                deletes.add(triple)
        elif fix.action == "ADD":
            novel = (
                [triple for triple in correct if triple not in graph] if correct else []
            )
            if not novel:
                residual.append(fix)
                continue
            for triple in novel:
                inserts.add(triple)
        elif fix.action == "REPLACE":
            if not matched or len(matched) != len(incorrect or []) or not correct:
                residual.append(fix)
                continue
            for triple in matched:
                deletes.add(triple)
            for triple in correct:
                inserts.add(triple)
        else:
            residual.append(fix)
            continue
        applied.append(fix)

    if not len(deletes) and not len(inserts):
        return CompiledFixes(residual=list(fixes))

    operations: list[TripleOp] = []
    if len(deletes):
        operations.append(TripleOp(type="delete", graph=deletes))
    if len(inserts):
        operations.append(TripleOp(type="insert", graph=inserts))
    logger.info(
        "Critic fixes: %d compiled to a patch (-%d/+%d triples), %d need a render",
        len(applied),
        len(deletes),
        len(inserts),
        len(residual),
    )
    return CompiledFixes(
        update=GraphUpdate(triple_operations=operations),
        applied=applied,
        residual=residual,
    )


def apply_compiled_patch(graph: RDFGraph, update: GraphUpdate) -> None:
    """Apply a compiled patch to ``graph`` in place, deletes before inserts.

    The ordering is the one :meth:`GraphUpdateRenderReport.to_graph_update`
    fixes, and the operations are the same ``TripleOp``s a render produces --
    this is the render's apply step over an in-memory graph, not a second way
    to mutate one.

    Args:
        graph: The unit graph to patch.
        update: The compiled patch.
    """
    for operation in update.triple_operations:
        if operation.type == "delete":
            for triple in operation.graph:
                graph.remove(triple)
        else:
            for triple in operation.graph:
                graph.add(triple)
