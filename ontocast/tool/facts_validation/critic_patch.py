"""Compile critic-proposed fixes into a validated graph patch, with no LLM call.

The loop's invariant is that **every mutation is a compiled, validated
``GraphUpdate``** -- that is what keeps the two-namespace contract, IRI policy
and literal repair in one place. It had drifted into a stricter and less
defensible rule: that every mutation must come from a *render call*. The
consequence was that a critic fix naming the exact triple to drop still cost a
full re-extraction of the unit, so the loop only ever spent that on fixes it
considered blocking and dropped the rest.

A fix that names the statements it removes and supplies the ones it adds is
already a patch. Compiling it here costs nothing and leaves the invariant
intact: the result goes through the same ``GraphUpdate`` the renderer's wire
compiles to.

**How a fix names what it removes matters more than anything else here.** The
original contract asked the critic to retype the offending statement into
``incorrect_value``, and to match it had to reproduce the stored triple exactly
-- same prefix form, same predicate, same literal shape. Across a large corpus
of real critiques that reproduction succeeds a minority of the time for
``REPLACE`` and almost never for a bare ``REMOVE``: the payload arrives as
prose, as a plausible-but-invented IRI, or as a node-shaped quote spanning
several statements with one predicate wrong, which fails the all-present check
as a whole. Authoring *new* content in ``correct_value`` has no such problem,
because nothing has to match.

So the primary path is by id: the critic is shown a numbered graph and cites
:attr:`TripleFix.triple_ids`, which resolve by lookup. Requoting remains as a
fallback for a fix that cites no id, under the same exact-match rule as before.

Still deliberately conservative. A cited id the index never issued sends the
whole fix back rather than acting on the part that resolved; a statement already
gone is not deleted again; and a fix that removes exactly what it re-adds is
recorded as asking for nothing rather than counted as a fix that landed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from rdflib import BNode, Graph
from rdflib.term import Node

from ontocast.onto.model import TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.triple_index import Triple, TripleIndex

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
    #: Fixes whose delete set and insert set are the same statements. They ask
    #: for no change, so they are neither applied nor sent back as work -- but
    #: they are counted, because a critique made mostly of these is a critic
    #: producing motion rather than corrections, and nothing used to see it.
    noop: list[TripleFix] = field(default_factory=list)
    #: Ids cited that the index never issued. Counted, never guessed at.
    bad_index_refs: int = 0
    #: True when the delete-share cap fired and the patch kept only its inserts.
    delete_capped: bool = False
    #: Delete halves withheld by screening, in fixes.
    deletes_refused: int = 0


@dataclass(frozen=True)
class CriticPatchPolicy:
    """How much destruction one critic pass is allowed to do.

    A compiled patch is transparent -- both halves are known before anything is
    touched -- so the limits are enforced by withholding the delete half rather
    than by inspecting the wreckage afterwards. Every rule here drops something
    and counts it; none of them raises.
    """

    #: Largest share of the target graph one pass may remove. A critique that
    #: wants to delete more than this has stopped correcting and started
    #: rewriting, which is the operation the loop deliberately does not offer.
    max_delete_share: float = 0.25
    #: Deletions always permitted regardless of share. Without a floor the cap
    #: is strictest exactly where it should be loosest: on a short unit a single
    #: legitimate correction is already a large fraction of the graph.
    min_deletes: int = 5
    #: Whether a REPLACE may delete statements about one subject while writing
    #: about another. That is a rename, not a correction, and it leaves the new
    #: subject untyped and unlabelled while orphaning the old one.
    allow_subject_rename: bool = False


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


#: A JSON-LD term object that turned up inside an otherwise-Turtle fragment.
_JSONLD_VALUE = re.compile(
    r'\{\s*"@value"\s*:\s*("(?:[^"\\]|\\.)*")\s*'
    r'(?:,\s*"@(?P<kind>type|language)"\s*:\s*"(?P<tag>[^"]+)"\s*)?\}'
)
_JSONLD_ID = re.compile(r'\{\s*"@id"\s*:\s*"([^"]+)"\s*\}')
#: An absolute IRI written without its angle brackets. The lookbehind keeps it
#: off IRIs that are already bracketed or sitting inside a quoted literal.
_BARE_IRI = re.compile(r"""(?<![<"'\w])(https?://[^\s<>"'\[\]{}]+)""")


def _turtleize(body: str) -> str:
    """Rewrite the JSON-LD shapes models mix into a Turtle fragment.

    The deployment's ``llm_graph_format`` names one syntax for fix payloads, and
    the model does not reliably use it -- under a JSON-LD deployment roughly two
    in five authored payloads come back as Turtle, and the ones that fail are
    usually a Turtle body with a JSON-LD term object where the object should be
    (``rdfs:label {"@value": "x", "@language": "en"}``). Neither parser accepts
    that hybrid, so it used to be discarded whole.

    Nothing here guesses at meaning: ``@value``/``@language``/``@type`` map onto
    the Turtle literal forms that mean the same thing, ``@id`` unwraps to the
    IRI it names, and a bare absolute IRI gets the angle brackets Turtle
    requires. A payload that is still not a statement afterwards stays
    unparseable.
    """

    def value(match: re.Match) -> str:
        literal, kind, tag = match.group(1), match.group("kind"), match.group("tag")
        if kind == "language":
            return f"{literal}@{tag}"
        if kind == "type":
            return f"{literal}^^{tag}"
        return literal

    body = _JSONLD_VALUE.sub(value, body)
    body = _JSONLD_ID.sub(lambda match: match.group(1), body)
    return _BARE_IRI.sub(lambda match: "<" + match.group(1).rstrip(".,;") + ">", body)


def _terminator_candidates(body: str) -> list[str]:
    """A fragment lifted out of a Turtle document, with its ending repaired.

    Such a fragment often keeps the trailing predicate-list separator it was cut
    before, or loses its terminator entirely.
    """
    candidates = [body]
    if body.endswith((";", ",")):
        candidates.append(body[:-1] + " .")
    if not body.endswith("."):
        candidates.append(body + " .")
    return candidates


def _parse_fragment(text: str | None, graph: Graph) -> Graph | None:
    """Parse one fix payload, tolerating what models actually emit.

    Deliberately format-tolerant rather than format-dispatched: the payload's
    syntax is whatever the model chose, not what the deployment asked for.

    Args:
        text: ``correct_value`` (or a legacy ``incorrect_value``) as written.
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
        # A JSON body that yields no statement -- a bare literal, or a node
        # reference with no predicates -- is not a patch, whatever it meant.
        return parsed if len(parsed) else None

    header = _prefix_header(graph)
    for candidate in [
        *_terminator_candidates(body),
        *_terminator_candidates(_turtleize(body)),
    ]:
        try:
            parsed = Graph()
            parsed.parse(data=header + candidate, format="turtle")
        except Exception:
            continue
        if len(parsed):
            return parsed
    return None


def _deletes_by_id(
    fix: TripleFix, graph: RDFGraph, index: TripleIndex
) -> tuple[list[Triple] | None, int]:
    """Resolve a fix's cited ids against the index it was shown.

    Returns ``(None, bad)`` when any cited id is unknown: a critique that names
    a statement the index never issued has lost track of the graph, and the
    honest response is to send the whole fix back rather than act on the part
    that happened to resolve.

    Ids are looked up in the index object handed to this call, which maps ids to
    the *terms* they stood for -- so a later renumbering cannot silently
    redirect them. What can still go stale is the graph, hence the membership
    check: a statement already gone is simply not deleted again.
    """
    resolved: list[Triple] = []
    bad = 0
    for triple_id in fix.triple_ids:
        triple = index.resolve(triple_id)
        if triple is None:
            bad += 1
            continue
        if triple in graph:
            resolved.append(triple)
    if bad:
        return None, bad
    return resolved, 0


@dataclass
class _Candidate:
    """One fix reduced to the statements it would remove and add."""

    fix: TripleFix
    deletes: list[Triple] = field(default_factory=list)
    inserts: list[Triple] = field(default_factory=list)


def _bnode_closure(deletes: list[Triple], graph: RDFGraph) -> list[Triple]:
    """Every statement about a blank node the fix touches.

    Deleting part of a blank node leaves the stub the ontology validator flags
    as a degenerate restriction: an ``owl:Restriction`` that constrains nothing,
    still hanging off its ``subClassOf`` edge. A blank node is one object, so it
    goes whole or not at all.
    """
    closure = list(deletes)
    seen = {triple for triple in deletes}
    for _, _, obj in list(deletes):
        if not isinstance(obj, BNode):
            continue
        for triple in graph.triples((obj, None, None)):
            if triple not in seen:
                seen.add(triple)
                closure.append(triple)
    for subject in {s for s, _, _ in deletes if isinstance(s, BNode)}:
        for triple in graph.triples((subject, None, None)):
            if triple not in seen:
                seen.add(triple)
                closure.append(triple)
        for triple in graph.triples((None, None, subject)):
            if triple not in seen:
                seen.add(triple)
                closure.append(triple)
    return closure


def _subject_survives(
    subject: Node,
    graph: RDFGraph,
    deletes: list[Triple],
    inserted_subjects: set[Node],
) -> bool:
    """Whether anything is still said about ``subject`` after the pass."""
    if isinstance(subject, BNode):
        # A blank node has no identity of its own to orphan -- it exists only as
        # the object of the statement that references it, and the closure rule
        # above removes that edge too. Emptying one *is* deleting it, which is
        # the intended outcome, not the accident this check guards against.
        return True
    if subject in inserted_subjects:
        return True
    remaining = [
        triple
        for triple in graph.triples((subject, None, None))
        if triple not in deletes
    ]
    return bool(remaining)


def _screen(
    candidates: list[_Candidate],
    graph: RDFGraph,
    policy: CriticPatchPolicy,
) -> tuple[list[_Candidate], list[TripleFix], int, bool]:
    """Withhold the delete halves the policy does not allow.

    Returns the surviving candidates, the fixes pushed back to residual, how
    many delete halves were refused, and whether the share cap fired.
    """
    refused = 0
    residual: list[TripleFix] = []
    inserted_subjects = {
        subject for candidate in candidates for subject, _, _ in candidate.inserts
    }

    kept: list[_Candidate] = []
    for candidate in candidates:
        if not candidate.deletes:
            kept.append(candidate)
            continue

        candidate.deletes = _bnode_closure(candidate.deletes, graph)
        delete_subjects = {subject for subject, _, _ in candidate.deletes}

        if candidate.fix.action == "REPLACE" and not policy.allow_subject_rename:
            insert_subjects = {subject for subject, _, _ in candidate.inserts}
            if not delete_subjects <= insert_subjects:
                # Writing about a different subject than it removes: the fix's
                # real intent is to mint a new node, which a delete/insert pair
                # cannot express without leaving the new one bare.
                candidate.deletes = []
                refused += 1
                if not candidate.inserts:
                    residual.append(candidate.fix)
                    continue

        if candidate.fix.action == "REMOVE":
            orphaned = [
                subject
                for subject in delete_subjects
                if not _subject_survives(
                    subject, graph, candidate.deletes, inserted_subjects
                )
            ]
            if orphaned:
                # Removing the last statement about a node deletes the node.
                # The repair contract is to correct a statement in place, so a
                # fix that empties a subject is asking for something else and
                # goes back for judgement rather than being carried out.
                candidate.deletes = []
                refused += 1
                residual.append(candidate.fix)
                continue

        kept.append(candidate)

    total_deletes = {triple for candidate in kept for triple in candidate.deletes}
    capped = False
    allowance = max(policy.min_deletes, policy.max_delete_share * len(graph))
    if len(total_deletes) > allowance:
        # Over the cap the inserts are still worth having -- they add rather
        # than destroy -- so only the delete side is dropped.
        capped = True
        for candidate in kept:
            candidate.deletes = []

    return kept, residual, refused, capped


def compile_critic_fixes(
    fixes: Sequence[TripleFix],
    graph: RDFGraph,
    *,
    index: TripleIndex | None = None,
    policy: CriticPatchPolicy | None = None,
) -> CompiledFixes:
    """Split a critique into a mechanical patch and the fixes needing a render.

    Args:
        fixes: Fixes from the critique report, in the order proposed.
        graph: The rendered unit graph the fixes refer to.
        index: The ids handed to the critic for this graph. When present, a fix
            that cites ids is resolved by lookup; the requoting path is the
            fallback for a fix that cites none.
        policy: Limits on what the patch may destroy. ``None`` uses the
            defaults, which are the facts-side ones.

    Returns:
        CompiledFixes: ``update`` is delete-then-insert over the fixes in
        ``applied``; ``residual`` holds what needs judgement, ``noop`` what asks
        for nothing.
    """
    active = policy if policy is not None else CriticPatchPolicy()
    candidates: list[_Candidate] = []
    residual: list[TripleFix] = []
    noop: list[TripleFix] = []
    bad_index_refs = 0

    for fix in fixes:
        correct = _parse_fragment(fix.correct_value, graph)
        insert_triples = (
            [triple for triple in correct if triple not in graph] if correct else []
        )

        matched: list[Triple] = []
        if fix.action in ("REMOVE", "REPLACE"):
            if index is not None and fix.triple_ids:
                by_id, bad = _deletes_by_id(fix, graph, index)
                bad_index_refs += bad
                if by_id is None:
                    residual.append(fix)
                    continue
                matched = by_id
            else:
                # Fallback for a fix that cites no id. The quoted statements must
                # all be present: a misquote has misunderstood the graph, and
                # acting on it would delete something the critic never looked at.
                incorrect = _parse_fragment(fix.incorrect_value, graph)
                quoted = (
                    [triple for triple in incorrect if triple in graph]
                    if incorrect
                    else []
                )
                if not quoted or len(quoted) != len(incorrect or []):
                    residual.append(fix)
                    continue
                matched = quoted

        # A fix that removes exactly what it re-adds changes nothing. Left in
        # `residual` it would ask the next pass to redo nothing; applied, it
        # would inflate the count of fixes that "landed".
        if matched and set(matched) == set(correct or []):
            noop.append(fix)
            continue

        if fix.action == "REMOVE":
            if not matched:
                residual.append(fix)
                continue
            candidates.append(_Candidate(fix=fix, deletes=matched))
        elif fix.action == "ADD":
            if not insert_triples:
                residual.append(fix)
                continue
            candidates.append(_Candidate(fix=fix, inserts=insert_triples))
        elif fix.action == "REPLACE":
            if not matched or not correct:
                residual.append(fix)
                continue
            candidates.append(
                _Candidate(fix=fix, deletes=matched, inserts=list(correct))
            )
        else:
            residual.append(fix)

    kept, screened_out, refused, capped = _screen(candidates, graph, active)
    residual.extend(screened_out)

    deletes = RDFGraph()
    inserts = RDFGraph()
    applied: list[TripleFix] = []
    for candidate in kept:
        if not candidate.deletes and not candidate.inserts:
            residual.append(candidate.fix)
            continue
        for triple in candidate.deletes:
            deletes.add(triple)
        for triple in candidate.inserts:
            inserts.add(triple)
        applied.append(candidate.fix)

    # Delete-then-insert leaves a triple on both sides present either way, so
    # dropping it from the delete side changes nothing and keeps the patch
    # honest about what it removes.
    for triple in inserts:
        deletes.remove(triple)

    if not len(deletes) and not len(inserts):
        return CompiledFixes(
            residual=residual + applied,
            noop=noop,
            bad_index_refs=bad_index_refs,
            delete_capped=capped,
            deletes_refused=refused,
        )

    operations: list[TripleOp] = []
    if len(deletes):
        operations.append(TripleOp(type="delete", graph=deletes))
    if len(inserts):
        operations.append(TripleOp(type="insert", graph=inserts))
    logger.info(
        "Critic fixes: %d compiled to a patch (-%d/+%d triples), "
        "%d need a render, %d asked for no change, %d delete half(s) refused%s",
        len(applied),
        len(deletes),
        len(inserts),
        len(residual),
        len(noop),
        refused,
        " (delete-share cap fired)" if capped else "",
    )
    return CompiledFixes(
        update=GraphUpdate(triple_operations=operations),
        applied=applied,
        residual=residual,
        noop=noop,
        bad_index_refs=bad_index_refs,
        delete_capped=capped,
        deletes_refused=refused,
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
