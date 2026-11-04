"""Coreference candidates for the model to resolve, and what its verdicts do.

Units render in parallel and never see each other's entities, so whether two
mentions from different units denote one thing is decided after aggregation.
The symbolic tiers settle most pairs; the ones they cannot settle are put to
the model, one call per document:

* ``unconfirmed_merge`` -- merged on a plausible lexical tier only (label
  overlap, string similarity, an IRI-derived name), never on an equal label,
  initials, a shared key or a confirmation. A ``different`` verdict splits it.
* ``near_miss`` -- similar enough to be a candidate, refused for ``lexical``
  alone, with no guard saying the two are distinct. A ``same`` verdict merges.
* ``shared_link`` -- two entities of one type from different units, the only
  two that point to one node through one predicate (both "applicant in" one
  case), with no name or embedding evidence: a role phrase and a name for one
  referent.

A verdict is evidence, not a decision: ``same`` merges like a shared key and
``different`` vetoes, and every distinctness guard still applies.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import combinations, zip_longest

from rdflib import OWL, RDF, RDFS, URIRef
from rdflib import Literal as RDFLiteral
from rdflib.term import Node

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.model import CoreferenceCandidate
from ontocast.onto.scaffolding import is_scaffolding, scaffolding_nodes

from .unit_scope import strip_unit_scope, unit_scope_index

#: Predicates that link nothing for the shared-link candidates.
_NON_LINKS: frozenset[URIRef] = frozenset(
    {RDF.type, OWL.sameAs, RDFS.label, RDFS.comment, RDFS.seeAlso}
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_VALUES = 5
_MAX_RELATIONS = 6
_MAX_TEXT = 160


def _local(term: Node) -> str:
    _, local = split_namespace_local(str(term))
    return strip_unit_scope(local or str(term))


def _unit_of(iri: str, units: Sequence[ContentUnit]) -> ContentUnit | None:
    index = unit_scope_index(iri)
    return next((unit for unit in units if unit.index == index), None)


def _shared_link_candidates(
    units: Sequence[ContentUnit], final_uri: dict[str, str]
) -> list[CoreferenceCandidate]:
    """Same-type entities of different units, the only two pointing to one node
    by one predicate."""
    pointing: dict[tuple[str, str], set[str]] = {}
    types: dict[str, frozenset[str]] = {}
    for unit in units:
        graph = unit.graph
        if graph is None:
            continue
        scaffolding = scaffolding_nodes(graph)
        for subject, predicate, obj in graph:
            if (
                not isinstance(subject, URIRef)
                or unit_scope_index(str(subject)) is None
            ):
                continue
            if predicate == RDF.type and isinstance(obj, URIRef):
                types[str(subject)] = types.get(str(subject), frozenset()) | {str(obj)}
                continue
            if (
                predicate in _NON_LINKS
                or not isinstance(obj, URIRef)
                or is_scaffolding((subject, predicate, obj), scaffolding)
            ):
                continue
            target = final_uri.get(str(obj), str(obj))
            pointing.setdefault((str(predicate), target), set()).add(str(subject))
    seen: set[frozenset[str]] = set()
    candidates: list[CoreferenceCandidate] = []
    for key in sorted(pointing):
        # Only a link exactly two entities hold is specific: a hub many point
        # to (one place, one analysis) says nothing about identity.
        if len(pointing[key]) != 2:
            continue
        for left, right in combinations(sorted(pointing[key]), 2):
            if unit_scope_index(left) == unit_scope_index(right):
                continue
            left_final = final_uri.get(left, left)
            right_final = final_uri.get(right, right)
            finals = frozenset((left_final, right_final))
            if left_final == right_final or finals in seen:
                continue
            if not types.get(left) or types.get(left) != types.get(right):
                continue
            seen.add(finals)
            candidates.append(
                CoreferenceCandidate(left=left, right=right, kind="shared_link")
            )
    return candidates


def coreference_candidates(
    *,
    unconfirmed_merges: Sequence[tuple[str, str, float]],
    lexical_near_misses: Sequence[tuple[str, str, float]],
    units: Sequence[ContentUnit],
    final_uri: dict[str, str],
    max_pairs: int,
) -> list[CoreferenceCandidate]:
    """The pairs to put to the model, at most ``max_pairs``.

    Only entities units minted are asked about; catalog terms are not. The
    three kinds take turns, so none crowds the others out: the least similar
    unconfirmed merges first (the likeliest wrong), the most similar near
    misses first (the likeliest right), then shared links.

    Args:
        unconfirmed_merges: ``AggregationResult.unconfirmed_merges``.
        lexical_near_misses: ``AggregationResult.lexical_near_misses``.
        units: The aggregated units, their graphs unit-scoped.
        final_uri: Source entity IRI -> its final IRI after aggregation.
        max_pairs: Cap on the pairs returned.
    """
    merges = [
        CoreferenceCandidate(
            left=left, right=right, kind="unconfirmed_merge", similarity=score
        )
        for left, right, score in sorted(unconfirmed_merges, key=lambda m: m[2])
    ]
    near = [
        CoreferenceCandidate(left=left, right=right, kind="near_miss", similarity=score)
        for left, right, score in sorted(lexical_near_misses, key=lambda m: -m[2])
    ]
    links = _shared_link_candidates(units, final_uri)

    def minted(candidates: list[CoreferenceCandidate]) -> list[CoreferenceCandidate]:
        return [
            c
            for c in candidates
            if unit_scope_index(c.left) is not None
            and unit_scope_index(c.right) is not None
        ]

    asked: set[frozenset[str]] = set()
    out: list[CoreferenceCandidate] = []
    for round_ in zip_longest(minted(merges), minted(near), links):
        for candidate in round_:
            if candidate is None or len(out) >= max_pairs:
                continue
            pair = frozenset((candidate.left, candidate.right))
            if pair in asked:
                continue
            asked.add(pair)
            out.append(candidate)
    return out


def describe_entity(iri: str, units: Sequence[ContentUnit]) -> str:
    """What the model sees of one entity: names, types, values, links, a sentence.

    Read from the graph of the unit that minted it, so each side of a pair is
    shown as its own unit stated it.
    """
    unit = _unit_of(iri, units)
    if unit is None or unit.graph is None:
        return f"id: {_local(URIRef(iri))}"
    graph = unit.graph
    node = URIRef(iri)
    labels = sorted(str(o) for o in graph.objects(node, RDFS.label))
    types = sorted(_local(o) for o in graph.objects(node, RDF.type))
    values: list[str] = []
    relations: list[str] = []
    for predicate, obj in sorted(graph.predicate_objects(node), key=str):
        if predicate in (RDF.type, RDFS.label):
            continue
        if isinstance(obj, RDFLiteral):
            if len(values) < _MAX_VALUES:
                values.append(f"{_local(predicate)} = {str(obj)[:60]}")
        elif isinstance(obj, URIRef) and len(relations) < _MAX_RELATIONS:
            name = next(
                iter(sorted(str(o) for o in graph.objects(obj, RDFS.label))), ""
            )
            relations.append(f"{_local(predicate)} -> {name or _local(obj)}")
    parts = [f"id: {_local(node)}"]
    if labels:
        parts.append("labels: " + "; ".join(f'"{label}"' for label in labels))
    if types:
        parts.append("types: " + ", ".join(types))
    if values:
        parts.append("values: " + "; ".join(values))
    if relations:
        parts.append("relations: " + "; ".join(relations))
    sentence = _source_sentence(unit.text, labels)
    if sentence:
        parts.append(f'source: "{sentence}"')
    return "\n    ".join(parts)


def _source_sentence(text: str, labels: Sequence[str]) -> str:
    """The first sentence of ``text`` naming one of ``labels``, shortened."""
    lowered = [label.lower() for label in labels if label.strip()]
    if not lowered:
        return ""
    for sentence in _SENTENCE_RE.split(text):
        if any(label in sentence.lower() for label in lowered):
            flat = " ".join(sentence.split())
            return flat if len(flat) <= _MAX_TEXT else f"{flat[: _MAX_TEXT - 1]}…"
    return ""


def verdict_pairs(
    candidates: Sequence[CoreferenceCandidate],
    verdicts: dict[int, str],
) -> tuple[set[frozenset[URIRef]], set[frozenset[URIRef]]]:
    """``(vetoes, confirmations)`` from verdicts keyed by 1-based pair number.

    ``same`` confirms a pair, ``different`` vetoes it; ``unsure`` and pairs
    with no verdict change nothing.
    """
    vetoes: set[frozenset[URIRef]] = set()
    confirmations: set[frozenset[URIRef]] = set()
    for number, candidate in enumerate(candidates, start=1):
        pair = frozenset((URIRef(candidate.left), URIRef(candidate.right)))
        verdict = verdicts.get(number)
        if verdict == "same":
            confirmations.add(pair)
        elif verdict == "different":
            vetoes.add(pair)
    return vetoes, confirmations


__all__ = [
    "CoreferenceCandidate",
    "coreference_candidates",
    "describe_entity",
    "verdict_pairs",
]
