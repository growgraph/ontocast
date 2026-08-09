"""Detect which document-type schema a document belongs to.

Section labels are only meaningful relative to a schema: a 10-Q scored against
the academic schema recognises almost nothing and comes back unlabeled. When the
caller supplies neither ``section_schema_id`` nor a matching
``document_type_hint``, the schema has to be inferred from the document itself.

The schema catalog is a **partition** -- every document belongs to exactly one
cell, with ``general`` as the residual. Only schemas carrying a
``document_profile`` are detection candidates; ``general`` deliberately has none,
because six of its seven labels are strict subsets of other schemas.

Three tiers, cheapest first, each seeing only what the previous one could not
decide:

1. **Lexical** (free, no model). Scores on *exclusive* evidence: a heading
   recognised by exactly one candidate schema counts, a heading recognised by
   several counts nothing. A shared heading such as ``References`` genuinely
   says nothing about which cell the document is in, so weighting it -- even
   fractionally -- only adds noise. Measured over the corpus in
   ``test/data/schema_corpus.json`` this classifies all nine cells correctly,
   with the tightest margin (a published trial protocol, which shares IMRaD
   headings with academic papers) at 2.0x.
2. **Semantic** (needs the chunker's embedding model). Each heading votes for
   its nearest schema by label-name similarity, damped by how close the
   runner-up is. Per-heading competition rather than per-schema mean similarity,
   because mean similarity is biased by how many label names a schema happens to
   have.
3. **Content** (last resort, heading-poor documents only). Body paragraphs
   against ``document_profile`` sentences. Measured accuracy is far below the
   heading tiers -- chemistry prose reads like a technical specification, and
   ``news`` behaves as a semantic attractor -- so it demands a much larger
   margin and is off unless explicitly enabled.

Every tier abstains rather than guessing. A wrong schema relabels an entire
document silently, whereas an abstention falls back to the manifest default and
leaves ``document_type_hint`` in charge.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Protocol

from ontocast.config.section_labels import (
    load_manifest,
    load_section_label_schema,
    normalise_heading_line,
    resolve_heading_label,
)
from ontocast.tool.chunk.outline import heading_is_sectionlike

logger = logging.getLogger(__name__)

# Below this many section-like headings the genericity filter is doing more harm
# than good -- product manuals and how-to guides use descriptive headings such
# as "Setting Up a Simple Proxy Server" that it rejects -- so fall back to
# scoring every heading.
MIN_SECTIONLIKE_HEADINGS = 6

# A document with fewer headings than this cannot be judged from headings at
# all, and is the only case where the content tier is allowed to run.
MAX_HEADINGS_FOR_CONTENT_TIER = 3

# Similarity gap at which a heading casts a full semantic vote.
SEMANTIC_VOTE_TAU = 0.05

_MAX_EXAMPLES = 5


class TextEmbedder(Protocol):
    """Embeds short texts, or returns ``None`` when no model is available."""

    def __call__(self, texts: list[str]) -> list[list[float]] | None: ...


@dataclass(frozen=True)
class SchemaEvidence:
    """Why one candidate schema scored what it did."""

    schema_id: str
    score: float
    share: float
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaDetection:
    """An accepted detection, with the tier that made it and its evidence."""

    schema_id: str
    tier: str
    score: float
    margin: float
    share: float
    evidence: list[SchemaEvidence]


def candidate_schema_ids() -> tuple[str, ...]:
    """Schemas eligible to be detected, in manifest order.

    A schema is a candidate only if it declares a ``document_profile``. That is
    what keeps ``general`` -- the partition's residual cell -- from ever being a
    positive detection.
    """
    ids: list[str] = []
    for entry in load_manifest().schemas:
        if load_section_label_schema(entry.id).document_profile.strip():
            ids.append(entry.id)
    return tuple(ids)


def _voting_headings(headings: list[str]) -> list[str]:
    """Headings that get to vote, relaxing the genericity filter when too strict."""
    generic = [
        heading
        for heading in headings
        if heading_is_sectionlike(normalise_heading_line(heading))
    ]
    if len(generic) >= MIN_SECTIONLIKE_HEADINGS:
        return generic
    return list(headings)


def _rank(
    scores: dict[str, float],
    examples: dict[str, list[str]],
    total: int,
) -> list[SchemaEvidence]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        SchemaEvidence(
            schema_id=schema_id,
            score=score,
            share=score / total if total else 0.0,
            examples=examples.get(schema_id, [])[:_MAX_EXAMPLES],
        )
        for schema_id, score in ranked
    ]


def score_headings_lexical(headings: list[str]) -> list[SchemaEvidence]:
    """Score candidate schemas on headings only one of them recognises.

    Args:
        headings: Raw heading lines in document order.

    Returns:
        Evidence per candidate schema, strongest first.
    """
    candidates = candidate_schema_ids()
    schemas = {sid: load_section_label_schema(sid) for sid in candidates}
    voting = _voting_headings(headings)
    scores = {sid: 0.0 for sid in candidates}
    examples: dict[str, list[str]] = {sid: [] for sid in candidates}

    for heading in voting:
        matched = {
            sid
            for sid in candidates
            if resolve_heading_label(heading, schemas[sid]) is not None
        }
        if len(matched) != 1:
            continue
        schema_id = next(iter(matched))
        scores[schema_id] += 1.0
        examples[schema_id].append(normalise_heading_line(heading))
    return _rank(scores, examples, len(voting))


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _label_prototypes(schema_id: str) -> list[str]:
    """Human-readable label names for a schema, used as semantic prototypes.

    Label ids only, deliberately: adding patterns and keywords makes a schema's
    prototype set larger and biases similarity towards it.
    """
    schema = load_section_label_schema(schema_id)
    return [label.id.replace("_", " ") for label in schema.labels]


def score_headings_semantic(
    headings: list[str], embed: TextEmbedder
) -> list[SchemaEvidence]:
    """Score candidate schemas by heading-to-label-name similarity.

    Each heading casts a single vote for its nearest schema, scaled by how far
    that schema beat the runner-up, so a heading that is ambiguous between two
    schemas contributes almost nothing.
    """
    candidates = candidate_schema_ids()
    voting = _voting_headings(headings)
    if not voting:
        return _rank({sid: 0.0 for sid in candidates}, {}, 0)

    prototypes: dict[str, list[str]] = {
        sid: _label_prototypes(sid) for sid in candidates
    }
    flat = [text for sid in candidates for text in prototypes[sid]]
    embedded_prototypes = embed(flat)
    embedded_headings = embed(voting)
    if embedded_prototypes is None or embedded_headings is None:
        return _rank({sid: 0.0 for sid in candidates}, {}, 0)

    by_schema: dict[str, list[list[float]]] = {}
    cursor = 0
    for sid in candidates:
        size = len(prototypes[sid])
        by_schema[sid] = [
            _normalise(vector) for vector in embedded_prototypes[cursor : cursor + size]
        ]
        cursor += size

    scores = {sid: 0.0 for sid in candidates}
    examples: dict[str, list[str]] = {sid: [] for sid in candidates}
    for heading, raw in zip(voting, embedded_headings):
        vector = _normalise(raw)
        best_per_schema = [
            (max(_cosine(vector, proto) for proto in by_schema[sid]), sid)
            for sid in candidates
            if by_schema[sid]
        ]
        if len(best_per_schema) < 2:
            continue
        best_per_schema.sort(reverse=True)
        (best, winner), (second, _) = best_per_schema[0], best_per_schema[1]
        weight = min(1.0, max(0.0, (best - second) / SEMANTIC_VOTE_TAU))
        if weight <= 0.0:
            continue
        scores[winner] += weight
        examples[winner].append(normalise_heading_line(heading))
    return _rank(scores, examples, len(voting))


def score_content(paragraphs: list[str], embed: TextEmbedder) -> list[SchemaEvidence]:
    """Score candidate schemas by body prose against their document profiles.

    Weak by construction -- see the module docstring -- and only meaningful for
    documents with essentially no headings.
    """
    candidates = [
        sid
        for sid in candidate_schema_ids()
        # News is a measured semantic attractor: any front matter drifts to it.
        if sid != "news"
    ]
    if not paragraphs or not candidates:
        return _rank({sid: 0.0 for sid in candidates}, {}, 0)

    profiles = embed(
        [load_section_label_schema(sid).document_profile.strip() for sid in candidates]
    )
    embedded = embed(paragraphs)
    if profiles is None or embedded is None:
        return _rank({sid: 0.0 for sid in candidates}, {}, 0)

    profile_vectors = [_normalise(vector) for vector in profiles]
    scores = {sid: 0.0 for sid in candidates}
    examples: dict[str, list[str]] = {sid: [] for sid in candidates}
    for paragraph, raw in zip(paragraphs, embedded):
        vector = _normalise(raw)
        ranked = sorted(
            (
                (_cosine(vector, profile_vectors[i]), sid)
                for i, sid in enumerate(candidates)
            ),
            reverse=True,
        )
        (best, winner), (second, _) = ranked[0], ranked[1]
        weight = min(1.0, max(0.0, (best - second) / SEMANTIC_VOTE_TAU))
        if weight <= 0.0:
            continue
        scores[winner] += weight
        examples[winner].append(" ".join(paragraph.split())[:60])
    return _rank(scores, examples, len(paragraphs))


def _accept(
    evidence: list[SchemaEvidence],
    tier: str,
    *,
    min_score: float,
    min_margin: float,
    min_share: float = 0.0,
) -> SchemaDetection | None:
    if len(evidence) < 2:
        return None
    best, runner_up = evidence[0], evidence[1]
    if best.score < min_score:
        return None
    if best.share < min_share:
        return None
    margin = best.score / runner_up.score if runner_up.score > 0 else math.inf
    if margin < min_margin:
        return None
    return SchemaDetection(
        schema_id=best.schema_id,
        tier=tier,
        score=best.score,
        margin=margin,
        share=best.share,
        evidence=evidence,
    )


def detect_document_schema(
    headings: list[str],
    paragraphs: list[str] | None = None,
    *,
    embed: TextEmbedder | None = None,
    allow_content_tier: bool = False,
    min_score: float = 2.0,
    min_margin: float = 1.8,
    min_share: float = 0.0,
    content_min_margin: float = 4.0,
) -> SchemaDetection | None:
    """Infer the document-type schema, or ``None`` when the evidence is thin.

    Args:
        headings: Raw heading lines in document order.
        paragraphs: Sampled body paragraphs, for the content tier.
        embed: Embedding callable; ``None`` restricts detection to the lexical
            tier, exactly as missing semantic extras do.
        allow_content_tier: Permit the weak content tier on heading-poor
            documents.
        min_score: Absolute evidence the winner must clear.
        min_margin: Factor by which the winner must beat the runner-up.
        min_share: Minimum fraction of voting items backing the winner.
        content_min_margin: Stricter margin for the content tier.

    Returns:
        The accepted detection, or ``None`` to fall back to the caller's
        default.
    """
    lexical = score_headings_lexical(headings)
    detection = _accept(
        lexical,
        "lexical",
        min_score=min_score,
        min_margin=min_margin,
        min_share=min_share,
    )
    if detection is not None:
        return detection

    if embed is not None and headings:
        semantic = score_headings_semantic(headings, embed)
        detection = _accept(
            semantic,
            "semantic",
            min_score=min_score,
            min_margin=min_margin,
            min_share=min_share,
        )
        if detection is not None:
            return detection

    if (
        allow_content_tier
        and embed is not None
        and paragraphs
        and len(headings) <= MAX_HEADINGS_FOR_CONTENT_TIER
    ):
        content = score_content(paragraphs, embed)
        return _accept(
            content, "content", min_score=min_score, min_margin=content_min_margin
        )
    return None


__all__ = [
    "MAX_HEADINGS_FOR_CONTENT_TIER",
    "MIN_SECTIONLIKE_HEADINGS",
    "SchemaDetection",
    "SchemaEvidence",
    "candidate_schema_ids",
    "detect_document_schema",
    "score_content",
    "score_headings_lexical",
    "score_headings_semantic",
]
