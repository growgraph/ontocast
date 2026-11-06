"""Section label assignment: every tier adds evidence, one rule decides.

A segment collects one :class:`~ontocast.onto.section_models.LabelEvidence` per
tier that speaks: its section's heading (shared by every segment of the
section), its own content, and the model. The label is the strongest piece of
evidence; between equals the heading's words win, then the content, then the
model, then a heading's nearest label by embedding. Strength is set where the evidence arises and never
inferred later:

- strong: an anchored heading pattern (or a heading that is exactly one of a
  label's keywords); a heading the tree places inside a strong section by its
  number or level; a reference list or acknowledgements recognised from the
  content; a citation run cut out by the block classifier;
- weak: a keyword inside a longer heading; a heading named by its nearest
  label in embedding space; a heading that inherits by its wording alone, or
  through the tree from a weak parent; front matter recognised by position;
  the content tier's guesses at methods, results or introduction; the model.

Only a strong label takes part in a drop decision, so an override by strong
content over a weak heading is what lets a block of thanks under an inherited
``results`` label be recognised as acknowledgements.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol

from ontocast.onto.enum import LabelStrength, SectionLabelSource
from ontocast.onto.section_models import LabelEvidence
from ontocast.tool.chunk.density import CONSERVATIVE_LABELS, classify_by_density

if TYPE_CHECKING:
    from ontocast.config.section_labels import SectionLabelSchema
    from ontocast.tool.chunk.structure import BlockKinds

# Tie-break among evidence of equal strength, by what the tier saw: a heading's
# words first, then content, then the model (heading, opening text and the
# whole outline), then a heading's nearest label by embedding (a few words).
_PRIORITY: tuple[SectionLabelSource, ...] = (
    SectionLabelSource.HEADING_PATTERN,
    SectionLabelSource.HEADING_INHERITED,
    SectionLabelSource.HEADING_KEYWORD,
    SectionLabelSource.FRONT_MATTER,
    SectionLabelSource.CONTENT_BLOCK,
    SectionLabelSource.CONTENT_DENSITY,
    SectionLabelSource.LLM,
    SectionLabelSource.HEADING_EMBEDDING,
)


def _rank(evidence: LabelEvidence) -> tuple[bool, int]:
    return evidence.strength is not LabelStrength.STRONG, _PRIORITY.index(
        evidence.source
    )


def decide(evidence: Sequence[LabelEvidence]) -> LabelEvidence | None:
    """The deciding piece of evidence: strongest first, then by tier.

    Args:
        evidence: Every tier's vote on one segment.

    Returns:
        The winning vote, or ``None`` when no tier spoke.
    """
    return min(evidence, key=_rank, default=None)


def content_evidence(
    text: str,
    schema: SectionLabelSchema,
    *,
    aggressive: bool,
    block_kinds: BlockKinds | None = None,
) -> LabelEvidence | None:
    """The content tier's vote on a segment, or ``None`` when it abstains.

    Only the labels the conservative tier can produce (reference lists and
    acknowledgements, whose form is near-unique) are strong; the aggressive
    tier's guesses are weak.
    """
    result = classify_by_density(
        text, schema, aggressive=aggressive, block_kinds=block_kinds
    )
    if result is None:
        return None
    label = result[0]
    return LabelEvidence(
        label=label,
        source=SectionLabelSource.CONTENT_DENSITY,
        strength=(
            LabelStrength.STRONG if label in CONSERVATIVE_LABELS else LabelStrength.WEAK
        ),
    )


class Assignable(Protocol):
    """A segment whose label is decided from its evidence."""

    evidence: list[LabelEvidence]
    section_label: str | None
    section_label_source: SectionLabelSource | None
    section_label_strength: LabelStrength | None


def assign(segment: Assignable) -> None:
    """Set a segment's label, source and strength from its evidence.

    With no evidence the label and strength are cleared and the source is
    kept: it may mark a section the outline saw but could not name.
    """
    best = decide(segment.evidence)
    if best is None:
        segment.section_label = None
        segment.section_label_strength = None
        return
    segment.section_label = best.label
    segment.section_label_source = best.source
    segment.section_label_strength = best.strength


def merged_evidence(*groups: Iterable[LabelEvidence]) -> tuple[LabelEvidence, ...]:
    """Evidence of several parts, deduplicated, in first-seen order."""
    seen: dict[tuple[str, SectionLabelSource, LabelStrength], LabelEvidence] = {}
    for group in groups:
        for item in group:
            seen.setdefault((item.label, item.source, item.strength), item)
    return tuple(seen.values())


__all__ = [
    "Assignable",
    "assign",
    "content_evidence",
    "decide",
    "merged_evidence",
]
