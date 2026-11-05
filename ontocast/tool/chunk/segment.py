"""Mutable text segments during chunk preparation (before sizing)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ontocast.config.section_labels import SectionLabelSchema, match_heading_line
from ontocast.onto.enum import LabelStrength, SectionLabelSource
from ontocast.onto.section_models import LabelEvidence
from ontocast.tool.chunk.sizing import DEFAULT_PART_SEPARATOR

logger = logging.getLogger(__name__)


@dataclass
class PrepareSegment:
    """One logical segment prior to min/max sizing.

    ``evidence`` holds every tier's vote; the label, its source and its
    strength are decided from it (:func:`ontocast.tool.chunk.assignment.assign`).
    With no evidence, a source of ``OUTLINE_UNRESOLVED`` marks a section the
    outline saw but could not name.
    """

    text: str
    headings: list[str] | None = None
    section_label: str | None = None
    section_label_source: SectionLabelSource | None = None
    section_label_strength: LabelStrength | None = None
    evidence: list[LabelEvidence] = field(default_factory=list)


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def is_abstract_exempt(segment: PrepareSegment, schema: SectionLabelSchema) -> bool:
    """Preserve short abstract segments instead of merging them into neighbors."""
    first_line = _first_non_empty_line(segment.text)
    if first_line and match_heading_line(first_line, schema) == "abstract":
        return True
    if segment.headings:
        for heading in segment.headings:
            if match_heading_line(heading, schema) == "abstract":
                return True
    return False


def starts_with_section_heading(
    segment: PrepareSegment, schema: SectionLabelSchema
) -> bool:
    """True when the segment opens with a recognised section heading line."""
    first_line = _first_non_empty_line(segment.text)
    return bool(first_line and match_heading_line(first_line, schema) is not None)


def _effective_length(segment: PrepareSegment, schema: SectionLabelSchema) -> int:
    if is_abstract_exempt(segment, schema):
        return max(len(segment.text.strip()), 1)
    return len(segment.text.strip())


def merge_into_right(left: PrepareSegment, right: PrepareSegment) -> PrepareSegment:
    left_text = left.text.strip()
    right_text = right.text.strip()
    combined_text = (
        f"{left_text}{DEFAULT_PART_SEPARATOR}{right_text}"
        if left_text and right_text
        else left_text or right_text
    )
    winner = right if right.section_label is not None else left
    return PrepareSegment(
        text=combined_text,
        headings=right.headings or left.headings,
        section_label=winner.section_label,
        section_label_source=winner.section_label_source,
        section_label_strength=winner.section_label_strength,
        evidence=list(winner.evidence),
    )


def segments_differ_in_structure(
    left: PrepareSegment,
    right: PrepareSegment,
    schema: SectionLabelSchema,
) -> bool:
    """True when merging would join distinct document sections."""
    if starts_with_section_heading(right, schema):
        return True
    if left.headings and right.headings and left.headings != right.headings:
        return True
    return False


def merge_into_left(left: PrepareSegment, right: PrepareSegment) -> PrepareSegment:
    left_text = left.text.strip()
    right_text = right.text.strip()
    combined_text = (
        f"{left_text}{DEFAULT_PART_SEPARATOR}{right_text}"
        if left_text and right_text
        else left_text or right_text
    )
    winner = left if left.section_label is not None else right
    return PrepareSegment(
        text=combined_text,
        headings=left.headings or right.headings,
        section_label=winner.section_label,
        section_label_source=winner.section_label_source,
        section_label_strength=winner.section_label_strength,
        evidence=list(winner.evidence),
    )


def _can_merge_small_into_left(
    segments: list[PrepareSegment],
    index: int,
    schema: SectionLabelSchema,
) -> bool:
    if index <= 0:
        return False
    small = segments[index]
    left = segments[index - 1]
    if starts_with_section_heading(small, schema):
        return False
    if is_abstract_exempt(left, schema):
        return False
    return not segments_differ_in_structure(left, small, schema)


def coalesce_small_segments_right(
    segments: list[PrepareSegment],
    min_chars: int,
    schema: SectionLabelSchema,
) -> list[PrepareSegment]:
    """Merge undersized segments into the right neighbor; otherwise into the left."""
    if min_chars <= 0 or len(segments) <= 1:
        return segments

    merge_count = 0
    index = 0
    while index < len(segments):
        if (
            is_abstract_exempt(segments[index], schema)
            or _effective_length(segments[index], schema) >= min_chars
        ):
            index += 1
            continue
        merged = False
        if index + 1 < len(segments) and not segments_differ_in_structure(
            segments[index], segments[index + 1], schema
        ):
            segments[index + 1] = merge_into_right(segments[index], segments[index + 1])
            del segments[index]
            merge_count += 1
            merged = True
        elif _can_merge_small_into_left(segments, index, schema):
            segments[index - 1] = merge_into_left(segments[index - 1], segments[index])
            del segments[index]
            merge_count += 1
            index -= 1
            merged = True
        if not merged:
            index += 1

    if merge_count:
        logger.debug(
            "Coalesced %s undersized segment(s) (min_chars=%s)",
            merge_count,
            min_chars,
        )
    return segments
