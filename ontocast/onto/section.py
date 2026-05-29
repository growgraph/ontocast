"""Document section span models and helpers for structured-document preprocessing."""

from __future__ import annotations

import re

from pydantic import Field

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.model import BasePydanticModel

# Normalised section labels and heading line patterns (after stripping markdown #).
_SECTION_HEADING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("introduction", re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s*)?introduction\s*$", re.I)),
    ("related_work", re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s*)?related\s+work\s*$", re.I)),
    ("background", re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s*)?background\s*$", re.I)),
    ("methods", re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s*)?methods?\s*$", re.I)),
    ("methods", re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s*)?methodology\s*$", re.I)),
    ("results", re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s*)?results?\s*$", re.I)),
    ("discussion", re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s*)?discussion\s*$", re.I)),
    ("conclusion", re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s*)?conclusions?\s*$", re.I)),
    ("future_work", re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s*)?future\s+work\s*$", re.I)),
    ("limitations", re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s*)?limitations?\s*$", re.I)),
)

_MAX_HEADING_LINE_LEN = 120


class SectionSpan(BasePydanticModel):
    """Character span of a document section with a normalised label."""

    label: str = Field(
        description="Normalised section label, e.g. results, future_work"
    )
    start: int = Field(description="Start character offset in input_text (inclusive)")
    end: int = Field(description="End character offset in input_text (exclusive)")


def _normalise_heading_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        stripped = stripped.lstrip("#").strip()
    return stripped


def _match_section_label(heading_line: str) -> str | None:
    normalised = _normalise_heading_line(heading_line)
    if not normalised or len(normalised) > _MAX_HEADING_LINE_LEN:
        return None
    for label, pattern in _SECTION_HEADING_PATTERNS:
        if pattern.match(normalised):
            return label
    return None


def detect_section_spans(text: str) -> list[SectionSpan]:
    """Detect academic-style section headings and return character spans."""
    if not text:
        return []

    heading_starts: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        label = _match_section_label(line)
        if label is not None:
            heading_starts.append((offset, label))
        offset += len(line)

    if not heading_starts:
        return []

    spans: list[SectionSpan] = []
    for index, (start, label) in enumerate(heading_starts):
        end = (
            heading_starts[index + 1][0]
            if index + 1 < len(heading_starts)
            else len(text)
        )
        if end > start:
            spans.append(SectionSpan(label=label, start=start, end=end))
    return spans


def _chunk_char_range(
    chunk_text: str, document_text: str, search_from: int
) -> tuple[int, int]:
    """Locate chunk in document_text; return (start, end) or (0, 0) if not found."""
    if not chunk_text:
        return 0, 0
    position = document_text.find(chunk_text, search_from)
    if position < 0:
        position = document_text.find(chunk_text)
    if position < 0:
        return 0, 0
    return position, position + len(chunk_text)


def resolve_section_label(
    chunk_text: str,
    document_text: str,
    spans: list[SectionSpan],
    search_from: int = 0,
) -> tuple[str | None, int]:
    """Return section label with max overlap; second value is next search offset."""
    start, end = _chunk_char_range(chunk_text, document_text, search_from)
    if end <= start or not spans:
        return None, end

    best_label: str | None = None
    best_overlap = 0
    for span in spans:
        overlap_start = max(start, span.start)
        overlap_end = min(end, span.end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = span.label
    return best_label, end


def assign_section_labels(
    units: list[ContentUnit],
    document_text: str,
    spans: list[SectionSpan],
) -> None:
    """Set section_label on each content unit from section spans."""
    search_from = 0
    for unit in units:
        label, search_from = resolve_section_label(
            unit.text, document_text, spans, search_from
        )
        unit.section_label = label


def filter_units_by_target_sections(
    units: list[ContentUnit],
    target_sections: list[str] | None,
) -> list[ContentUnit]:
    """Drop units whose section_label is not in target_sections."""
    if target_sections is None:
        return units
    allowed = {
        section.strip().lower() for section in target_sections if section.strip()
    }
    if not allowed:
        return units
    return [
        unit
        for unit in units
        if unit.section_label is not None and unit.section_label.lower() in allowed
    ]


def should_summarize_unit(
    unit: ContentUnit,
    summarize_sections: list[str] | None,
) -> bool:
    """Whether a unit should be passed through the summarization node."""
    if summarize_sections is None:
        return False
    if not summarize_sections or "*" in summarize_sections:
        return True
    if unit.section_label is None:
        return False
    allowed = {section.strip().lower() for section in summarize_sections}
    return unit.section_label.lower() in allowed
