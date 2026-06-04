"""Document section span detection and overlap-based labeling for chunk prepare."""

from __future__ import annotations

from docling_core.types.doc import DoclingDocument

from ontocast.config.section_labels import (
    SectionLabelSchema,
    canonical_labels,
    get_default_section_schema,
    match_heading_line,
)
from ontocast.onto.section_models import SectionSpan

ABSTRACT_FRONT_MATTER_MAX_CHARS = 6000
_IMRAD_START_LABELS = frozenset({"introduction", "related_work", "background"})


def document_text_for_section_tagging(doc: DoclingDocument) -> str:
    """Export document text used for section heading detection."""
    return doc.export_to_markdown()


def _build_spans_from_heading_starts(
    text: str, heading_starts: list[tuple[int, str]]
) -> list[SectionSpan]:
    if not heading_starts:
        return []
    sorted_starts = sorted(heading_starts, key=lambda item: item[0])
    spans: list[SectionSpan] = []
    for index, (start, label) in enumerate(sorted_starts):
        end = (
            sorted_starts[index + 1][0] if index + 1 < len(sorted_starts) else len(text)
        )
        if end > start:
            spans.append(SectionSpan(label=label, start=start, end=end))
    return spans


def detect_section_spans(
    text: str,
    schema: SectionLabelSchema | None = None,
) -> list[SectionSpan]:
    """Detect section headings via regex and return character spans."""
    if not text:
        return []
    active = schema or get_default_section_schema()
    heading_starts: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        label = match_heading_line(line, active)
        if label is not None:
            heading_starts.append((offset, label))
        offset += len(line)
    spans = _build_spans_from_heading_starts(text, heading_starts)
    return inject_front_matter_spans(spans, text, active)


def inject_front_matter_spans(
    spans: list[SectionSpan],
    text: str,
    schema: SectionLabelSchema,
    *,
    min_gap_chars: int = 80,
    max_gap_chars: int = ABSTRACT_FRONT_MATTER_MAX_CHARS,
) -> list[SectionSpan]:
    """Insert an abstract span for unheaded front matter before the first IMRaD section."""
    if "abstract" not in canonical_labels(schema):
        return spans
    if any(span.label == "abstract" for span in spans):
        return spans
    if not spans:
        return spans

    first = min(spans, key=lambda span: span.start)
    if first.label not in _IMRAD_START_LABELS:
        return spans

    gap = text[: first.start].strip()
    gap_len = len(gap)
    if gap_len < min_gap_chars or gap_len > max_gap_chars:
        return spans

    abstract_span = SectionSpan(label="abstract", start=0, end=first.start)
    return [abstract_span, *spans]


def build_section_spans_from_labels(
    text: str, labeled_headings: list[tuple[int, str]]
) -> list[SectionSpan]:
    """Build section spans from explicit (offset, label) pairs."""
    return _build_spans_from_heading_starts(text, labeled_headings)


def _chunk_char_range(
    chunk_text: str, document_text: str, search_from: int
) -> tuple[int, int]:
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
    """Return section label with max overlap; second value is next search offset.

    When the chunk text cannot be located in ``document_text`` the cursor is
    preserved at ``search_from`` (not reset to 0) so that subsequent segments
    are not mis-anchored to the start of the document.
    """
    start, end = _chunk_char_range(chunk_text, document_text, search_from)
    if end <= start or not spans:
        # Preserve cursor instead of returning 0 so subsequent finds stay ordered.
        return None, search_from

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


def label_text_from_spans(
    text: str,
    document_text: str,
    spans: list[SectionSpan],
    search_from: int,
) -> tuple[str | None, int]:
    """Assign section label via span overlap; return label and next search offset."""
    return resolve_section_label(text, document_text, spans, search_from)


def label_from_headings(
    headings: list[str] | None,
    schema: SectionLabelSchema,
) -> str | None:
    """Return the first schema label matched from the heading breadcrumb (most-specific first).

    This uses docling's structural metadata directly — no substring search
    required — so it is reliable even when the markdown export differs from
    the hybrid-chunker text.
    """
    if not headings:
        return None
    for heading in reversed(headings):
        label = match_heading_line(heading, schema)
        if label is not None:
            return label
    return None


__all__ = [
    "ABSTRACT_FRONT_MATTER_MAX_CHARS",
    "SectionSpan",
    "build_section_spans_from_labels",
    "detect_section_spans",
    "document_text_for_section_tagging",
    "inject_front_matter_spans",
    "label_from_headings",
    "label_text_from_spans",
    "resolve_section_label",
]
