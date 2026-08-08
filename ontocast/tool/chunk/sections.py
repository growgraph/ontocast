"""Document section span detection and overlap-based labeling for chunk prepare."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ontocast.config.section_labels import (
    SectionLabelSchema,
    canonical_labels,
    get_default_section_schema,
    match_heading_line,
)
from ontocast.onto.enum import SectionLabelSource
from ontocast.onto.section_models import DocumentOutline, HeadingNode, SectionSpan
from ontocast.tool.chunk.outline import build_document_outline, outline_to_spans

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

ABSTRACT_FRONT_MATTER_MAX_CHARS = 6000


def document_text_for_section_tagging(doc: DoclingDocument) -> str:
    """Export document text used for section heading detection."""
    return doc.export_to_markdown()


def detect_section_spans(
    text: str,
    schema: SectionLabelSchema | None = None,
    *,
    include_text_headings: bool = False,
) -> list[SectionSpan]:
    """Detect document sections and return a partition of the text into spans.

    Every heading closes the preceding span, so an unrecognised heading yields an
    explicitly unresolved (``label=None``) span instead of letting the previous
    label run on to the next recognised heading.

    Args:
        text: Document text (the markdown export).
        schema: Section label schema; the manifest default when omitted.
        include_text_headings: Enable the plain-text heading heuristic for
            documents with no markdown heading structure.

    Returns:
        Section spans tiling ``text``, ordered by start offset.
    """
    if not text:
        return []
    active = schema or get_default_section_schema()
    outline = build_document_outline(
        text, active, include_text_headings=include_text_headings
    )
    spans = outline_to_spans(outline)
    return inject_front_matter_spans(spans, text, active)


def inject_front_matter_spans(
    spans: list[SectionSpan],
    text: str,
    schema: SectionLabelSchema,
    *,
    min_gap_chars: int = 80,
    max_gap_chars: int = ABSTRACT_FRONT_MATTER_MAX_CHARS,
) -> list[SectionSpan]:
    """Label unheaded front matter before the first labeled section as abstract.

    Leading unresolved spans (a title block, an unrecognised banner heading) are
    skipped when locating the first labeled section, so front matter is still
    recovered on documents whose first recognised section is not an IMRaD
    opener -- papers that jump straight to ``Results`` are common.
    """
    if "abstract" not in canonical_labels(schema):
        return spans
    if any(span.label == "abstract" for span in spans):
        return spans
    if not spans:
        return spans

    ordered = sorted(spans, key=lambda span: span.start)
    first_labeled = next(
        (span for span in ordered if span.label is not None),
        None,
    )
    if first_labeled is None or first_labeled.start <= 0:
        return spans

    gap = text[: first_labeled.start].strip()
    gap_len = len(gap)
    if gap_len < min_gap_chars or gap_len > max_gap_chars:
        return spans

    # The front matter may already be covered by unresolved spans; replace them
    # rather than overlapping, so the result stays a partition.
    tail = [span for span in ordered if span.start >= first_labeled.start]
    abstract_span = SectionSpan(
        label="abstract",
        start=0,
        end=first_labeled.start,
        source=SectionLabelSource.FRONT_MATTER,
        confidence=0.5,
    )
    return [abstract_span, *tail]


def build_section_spans_from_labels(
    text: str, labeled_headings: list[tuple[int, str]]
) -> list[SectionSpan]:
    """Build section spans from explicit ``(offset, label)`` pairs."""
    outline = DocumentOutline(
        text_len=len(text),
        nodes=[
            HeadingNode(
                text=label,
                normalised=label,
                start=start,
                body_start=start,
                label=label,
                source=SectionLabelSource.HEADING_PATTERN,
                confidence=0.95,
            )
            for start, label in sorted(labeled_headings, key=lambda item: item[0])
        ],
    )
    return [span for span in outline_to_spans(outline) if span.label is not None]


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
    "DocumentOutline",
    "HeadingNode",
    "SectionSpan",
    "build_section_spans_from_labels",
    "detect_section_spans",
    "document_text_for_section_tagging",
    "inject_front_matter_spans",
    "label_from_headings",
    "label_text_from_spans",
    "resolve_section_label",
]
