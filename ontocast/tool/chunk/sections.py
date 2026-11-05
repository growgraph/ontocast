"""Document section spans for chunk prepare: outline spans plus front matter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ontocast.config.section_labels import (
    SectionLabelSchema,
    canonical_labels,
    get_default_section_schema,
)
from ontocast.onto.enum import LabelStrength, SectionLabelSource
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
        strength=LabelStrength.WEAK,
    )
    return [abstract_span, *tail]


__all__ = [
    "ABSTRACT_FRONT_MATTER_MAX_CHARS",
    "DocumentOutline",
    "HeadingNode",
    "SectionSpan",
    "detect_section_spans",
    "document_text_for_section_tagging",
    "inject_front_matter_spans",
]
