"""Pydantic models for document outlines and section spans."""

from pydantic import Field

from ontocast.config.section_labels import (
    canonical_labels,
    get_default_section_schema,
)
from ontocast.onto.enum import SectionLabelSource
from ontocast.onto.model import BasePydanticModel

# Backward-compatible alias for prompts and imports.
CANONICAL_SECTION_LABELS: tuple[str, ...] = canonical_labels(
    get_default_section_schema()
)


class HeadingNode(BasePydanticModel):
    """One detected heading in the document outline.

    Attributes:
        text: Raw heading line as it appears in the document text.
        normalised: Heading text after decoration/numbering stripping.
        start: Character offset of the heading line itself.
        body_start: Character offset just past the heading line.
        level: Markdown heading depth (1 = top). Docling reports a flat level
            for PDF conversions, so this is informational only.
        sectionlike: Whether the heading reads as a generic section name rather
            than a descriptive subsection title or a document title.
        label: Canonical section label, when resolved.
        source: How ``label`` was decided.
        confidence: Confidence in ``label`` in ``[0, 1]``.
    """

    text: str
    normalised: str
    start: int
    body_start: int
    level: int = 1
    sectionlike: bool = True
    label: str | None = None
    source: SectionLabelSource = SectionLabelSource.OUTLINE_UNRESOLVED
    confidence: float = 0.0


class DocumentOutline(BasePydanticModel):
    """Ordered headings detected in a document, with the document length."""

    text_len: int
    nodes: list[HeadingNode] = Field(default_factory=list)


class SectionSpan(BasePydanticModel):
    """Character span of a document section with a normalised label.

    ``label`` is ``None`` for a region whose section type is not (yet) known —
    for example an unrecognised but section-like heading. Such a span is
    explicitly unresolved rather than absent, which is what stops a neighbouring
    label from being smeared across it.
    """

    label: str | None = None
    start: int
    end: int
    source: SectionLabelSource = SectionLabelSource.OUTLINE_UNRESOLVED
    confidence: float = 0.0


__all__ = [
    "CANONICAL_SECTION_LABELS",
    "DocumentOutline",
    "HeadingNode",
    "SectionSpan",
]
