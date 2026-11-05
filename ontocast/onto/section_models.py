"""Pydantic models for document outlines and section spans."""

from pydantic import Field

from ontocast.config.section_labels import (
    canonical_labels,
    get_default_section_schema,
)
from ontocast.onto.enum import LabelStrength, SectionLabelSource
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
            for PDF conversions, so levels place a heading only in documents
            whose levels vary.
        number: The heading's section number ("2.1", "II", "A.2"), if any.
        parent: Index in the outline of the heading this one sits under.
        tree_placed: Whether numbering or markdown levels place this heading;
            ``parent is None`` then means top level. Unplaced headings are
            judged by genericity.
        sectionlike: Whether the heading reads as a generic section name rather
            than a descriptive subsection title or a document title.
        label: Canonical section label, when resolved.
        source: How ``label`` was decided.
        strength: Whether ``label`` may take part in a drop decision.
    """

    text: str
    normalised: str
    start: int
    body_start: int
    level: int = 1
    number: str | None = None
    parent: int | None = None
    tree_placed: bool = False
    sectionlike: bool = True
    label: str | None = None
    source: SectionLabelSource = SectionLabelSource.OUTLINE_UNRESOLVED
    strength: LabelStrength = LabelStrength.WEAK


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
    strength: LabelStrength = LabelStrength.WEAK
    heading: str | None = Field(
        default=None,
        description="The heading line that opens the span; none for unheaded text.",
    )


class LabelEvidence(BasePydanticModel):
    """One tier's vote on a unit's section label.

    Every tier that speaks adds one; the unit's label is the strongest vote,
    and between equals the heading's (see ``ontocast.tool.chunk.assignment``).
    """

    label: str
    source: SectionLabelSource
    strength: LabelStrength


class OutlineLabelReport(BasePydanticModel):
    """What the model tier (``CHUNK_SECTION_CLASSIFIER=llm``) did for one document.

    The tier is best effort: a window whose call fails gives no votes and the
    document is still prepared. ``windows_failed`` above zero means part of
    the outline was labelled without the model.
    """

    windows: int = Field(default=0, description="Outline windows sent to the model.")
    windows_failed: int = Field(
        default=0, description="Windows whose call raised and gave no votes."
    )
    errors: dict[str, int] = Field(
        default_factory=dict,
        description="Failed windows by exception class name.",
    )


__all__ = [
    "CANONICAL_SECTION_LABELS",
    "DocumentOutline",
    "HeadingNode",
    "LabelEvidence",
    "OutlineLabelReport",
    "SectionSpan",
]
