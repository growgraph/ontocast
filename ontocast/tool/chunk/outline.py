"""Document outline detection for chunk preparation.

The outline is the backbone of section classification: every heading in the
document closes the preceding section, whether or not the heading maps to a
known label. The previous span builder ended each span at the next *recognised*
heading, so a single unrecognised heading let one label smear across the rest of
the document -- and because the label was stamped onto segments at split time,
no later tier could correct it.

Two heading kinds must be told apart, because they need opposite treatment:

- generic section names (``Results``, ``Experimental Section``, ``References``)
  start a new section; when unrecognised they open an explicitly *unresolved*
  span rather than inheriting the previous label;
- descriptive subsection titles and document titles (``Cooperative ensemble
  breaks population-inversion limitation``) sit *inside* a section and must
  inherit its label, or their body text is lost from the parent section.

Docling gives no usable hierarchy to make this call -- PDF conversions report a
flat heading level for every header -- so the discriminator is heading
genericity: the number of content words after stopword removal.

Subject-domain-agnostic but, like :mod:`ontocast.tool.chunk.bibliography`, not
language-agnostic: the stopword list and the sentence-punctuation cues are
English/Latin-script. The failure mode is a miss (a heading treated as
descriptive), not a misfire.
"""

from __future__ import annotations

import logging
import re

from ontocast.config.section_labels import (
    SectionLabelSchema,
    normalise_heading_line,
    resolve_heading_label,
)
from ontocast.onto.enum import SectionLabelSource
from ontocast.onto.section_models import DocumentOutline, HeadingNode, SectionSpan

logger = logging.getLogger(__name__)

# A generic section name is a short noun phrase. Measured over the section
# headers of real docling conversions, every true section heading has at most
# three content words, while document titles (7-12) and descriptive subsection
# titles (5) sit above.
SECTIONLIKE_MAX_CONTENT_WORDS = 3

# Closed-class words carry no topical content, so they do not count towards the
# genericity budget ("Materials and Methods" is two content words, not three).
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "via",
        "with",
    }
)

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")

_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)

# Sentence-like punctuation inside a line marks prose, not a section name.
_SENTENCE_PUNCTUATION = re.compile(r"[.;:!?]\s+\S|[,;]")

# A standalone line that reads as a heading in documents with no markdown
# structure at all: no terminal sentence punctuation, and either fully
# upper-case or explicitly numbered.
_TEXT_HEADING_NUMBERED = re.compile(r"^(?:\d+|[IVXLivxl]+)(?:\.\d+)*[.)]?\s+\S")

_TEXT_HEADING_MAX_CHARS = 120


def content_words(normalised: str) -> list[str]:
    """Topical words of a normalised heading, with closed-class words removed."""
    return [
        word
        for word in (match.group(0).lower() for match in _WORD.finditer(normalised))
        if word not in _STOPWORDS
    ]


def heading_is_sectionlike(normalised: str) -> bool:
    """Whether a heading reads as a generic section name.

    Section-like headings start a new section and, when unrecognised, leave the
    span explicitly unresolved. Non-section-like headings (descriptive
    subsection titles, document titles) inherit the enclosing section's label.

    Args:
        normalised: Heading text after :func:`normalise_heading_line`.

    Returns:
        True when the heading is a short, generic section name.
    """
    if not normalised:
        return False
    if _SENTENCE_PUNCTUATION.search(normalised):
        return False
    words = content_words(normalised)
    if not words:
        return False
    return len(words) <= SECTIONLIKE_MAX_CONTENT_WORDS


def _make_node(text: str, start: int, body_start: int, level: int) -> HeadingNode:
    normalised = normalise_heading_line(text)
    return HeadingNode(
        text=text.strip(),
        normalised=normalised,
        start=start,
        body_start=body_start,
        level=level,
        sectionlike=heading_is_sectionlike(normalised),
    )


def markdown_headings(text: str) -> list[HeadingNode]:
    """Detect ``#``-prefixed headings in the markdown export.

    This is the complete structural signal available: docling renders every
    ``SECTION_HEADER`` item as a markdown heading line, so scanning the export
    finds exactly the items a docling walk would, with exact character offsets.
    """
    nodes: list[HeadingNode] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        match = _MARKDOWN_HEADING.match(line.strip())
        if match is not None:
            nodes.append(
                _make_node(
                    match.group(2),
                    start=offset,
                    body_start=offset + len(line),
                    level=len(match.group(1)),
                )
            )
        offset += len(line)
    return nodes


def text_headings(text: str) -> list[HeadingNode]:
    """Detect headings in documents that carry no markdown heading structure.

    Only blank-line-delimited short lines that are upper-case or explicitly
    numbered qualify, which keeps the detector from firing inside prose. Used
    as a fallback when the structural scan finds nothing.
    """
    nodes: list[HeadingNode] = []
    offset = 0
    lines = text.splitlines(keepends=True)
    stripped_lines = [line.strip() for line in lines]
    for index, line in enumerate(lines):
        stripped = stripped_lines[index]
        start = offset
        offset += len(line)
        if not stripped or len(stripped) > _TEXT_HEADING_MAX_CHARS:
            continue
        previous_blank = index == 0 or not stripped_lines[index - 1]
        next_blank = index + 1 >= len(lines) or not stripped_lines[index + 1]
        if not (previous_blank and next_blank):
            continue
        if stripped[-1] in ".,;:":
            continue
        letters = [char for char in stripped if char.isalpha()]
        is_upper = bool(letters) and all(char.isupper() for char in letters)
        if not (is_upper or _TEXT_HEADING_NUMBERED.match(stripped)):
            continue
        node = _make_node(stripped, start=start, body_start=start + len(line), level=1)
        if node.sectionlike:
            nodes.append(node)
    return nodes


def detect_headings(
    text: str, *, include_text_headings: bool = False
) -> list[HeadingNode]:
    """Detect document headings, falling back to the text heuristic when empty.

    Args:
        text: Document text (the markdown export used for section tagging).
        include_text_headings: Enable the plain-text heading heuristic for
            documents with no markdown heading structure at all.

    Returns:
        Headings ordered by character offset.
    """
    nodes = markdown_headings(text)
    if nodes or not include_text_headings:
        return nodes
    nodes = text_headings(text)
    if nodes:
        logger.debug("No markdown headings; text heuristic found %s", len(nodes))
    return nodes


def label_outline(outline: DocumentOutline, schema: SectionLabelSchema) -> None:
    """Assign labels to outline headings from the schema (mutates in place).

    Runs the anchored-pattern tier, then the keyword recall tier. Both are
    gated on ``sectionlike``: applying keyword matching to a descriptive
    subsection title mislabels it from an incidental word -- a title containing
    "limitation" is not a limitations section.
    """
    for node in outline.nodes:
        if node.label is not None or not node.sectionlike:
            continue
        resolved = resolve_heading_label(node.text, schema)
        if resolved is None:
            continue
        label, confidence, source = resolved
        node.label = label
        node.confidence = confidence
        node.source = (
            SectionLabelSource.HEADING_PATTERN
            if source == "heading_pattern"
            else SectionLabelSource.HEADING_KEYWORD
        )


def inherit_subsection_labels(outline: DocumentOutline) -> None:
    """Propagate the enclosing section's label onto descriptive subheadings.

    A non-section-like heading (a descriptive subsection title) sits inside the
    section opened by the last section-like heading, so it takes that label.
    Without this, such a heading would open an unresolved span and its body
    text would be lost from the parent section.
    """
    current_label: str | None = None
    for node in outline.nodes:
        if node.sectionlike:
            current_label = node.label
            continue
        if node.label is None and current_label is not None:
            node.label = current_label
            node.source = SectionLabelSource.HEADING_INHERITED
            # Inherited labels are weaker than the heading they came from.
            node.confidence = 0.6


def outline_to_spans(outline: DocumentOutline) -> list[SectionSpan]:
    """Convert an outline into a partition of the document into section spans.

    Every heading closes the preceding span, recognised or not. The returned
    spans tile ``[0, text_len)`` with no gaps and no overlaps; a span whose
    section type is unknown carries ``label=None`` rather than being merged into
    its neighbour.
    """
    spans: list[SectionSpan] = []
    nodes = sorted(outline.nodes, key=lambda node: node.start)
    if not nodes:
        if outline.text_len > 0:
            spans.append(SectionSpan(label=None, start=0, end=outline.text_len))
        return spans

    if nodes[0].start > 0:
        spans.append(SectionSpan(label=None, start=0, end=nodes[0].start))

    for index, node in enumerate(nodes):
        end = nodes[index + 1].start if index + 1 < len(nodes) else outline.text_len
        if end <= node.start:
            continue
        spans.append(
            SectionSpan(
                label=node.label,
                start=node.start,
                end=end,
                source=node.source,
                confidence=node.confidence,
            )
        )
    return spans


def build_document_outline(
    text: str,
    schema: SectionLabelSchema,
    *,
    include_text_headings: bool = False,
) -> DocumentOutline:
    """Detect headings and resolve the labels obtainable from heading text alone."""
    outline = DocumentOutline(
        text_len=len(text),
        nodes=detect_headings(text, include_text_headings=include_text_headings),
    )
    label_outline(outline, schema)
    inherit_subsection_labels(outline)
    return outline


def format_outline(outline: DocumentOutline) -> list[str]:
    """Render the outline as human-readable lines for logging and the CLI."""
    return [
        f"{node.start:>8}  {'#' * node.level:<6} "
        f"{('section' if node.sectionlike else 'sub/desc'):<8} "
        f"{str(node.label):<16} {node.source.value:<18} "
        f"{node.confidence:.2f}  {node.text[:70]}"
        for node in outline.nodes
    ]


__all__ = [
    "SECTIONLIKE_MAX_CONTENT_WORDS",
    "build_document_outline",
    "content_words",
    "detect_headings",
    "format_outline",
    "heading_is_sectionlike",
    "inherit_subsection_labels",
    "label_outline",
    "markdown_headings",
    "outline_to_spans",
    "text_headings",
]
