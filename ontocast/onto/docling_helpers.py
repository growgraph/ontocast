"""Helpers for constructing and normalizing DoclingDocument instances."""

import re

from docling_core.types.doc import DocItemLabel, DoclingDocument

_LIGATURE_GAP_RE = re.compile(r"(?<=[A-Za-z]) (ffi|ffl|fi|fl|ff) (?=[A-Za-z])")


def plain_text_to_docling_doc(text: str, doc_name: str) -> DoclingDocument:
    """Wrap plain text as a single-paragraph DoclingDocument."""
    doc = DoclingDocument(name=doc_name)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text=text)
    return doc


def repair_ligature_gaps(text: str) -> str:
    """TEMP: repair common ASCII ligature gaps in extracted publisher-PDF text."""
    return _LIGATURE_GAP_RE.sub(r"\1", text)


def apply_text_sanitizers(
    doc: DoclingDocument,
    *,
    repair_ligature_gaps_enabled: bool = False,
) -> DoclingDocument:
    """Apply optional post-conversion text sanitizers to a DoclingDocument."""
    if not repair_ligature_gaps_enabled:
        return doc

    # TEMP: Work around publisher-PDF ligature splits that Docling still passes through.
    # Remove once upstream Docling reliably normalizes ASCII fi/fl/ff gap patterns.
    for item in doc.texts:
        item.text = repair_ligature_gaps(item.text)

    return doc
