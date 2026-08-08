"""Helpers for constructing and normalizing DoclingDocument instances.

``docling-core`` is resolved on demand: it ships in the ``documents`` extra, and
these helpers sit on the import path of modules the light core does load.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ontocast.util.optional import require

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

_LIGATURE_GAP_RE = re.compile(r"(?<=[A-Za-z]) (ffi|ffl|fi|fl|ff) (?=[A-Za-z])")


def plain_text_to_docling_doc(text: str, doc_name: str) -> DoclingDocument:
    """Wrap plain text as a single-paragraph DoclingDocument."""
    doc_module = require("docling_core.types.doc", feature="Docling documents")
    doc = doc_module.DoclingDocument(name=doc_name)
    doc.add_text(label=doc_module.DocItemLabel.PARAGRAPH, text=text)
    return doc


def json_payload_text(payload: object) -> str | None:
    """Document text inside a JSON payload, by a small top-level heuristic.

    ``text`` when present, else the longest top-level string. Lives here rather
    than in the conversion agent because JSON inputs are routed around the
    Docling converter entirely -- anything that reads a document from a path has
    to make the same choice, and two copies of the heuristic would let the CLI
    and the pipeline disagree about what a file's text even is.

    Args:
        payload: Parsed JSON, expected to be an object.

    Returns:
        The document text, or ``None`` when the payload is not an object or
        holds no string.
    """
    if not isinstance(payload, dict):
        return None
    text_value = payload.get("text")
    if isinstance(text_value, str):
        return text_value

    largest_text: str | None = None
    for value in payload.values():
        if isinstance(value, str):
            if largest_text is None or len(value) > len(largest_text):
                largest_text = value
    return largest_text


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
