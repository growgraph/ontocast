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

# Single-sided ligature gaps, accepted only where the joined form cannot be
# two real words. No English word starts with "ff", so "a ffected" and
# "di fferent" have one reading; and a word ending in "fi"/"fl" with a letter
# before it ("signifi cant", "confi ned", "refl ected") has no two-word
# reading either -- "fl oz" is excluded by that letter-before requirement. The
# symmetric gap *before* "fi"/"fl" ("the field", "a flat", "of it") is a real
# two-word phrase far too often to touch.
_LIGATURE_GAP_BEFORE_FF_RE = re.compile(r"(?<=[A-Za-z]) (?=ff)")
_LIGATURE_GAP_AFTER_FI_FL_RE = re.compile(r"(?<=[A-Za-z]f[il]) (?=[A-Za-z])")

# The entities conversion leaves escaped inside prose (a "<" in "T &lt; 300 K"
# is a comparison, not markup). Deliberately not html.unescape, which also
# expands semicolon-less entities and would turn a "&para" in running text
# into a pilcrow.
_HTML_ENTITY_RE = re.compile(r"&(lt|gt|amp|quot|apos);")
_HTML_ENTITIES = {"lt": "<", "gt": ">", "amp": "&", "quot": '"', "apos": "'"}

# A carriage return, inline whitespace, newline is a column-wrap artifact.
# Only inline whitespace is consumed: "\r \n\n" is a paragraph break and
# must stay one.
_CARRIAGE_RETURN_WRAP_RE = re.compile(r"\r[ \t\f\v]*\n")

# "2 × 10 6" is "2 × 10^6" with the superscript flattened: one or two exponent
# digits, a sign allowed, and never the start of a hyphenated word
# ("10 6-membered"). The bare "10 6" is rejoined only behind an approximation
# cue ("~", "≈", "order of"); on its own it is two numbers more often than one.
_EXPONENT_TAIL = r"10(?:\s+|\s*(?P<sign>[-−–])\s*)(?P<exp>\d{1,2})\b(?![-−–]\w)"
_EXPONENT_PRODUCT_RE = re.compile(r"(?P<mantissa>\d)\s*[×x]\s*" + _EXPONENT_TAIL)
_EXPONENT_BARE_RE = re.compile(r"(?P<cue>(?:order\s+of|[~≈∼≃])\s*)" + _EXPONENT_TAIL)


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
    """Repair common ASCII ligature gaps in extracted publisher-PDF text."""
    return _LIGATURE_GAP_RE.sub(r"\1", text)


def unescape_html_entities(text: str) -> str:
    """Expand ``&lt;``, ``&gt;``, ``&amp;``, ``&quot;`` and ``&apos;`` only."""
    return _HTML_ENTITY_RE.sub(lambda match: _HTML_ENTITIES[match.group(1)], text)


def normalize_carriage_return_wraps(text: str) -> str:
    """Collapse ``\\r<inline whitespace>\\n`` column wraps to a newline."""
    return _CARRIAGE_RETURN_WRAP_RE.sub("\n", text)


def _rejoin_exponent(prefix: str, match: re.Match[str]) -> str:
    sign = "-" if match.group("sign") else ""
    return f"{prefix}10^{sign}{match.group('exp')}"


def rejoin_flattened_exponents(text: str) -> str:
    """Rejoin ``2 × 10 6`` to ``2 × 10^6`` and ``~10 6`` to ``~10^6``.

    The product form needs a mantissa digit before ``×``/``x``; the bare form
    needs an approximation cue before ``10``. Neither fires when the would-be
    exponent starts a hyphenated word.
    """
    text = _EXPONENT_PRODUCT_RE.sub(
        lambda match: _rejoin_exponent(f"{match.group('mantissa')} × ", match), text
    )
    return _EXPONENT_BARE_RE.sub(
        lambda match: _rejoin_exponent(match.group("cue"), match), text
    )


def repair_single_sided_ligature_gaps(text: str) -> str:
    """Close ligature gaps that leave the ligature glued to one side.

    Only the two shapes with a single reading are closed: a gap before ``ff``
    (``a ffected``) and a gap after a letter-preceded ``fi``/``fl``
    (``signifi cant``). See the pattern comments for what is left alone.
    """
    text = _LIGATURE_GAP_BEFORE_FF_RE.sub("", text)
    return _LIGATURE_GAP_AFTER_FI_FL_RE.sub("", text)


def repair_numeric_artifacts(text: str) -> str:
    """Repair the conversion artifacts that are pattern-local and safe.

    Composes :func:`unescape_html_entities`,
    :func:`normalize_carriage_return_wraps`,
    :func:`rejoin_flattened_exponents` and
    :func:`repair_single_sided_ligature_gaps`, in that order. Each rule
    rewrites only a span whose repaired reading is the sole plausible one;
    superscript/subscript duplication and citation markers fused into values
    are not recoverable from the text and are deliberately not touched.

    Args:
        text: One text item as emitted by conversion.

    Returns:
        The repaired text; unchanged when no rule applies.
    """
    text = unescape_html_entities(text)
    text = normalize_carriage_return_wraps(text)
    text = rejoin_flattened_exponents(text)
    return repair_single_sided_ligature_gaps(text)


def apply_text_sanitizers(
    doc: DoclingDocument,
    *,
    repair_ligature_gaps_enabled: bool = False,
    repair_numeric_artifacts_enabled: bool = False,
) -> DoclingDocument:
    """Apply the enabled post-conversion text sanitizers to every text item.

    Runs once at conversion time, so chunk boundaries and the on-disk chunk
    cache are computed on repaired text; both flags sit in the converter cache
    key. The two-sided ligature rule runs first so a fully isolated ligature
    (``e ff ect``) is closed before the single-sided rules see its remainder.
    """
    if not (repair_ligature_gaps_enabled or repair_numeric_artifacts_enabled):
        return doc

    for item in doc.texts:
        text = item.text
        if repair_ligature_gaps_enabled:
            # Works around publisher-PDF ligature splits that Docling passes
            # through. Removable once upstream Docling normalises ASCII
            # fi/fl/ff gap patterns -- a breaking change, since the flag is in
            # the converter cache key.
            text = repair_ligature_gaps(text)
        if repair_numeric_artifacts_enabled:
            text = repair_numeric_artifacts(text)
        item.text = text

    return doc
