"""Content-based section classification for regions with no usable heading.

Some documents carry no section headings at all -- publisher "Letter" formats
run the whole body as continuous prose -- so heading analysis leaves the text
unresolved. This module labels such regions from the surface form of the text
itself, in the same style as :mod:`ontocast.tool.chunk.bibliography`: count
marker features, normalise per kilochar, and require a conservative threshold.

Two tiers, selected by ``CHUNK_SECTION_DENSITY``:

- ``conservative`` (default) recognises only the two section types with a
  near-unique surface form: reference lists and acknowledgements. Both are
  boilerplate that the default exclusions drop anyway, so a miss is cheap and a
  hit is worth having.
- ``aggressive`` additionally guesses at methods/results/introduction. These
  features do **not** cleanly separate those sections -- figure references and
  past-tense passives appear in results *and* methods *and* discussion -- so it
  is off by default. A wrong label is worse than no label here, because the
  section filters act on it and a mislabeled chunk is silently dropped.

Subject-domain-agnostic, but like the bibliography detector it assumes
English/Latin-script prose. The failure mode is a miss, not a misfire.
"""

from __future__ import annotations

import re

from ontocast.config.section_labels import SectionLabelSchema, canonical_labels
from ontocast.tool.chunk.bibliography import looks_like_bibliography

# Gratitude and funding attributions, which are close to unique to
# acknowledgement sections.
_GRATITUDE = re.compile(
    r"(?i)\b(?:we|the\s+authors?)\s+(?:gratefully\s+|sincerely\s+)?"
    r"(?:thank|acknowledge|are\s+grateful\s+to|wish\s+to\s+thank)\b"
)
_FUNDING = re.compile(
    r"(?i)\b(?:grant|award|contract|agreement)\s*(?:no\.?|number|#)\s*[\w\-/]+"
    r"|\bfunded\s+by\b|\bfinancial\s+support\s+(?:from|of)\b"
    r"|\bsupported\s+(?:in\s+part\s+)?by\s+the\b"
)

# Cross-references to figures, tables and equations: dense in results, but also
# present in methods and discussion, hence aggressive-only.
_FLOAT_REF = re.compile(
    r"(?i)\b(?:fig(?:ure|s?\.)?|tab(?:le|s?\.)?|scheme|eq(?:uation|n?\.)?)\s*"
    r"S?\d+[a-z]?\b"
)
# Reported quantities: decimals, percentages, tolerances.
_QUANTITY = re.compile(r"\b\d+\.\d+\b|\d\s*%|±")
# Procedural passive voice typical of a methods write-up.
_PROCEDURAL = re.compile(
    r"(?i)\b(?:was|were)\s+\w+(?:ed|n)\b"
    r"|\b(?:carried\s+out|performed\s+using|according\s+to|as\s+described\s+(?:in|by))\b"
)
# Inline citation markers, dense in introductions and related work.
_INLINE_CITATION = re.compile(
    r"\[\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\]"
    r"|\((?:[A-Z][A-Za-z'’-]+(?:\s+(?:et\s+al\.|and\s+[A-Z][A-Za-z'’-]+))?,\s*"
    r"(?:19|20)\d{2}[a-z]?)\)"
)

MIN_TEXT_CHARS = 200


def _per_kilochar(count: int, length: int) -> float:
    return count * 1000.0 / max(length, 1)


def score_section_labels(
    text: str,
    schema: SectionLabelSchema,
    *,
    aggressive: bool = False,
) -> dict[str, float]:
    """Score candidate section labels from content features.

    Args:
        text: Chunk or section text.
        schema: Active label schema; labels absent from it are never scored.
        aggressive: Include the low-precision methods/results/introduction
            features.

    Returns:
        Mapping of label to score; higher is stronger. Empty when the text is
        too short to judge.
    """
    stripped = text.strip()
    if len(stripped) < MIN_TEXT_CHARS:
        return {}

    allowed = set(canonical_labels(schema))
    length = len(stripped)
    scores: dict[str, float] = {}

    if "references" in allowed and looks_like_bibliography(stripped):
        scores["references"] = 0.9

    if "acknowledgements" in allowed:
        gratitude = len(_GRATITUDE.findall(stripped))
        funding = len(_FUNDING.findall(stripped))
        if gratitude or funding:
            # Acknowledgements are short, so a single gratitude clause in a
            # brief passage is strong evidence; the same clause buried in a
            # long section is not.
            density = _per_kilochar(gratitude * 2 + funding, length)
            scores["acknowledgements"] = min(0.95, density / 2.0)

    if not aggressive:
        return {label: score for label, score in scores.items() if score > 0.0}

    floats = _per_kilochar(len(_FLOAT_REF.findall(stripped)), length)
    quantities = _per_kilochar(len(_QUANTITY.findall(stripped)), length)
    procedural = _per_kilochar(len(_PROCEDURAL.findall(stripped)), length)
    citations = _per_kilochar(len(_INLINE_CITATION.findall(stripped)), length)

    if "results" in allowed:
        scores["results"] = min(0.8, (floats * 0.6 + quantities * 0.15))
    if "methods" in allowed:
        scores["methods"] = min(0.8, procedural * 0.25)
    if "introduction" in allowed:
        scores["introduction"] = min(0.8, max(0.0, citations * 0.3 - floats * 0.3))

    return {label: score for label, score in scores.items() if score > 0.0}


def classify_by_density(
    text: str,
    schema: SectionLabelSchema,
    *,
    aggressive: bool = False,
    min_score: float = 0.6,
    margin: float = 1.5,
) -> tuple[str, float] | None:
    """Label a text region from content features, or ``None`` if unclear.

    A label is returned only when the best score clears ``min_score`` *and*
    beats the runner-up by a factor of ``margin``. Ambiguity resolves to
    ``None`` deliberately: an unlabeled chunk is merely unselectable, whereas a
    wrongly labeled one is silently dropped or wrongly extracted.

    Args:
        text: Chunk or section text.
        schema: Active label schema.
        aggressive: Include the low-precision feature set.
        min_score: Absolute floor the winning score must clear.
        margin: Factor by which the winner must beat the runner-up.

    Returns:
        ``(label, score)`` or ``None``.
    """
    scores = score_section_labels(text, schema, aggressive=aggressive)
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    label, score = ranked[0]
    if score < min_score:
        return None
    if len(ranked) > 1 and score < ranked[1][1] * margin:
        return None
    return label, score


__all__ = [
    "MIN_TEXT_CHARS",
    "classify_by_density",
    "score_section_labels",
]
