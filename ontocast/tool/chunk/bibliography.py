"""Bibliography detection and routing for content units.

Reference lists mined as domain facts pollute the graph with author/venue
entities and citation-title vocabulary. Detection is deterministic: a section
label from the chunk-prepare pipeline when available, otherwise content
heuristics (numbered citation runs, DOI density, year/venue patterns).

Subject-domain-agnostic but **not** language- or script-agnostic. The section
labels are English (``references``, ``bibliography``), the venue hints are
English/Latin-script abbreviations (``et al.``, ``vol.``, ``pp.``), and the
citation-marker pattern assumes Western numbered or parenthesized styles. A
German ``Literatur`` or Chinese ``参考文献`` section is not detected by label,
and author-year styles without numbered markers rely on DOI density alone.
The failure mode is a miss, not a misfire: an undetected bibliography is
extracted as ordinary content, which is the pre-existing behaviour.

Routing is decided by ``CHUNK_BIBLIOGRAPHY_MODE``:

- ``citations_only`` (default): units are marked ``is_citation_metadata`` and
  the facts renderer extracts bibliographic metadata only;
- ``skip``: units are dropped before extraction;
- ``domain_facts``: legacy behavior, no special handling.
"""

from __future__ import annotations

import re

# Section labels (from the section-label schemas) that denote reference lists.
BIBLIOGRAPHY_SECTION_LABELS = frozenset({"references", "bibliography"})

# One cited work, at the start of a line: "[12] ...", "12. ...", "(12) ...".
_CITATION_MARKER = re.compile(r"(?m)^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}[.)])\s+\S")

_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+")

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# Venue/pagination tokens common across citation styles.
_VENUE_HINT = re.compile(
    r"\bet al\.|\bvol\.\s*\d|\bpp?\.\s*\d|\bno\.\s*\d|\bdoi\b|\barXiv\b",
    re.IGNORECASE,
)


def looks_like_bibliography(text: str) -> bool:
    """Heuristic content test for reference-list chunks.

    Fires when the chunk reads as a run of citations rather than prose:
    many numbered citation markers accompanied by a comparable number of
    publication years, a run of DOIs, or a dense mix of citation markers and
    venue tokens. Thresholds are deliberately conservative — a false negative
    costs some graph noise, a false positive silences a content section.

    Args:
        text: Chunk text (markdown or plain).

    Returns:
        True when the chunk is dominated by citation entries.
    """
    stripped = text.strip()
    if len(stripped) < 200:
        return False

    markers = len(_CITATION_MARKER.findall(stripped))
    dois = len(_DOI.findall(stripped))
    years = len(_YEAR.findall(stripped))
    venue_hints = len(_VENUE_HINT.findall(stripped))
    per_kilochar = 1000.0 / max(len(stripped), 1)

    if dois >= 4 and dois * per_kilochar >= 2.0:
        return True
    if markers >= 5 and years >= markers // 2 and markers * per_kilochar >= 2.0:
        return True
    if (
        markers >= 3
        and (venue_hints + dois) >= markers
        and (markers + venue_hints) * per_kilochar >= 4.0
    ):
        return True
    return False


def is_bibliography_unit(text: str, section_label: str | None) -> bool:
    """Combine the section label (when tagged) with content heuristics."""
    if section_label is not None and section_label.lower().strip() in (
        BIBLIOGRAPHY_SECTION_LABELS
    ):
        return True
    return looks_like_bibliography(text)
