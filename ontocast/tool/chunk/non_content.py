"""Non-content unit detection: front and back matter that carries no domain facts.

Author blocks, ORCID lists, competing-interest notes, data-availability and
licence statements survive chunking as their own units whenever the section
classifier gives them no strong label. Each then costs a full render and a
critic call and yields nothing but mistyped people and identifiers. Detection
reads the content only -- a section labelled as boilerplate is routed from its
schema's route class -- and errs toward keeping:

- a unit whose leading heading (its first line, or the most specific docling
  breadcrumb heading) names a section of the document's schema whose route
  class is ``non_content`` or ``boilerplate``, **and** whose text states no
  unit-adjacent number (``util.measurement_lexicon``, the same
  reading the coverage inventory and the density split use), is non-content;
  a measurement anywhere in it keeps it, however it is headed;
- a unit whose tokens are mostly emails, URLs, ORCIDs and initials, and
  which states no measurement, is non-content;
- a short unit that is licence boilerplate with no measurement is
  non-content.

The heading vocabulary is the schema's, so "Notes" is back matter in a paper
and content in a manual; without a schema (classification off, or a schema
nobody chose) only the content rules apply. The identifier shapes are Western.
A miss is extracted as ordinary content.

Routing is decided by ``CHUNK_NON_CONTENT_MODE``:

- ``extract`` (default): the unit is kept and marked ``is_non_content`` so
  downstream checks that presume domain prose can stand down;
- ``skip``: the unit is dropped before extraction.
"""

from __future__ import annotations

import re

from ontocast.config.section_labels import (
    RouteClass,
    SectionLabelSchema,
    label_route_class,
    match_heading_line,
    match_heading_name,
)
from ontocast.util.measurement_lexicon import unit_adjacent_numbers

#: Share of identifier-shaped tokens above which a unit is an author block.
NON_CONTENT_TOKEN_SHARE = 0.4

#: Longest unit the licence-boilerplate rule may claim; a licensing
#: discussion in body prose runs longer than a licence notice.
LICENCE_BOILERPLATE_MAX_CHARS = 1200

_MAX_HEADING_LEN = 80

_FRONT_BACK_MATTER = frozenset({RouteClass.NON_CONTENT, RouteClass.BOILERPLATE})

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+[.,;:)]*$")
_URL_RE = re.compile(r"^(?:https?://|www\.|doi\.org/|orcid\.org/|doi:)\S+$", re.I)
_ORCID_RE = re.compile(
    r"^(?:https?://orcid\.org/)?\d{4}-\d{4}-\d{4}-\d{3}[\dX][.,;:)]*$"
)
_INITIALS_RE = re.compile(r"^(?:[A-Z]\.-?)+[,;]?$")
#: Punctuation-only tokens (markdown glyphs, bullets, comment markers) that
#: would only dilute the share.
_NOISE_TOKEN_RE = re.compile(r"^[^\w]+$")

# Notice shapes only: a paragraph that *discusses* licences ("we compare
# Creative Commons licences") must not match, so the CC mention has to sit in
# a "licensed under" construction.
_LICENCE_RE = re.compile(
    r"(?:licen[cs]ed|distributed|published|available)\s+under\s+(?:a|an|the)\s+"
    r"creative\s+commons"
    r"|all\s+rights\s+reserved|©\s*(?:19|20)\d{2}"
    r"|\(c\)\s*(?:19|20)\d{2}|copyright\s*©?\s*(?:19|20)\d{2}",
    re.IGNORECASE,
)


def first_line(text: str) -> str:
    """The first non-empty line of ``text``, stripped; empty when there is none."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def has_non_content_heading(
    text: str, headings: list[str] | None, schema: SectionLabelSchema
) -> bool:
    """Whether the unit's leading heading names a front/back-matter section.

    Both the first line of the text and the most specific breadcrumb heading
    are tried: a unit that opens with ``## Notes`` matches by its first line,
    and the tail of a long author block matches by its breadcrumb. A heading
    counts only when it names, exactly, a label of the ``non_content`` or
    ``boilerplate`` class.

    Args:
        text: Unit text.
        headings: Docling heading breadcrumb, outermost first, when known.
        schema: The document's section schema, the source of the vocabulary.

    Returns:
        True when either candidate names such a section.
    """
    candidates = [first_line(text)]
    if headings:
        candidates.append(headings[-1])
    for candidate in candidates:
        if not candidate or len(candidate) > _MAX_HEADING_LEN:
            continue
        label = match_heading_line(candidate, schema) or match_heading_name(
            candidate, schema
        )
        if label is not None and label_route_class(label, schema) in _FRONT_BACK_MATTER:
            return True
    return False


def metadata_token_share(text: str) -> float:
    """Fraction of whitespace tokens that are emails, URLs, ORCIDs or initials.

    Args:
        text: Unit text.

    Returns:
        A value in ``[0, 1]``; 0 for empty text.
    """
    tokens = [token for token in text.split() if not _NOISE_TOKEN_RE.match(token)]
    if not tokens:
        return 0.0
    hits = sum(
        1
        for token in tokens
        if _EMAIL_RE.match(token)
        or _URL_RE.match(token)
        or _ORCID_RE.match(token)
        or _INITIALS_RE.match(token)
    )
    return hits / len(tokens)


def states_measurement(text: str) -> bool:
    """Whether ``text`` states at least one unit-adjacent number.

    A single-letter unit followed by a period is discounted: in an author
    block the affiliation digit and the initial that follows it ("Smith,1 A.
    B. Jones") read as "1 A" -- one ampere -- and that is the one shape of
    front matter this test exists to catch. A real single-letter unit ends a
    sentence rarely enough that the loss is a kept unit, not a dropped one.

    Args:
        text: Unit text.

    Returns:
        True when a measurement is stated.
    """
    for mention in unit_adjacent_numbers(text):
        if len(mention.unit) > 1:
            return True
        if text[mention.end : mention.end + 1] != ".":
            return True
    return False


def is_non_content_unit(
    text: str, headings: list[str] | None, schema: SectionLabelSchema | None
) -> bool:
    """Whether a prepared unit is front/back matter with no domain facts.

    Args:
        text: Unit text (markdown or plain).
        headings: Docling heading breadcrumb for the unit, when known.
        schema: The document's chosen section schema; ``None`` (classification
            off, or only a fallback schema) leaves the content rules alone.

    Returns:
        True when the unit should not be mined for domain facts.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if schema is not None and has_non_content_heading(stripped, headings, schema):
        if not states_measurement(stripped):
            return True
    if metadata_token_share(
        stripped
    ) >= NON_CONTENT_TOKEN_SHARE and not states_measurement(stripped):
        return True
    if (
        len(stripped) <= LICENCE_BOILERPLATE_MAX_CHARS
        and _LICENCE_RE.search(stripped)
        and not states_measurement(stripped)
    ):
        return True
    return False
