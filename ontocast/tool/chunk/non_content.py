"""Non-content unit detection: front and back matter that carries no domain facts.

Author blocks, ORCID lists, competing-interest notes, data-availability and
licence statements survive chunking as their own units whenever the section
classifier has no label for them -- the sibling sub-blocks of a labelled
``acknowledgements`` heading fall out unlabeled and are kept. Each then costs
a full render and a critic call and yields nothing but mistyped people and
identifiers. Detection is deterministic and errs toward keeping:

- a unit whose leading heading (its first line, or the most specific docling
  breadcrumb heading) names a front/back-matter section, **and** whose text
  states no unit-adjacent number (``util.measurement_lexicon``, the same
  reading the coverage inventory and the density split use), is non-content;
  a measurement anywhere in it keeps it, however it is headed;
- a unit whose tokens are mostly emails, URLs, ORCIDs and initials, and
  which states no measurement, is non-content;
- a short unit that is licence boilerplate with no measurement is
  non-content.

Like bibliography detection this is subject-domain-agnostic but English and
Latin-script bound: the heading vocabulary is English and the identifier
shapes are Western. A miss is extracted as ordinary content, which is the
pre-existing behaviour.

Routing is decided by ``CHUNK_NON_CONTENT_MODE``:

- ``extract`` (default): the unit is kept and marked ``is_non_content`` so
  downstream checks that presume domain prose can stand down;
- ``skip``: the unit is dropped before extraction.
"""

from __future__ import annotations

import re

from ontocast.config.section_labels import normalise_heading_line
from ontocast.util.measurement_lexicon import unit_adjacent_numbers

#: Section labels (from the section-label schemas) whose units are
#: front/back matter by construction.
NON_CONTENT_SECTION_LABELS = frozenset({"acknowledgements"})

#: Share of identifier-shaped tokens above which a unit is an author block.
NON_CONTENT_TOKEN_SHARE = 0.4

#: Longest unit the licence-boilerplate rule may claim; a licensing
#: discussion in body prose runs longer than a licence notice.
LICENCE_BOILERPLATE_MAX_CHARS = 1200

_MAX_HEADING_LEN = 80

# Anchored over the normalised heading line (markdown, glyphs, numbering and
# emphasis stripped), so "## ■ AUTHOR INFORMATION" reaches this as
# "AUTHOR INFORMATION".
_HEADING_ALTERNATIVES = (
    r"authors?",
    r"authors?[’']?s?\s+information",
    r"author\s+contributions?",
    r"author\s+details",
    r"affiliations?",
    r"corresponding\s+authors?",
    r"notes?",
    r"orcid(?:\s*i\.?d\.?s?)?",
    r"(?:data|code|materials?)(?:\s+and\s+(?:code|materials?|software))?"
    r"\s+availability(?:\s+statement)?",
    r"availability\s+of\s+(?:data|code|materials?)(?:\s+and\s+(?:code|materials?))?",
    r"competing\s+(?:financial\s+)?interests?(?:\s+statement)?",
    r"declarations?\s+of\s+(?:competing\s+)?interests?",
    r"conflicts?\s+of\s+interests?(?:\s+statement)?",
    r"licen[cs]e(?:\s+information)?",
    r"open\s+access",
    r"rights\s+and\s+permissions",
    r"supporting\s+information(?:\s+available)?",
    r"associated\s+content",
    r"additional\s+information",
    r"publisher[’']?s?\s+note",
    r"ethics\s+(?:declarations?|statement|approval)",
    r"funding(?:\s+(?:information|statement|sources?))?",
    r"acknowledg(?:e?ments?|ments?)",
    r"copyright",
    r"disclaimer",
)
_HEADING_RE = re.compile(
    r"^(?:" + "|".join(_HEADING_ALTERNATIVES) + r")\s*$", re.IGNORECASE
)

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


def has_non_content_heading(text: str, headings: list[str] | None) -> bool:
    """Whether the unit's leading heading names a front/back-matter section.

    Both the first line of the text and the most specific breadcrumb heading
    are tried: a unit that opens with ``## Notes`` matches by its first line,
    and the tail of a long author block matches by its breadcrumb.

    Args:
        text: Unit text.
        headings: Docling heading breadcrumb, outermost first, when known.

    Returns:
        True when either candidate normalises to a listed heading.
    """
    candidates = [first_line(text)]
    if headings:
        candidates.append(headings[-1])
    for candidate in candidates:
        if not candidate or len(candidate) > _MAX_HEADING_LEN:
            continue
        normalised = normalise_heading_line(candidate)
        if normalised and _HEADING_RE.match(normalised):
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
    text: str,
    headings: list[str] | None,
    section_label: str | None,
) -> bool:
    """Whether a prepared unit is front/back matter with no domain facts.

    Args:
        text: Unit text (markdown or plain).
        headings: Docling heading breadcrumb for the unit, when known.
        section_label: Section label from the chunk-prepare pipeline, when any.

    Returns:
        True when the unit should not be mined for domain facts.
    """
    stripped = text.strip()
    if not stripped:
        return False
    labelled = (
        section_label is not None
        and section_label.lower().strip() in NON_CONTENT_SECTION_LABELS
    )
    if labelled or has_non_content_heading(stripped, headings):
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
