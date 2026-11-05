"""Reference-list detection at block level, and trimming of citation runs.

A unit is split into blocks (:func:`ontocast.tool.chunk.structure.split_blocks`)
and each text or footnote block is tested as one cited work, or as part of one
when an entry runs over several blocks; captions, formulas, code, pictures and
table rows never are. Units are judged by the share of their blocks that are citation entries,
not by counting citation tokens over the whole unit, so a table that cites its
sources or a discussion that enumerates prior studies is not a reference list.

A reference list does not sit inside prose, so only a run of entries at a
unit's start or end is trimmed into a unit of its own; a run in the middle is
left alone.

Subject-domain-agnostic but **not** language- or script-agnostic: markers,
``et al.``, venue abbreviations and the author shapes are Western conventions,
and author-year styles that put the year last rely on the venue cue alone. The
failure mode is a miss -- a reference list extracted as content -- never a
dropped section.
"""

from __future__ import annotations

import re

from ontocast.tool.chunk.non_content import states_measurement
from ontocast.tool.chunk.structure import (
    ENTRY_MARKER,
    LIST_BULLET,
    PLACEHOLDER,
    Block,
    BlockKind,
    BlockKinds,
    split_blocks,
)

#: Share of a unit's text blocks that must be citation entries for the content
#: alone to make it a reference list. A reference list is nearly all entries.
CONTENT_ONLY_SHARE = 0.8

#: Share that suffices when a strong section label already says "reference
#: list": the list may carry a short preamble or notes.
LABEL_AGREEMENT_SHARE = 0.3

#: Fewest entries for content alone to call a unit a reference list.
MIN_LIST_ENTRIES = 3

#: Fewest contiguous entries at a unit's edge that are trimmed into their own unit.
MIN_TRIM_ENTRIES = 5

_MIN_ENTRY_CHARS = 25
_MAX_ENTRY_CHARS = 600

#: Blocks that can be one cited work.
_ENTRY_KINDS = frozenset({BlockKind.TEXT, BlockKind.FOOTNOTE})

#: Blocks that count against a citation share: everything that carries text
#: but a heading or a table row.
_COUNTED_KINDS = _ENTRY_KINDS | {BlockKind.CAPTION, BlockKind.FORMULA, BlockKind.CODE}

#: Blocks a citation run passes over: they title or decorate the list.
_RUN_TRANSPARENT_KINDS = frozenset({BlockKind.HEADING, BlockKind.PICTURE})

# An entry's own marker: "[12]", "(12)", "12.", "12)", a bracketed label
# ("[RFC3986]"), or a bare number before a capital ("17 Y. Hamada").
_MARKER = re.compile(
    r"^(?:\[[^\]\s]{1,40}\]|\(\d{1,4}\)|\d{1,4}[.)]|\d{1,4}(?=\s+[A-Z]))\s+(?=\S)"
)
# "Smith, J." / "Smith J" / "Smith, J. A." at the start of an entry.
_AUTHOR_SURNAME_FIRST = re.compile(
    r"^[A-Z][\w'’-]+,?\s+(?:[A-Z]\.?\s?-?){1,3}(?:[,;&]|and\b|\s)"
)
# "J. Smith" / "J.-P. Smith" at the start of an entry.
_AUTHOR_INITIALS_FIRST = re.compile(r"^(?:[A-Z]\.\s?-?){1,3}\s*[A-Z][\w'’-]+")
_ET_AL = re.compile(r"\bet\s+al\.?", re.IGNORECASE)
# One word of an author list: a capitalised name or initials, a connective, or
# a lower-case surname particle.
_NAME_WORD = re.compile(
    r"^(?:[A-Z][\w'’.-]*|(?:[A-Z]\.-?)+|and|&|et|al\.?|Eds?\.?"
    r"|van|von|de|der|den|da|di|du|la|le)$"
)
_NAME_SEPARATOR = re.compile(r"[\s,;]+")
# The longest author list read before the year ("Ana Lopez, Ben Okafor, ... 2018.").
_MAX_AUTHOR_LIST_CHARS = 300

_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}[a-z]?(?!\d)")

_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+")
_VENUE_TOKEN = re.compile(
    r"\bvol\.\s*\d|\bpp?\.\s*\d|\bno\.\s*\d|\bdoi\b|\barXiv\b|\bISBN\b|\bProc\.",
    re.IGNORECASE,
)
# Case matters here: "In EMNLP" names a venue, "in prior work" does not.
_VENUE_PHRASE = re.compile(
    r"(?:^|[.,]\s)In:?\s+(?:Proc\b|Proceedings\b|[A-Z]{2,}(?:\s+\d{4})?\.)"
    r"|\b[Pp]ages\s+\d+\s*[–-]\s*\d"
    r"|\bURL\s+https?://"
    r"|\bAvailable\s+(?:online|from|at)\b"
    r"|\bRFC\s?\d{3,5}\b"
)
# Volume and pages, with an optional issue: "14, 193-198", "119, 3036",
# "30: 1806702", "17(2), 87-96", "6(Nov):1817-1853", "140, A1133".
_VOLUME_PAGES = re.compile(
    r"\b\d{1,4}\s*(?:\([^()]{1,12}\))?\s*[,:]\s*[A-Z]?\d{1,7}"
    r"(?:\s*[–-]\s*[A-Z]?\d{1,7})?\b"
)
# Year, volume and pages: "2013;310(20):2191-4".
_YEAR_VOLUME = re.compile(r"(?<!\d)(?:19|20)\d{2}\s*;\s*\d")
# Two abbreviated title words in a row: "Phys. Rev.", "Nature Mater.".
_ABBREVIATED_VENUE = re.compile(r"\b[A-Z][a-z]{1,8}\.\s+[A-Z][a-z]{1,8}\.")
# Link targets and addresses: their digits and percent signs are no measurement.
_ADDRESS = re.compile(
    r"\]\([^)]*\)|<[^>\s]+>|&lt;\S+?&gt;|\bhttps?://\S+|\bwww\.\S+|\b10\.\d{4,9}/\S+"
)


def _body(text: str) -> str:
    return " ".join(LIST_BULLET.sub("", text).split())


def _author_list_before_year(body: str) -> bool:
    """Whether everything before the first year is a list of names."""
    year = _YEAR.search(body)
    if year is None or year.start() > _MAX_AUTHOR_LIST_CHARS:
        return False
    words = [
        word
        for word in _NAME_SEPARATOR.split(body[: year.start()].strip(" .,;:("))
        if word
    ]
    return len(words) >= 2 and all(_NAME_WORD.match(word) for word in words)


def _opens_entry(body: str) -> bool:
    """A citation marker or an author shape at the start of the block."""
    marker = _MARKER.match(body) or ENTRY_MARKER.match(body)
    if marker is not None:
        return True
    return bool(
        _AUTHOR_SURNAME_FIRST.match(body)
        or _AUTHOR_INITIALS_FIRST.match(body)
        or _author_list_before_year(body)
        or _ET_AL.search(body[:160])
    )


def is_citation_entry(text: str) -> bool:
    """Whether one block reads as a single cited work.

    It opens with a citation marker or an author shape, carries a publication
    year and a venue cue (DOI, venue token or phrase, volume and pages, or an
    abbreviated venue), has the length of one entry, and states no measurement
    outside its links and DOIs. The last test separates "Smith et al. (2019)
    stored films at 85 °C ..." from a reference.

    Args:
        text: One block.

    Returns:
        True when the block is a citation entry.
    """
    body = _body(text)
    if not _MIN_ENTRY_CHARS <= len(body) <= _MAX_ENTRY_CHARS:
        return False
    if not _opens_entry(body) or not _YEAR.search(body):
        return False
    has_venue = bool(
        _DOI.search(body)
        or _VENUE_TOKEN.search(body)
        or _VENUE_PHRASE.search(body)
        or _VOLUME_PAGES.search(body)
        or _YEAR_VOLUME.search(body)
        or _ABBREVIATED_VENUE.search(body)
    )
    return has_venue and not states_measurement(_ADDRESS.sub(" ", body))


def _entry_flags(blocks: list[Block]) -> list[bool]:
    """Per block, whether it is a citation entry or part of one.

    An entry runs from one opener to the next: a page break or a wrapped line
    can leave its year or venue in the following block. A block that opens an
    entry but is none on its own is joined with the blocks after it that open
    nothing, and all of them count when the joined text is an entry.
    """
    flags = [
        block.kind in _ENTRY_KINDS and is_citation_entry(block.text) for block in blocks
    ]

    def continues(index: int) -> bool:
        block = blocks[index]
        return (
            block.kind in _ENTRY_KINDS
            and not flags[index]
            and not _opens_entry(_body(block.text))
        )

    start = 0
    while start < len(blocks):
        block = blocks[start]
        if (
            flags[start]
            or block.kind not in _ENTRY_KINDS
            or not _opens_entry(_body(block.text))
        ):
            start += 1
            continue
        joined = block.text
        end = start + 1
        while end < len(blocks) and continues(end):
            joined = f"{joined} {blocks[end].text}"
            end += 1
            if len(_body(joined)) > _MAX_ENTRY_CHARS:
                break
            if is_citation_entry(joined):
                flags[start:end] = [True] * (end - start)
                break
        start = end
    return flags


def citation_counts(text: str, kinds: BlockKinds | None = None) -> tuple[int, int]:
    """``(entries, counted_blocks)``: citation entries among the unit's text blocks.

    Headings, table rows and pictures are not counted, so a table never counts;
    captions, formulas and code count but are never entries.

    Args:
        text: Unit text.
        kinds: Docling block kinds of the document, when it was converted.
    """
    blocks = split_blocks(text, kinds)
    flags = _entry_flags(blocks)
    counted = sum(1 for block in blocks if block.kind in _COUNTED_KINDS)
    return sum(flags), counted


def citation_share(text: str, kinds: BlockKinds | None = None) -> float:
    """Citation entries divided by counted blocks, in ``[0, 1]``; 0 with none."""
    entries, counted = citation_counts(text, kinds)
    return entries / counted if counted else 0.0


def looks_like_bibliography(text: str, kinds: BlockKinds | None = None) -> bool:
    """Whether the content alone makes a unit a reference list.

    Args:
        text: Unit text.
        kinds: Docling block kinds of the document, when it was converted.

    Returns:
        True when at least :data:`MIN_LIST_ENTRIES` entries make up at least
        :data:`CONTENT_ONLY_SHARE` of the counted blocks.
    """
    entries, counted = citation_counts(text, kinds)
    return (
        entries >= MIN_LIST_ENTRIES
        and counted > 0
        and entries / counted >= CONTENT_ONLY_SHARE
    )


def _edge_runs(blocks: list[Block], flags: list[bool]) -> tuple[int, int]:
    """Block counts of the leading and trailing citation runs.

    Headings and pictures directly before a run belong to it ("## References"
    titles the list after it). A leading run may begin with them too.
    """

    def run_length(order: list[int]) -> int:
        entries = 0
        length = 0
        pending_headings = 0
        for index in order:
            if flags[index]:
                entries += 1
                length += 1 + pending_headings
                pending_headings = 0
            elif blocks[index].kind in _RUN_TRANSPARENT_KINDS:
                pending_headings += 1
            else:
                break
        if entries < MIN_TRIM_ENTRIES:
            return 0
        return length

    count = len(blocks)
    leading = run_length(list(range(count)))
    trailing = 0
    # Walk backwards over entries, then claim the headings just before them.
    index = count - 1
    entries = 0
    while index >= 0 and flags[index]:
        entries += 1
        index -= 1
    if entries >= MIN_TRIM_ENTRIES:
        while index >= 0 and blocks[index].kind in _RUN_TRANSPARENT_KINDS:
            index -= 1
        trailing = count - 1 - index
    return leading, trailing


def trim_citation_runs(
    text: str, kinds: BlockKinds | None = None
) -> list[tuple[str, bool]]:
    """Cut citation runs at a unit's start or end into pieces of their own.

    Args:
        text: Unit text.
        kinds: Docling block kinds of the document, when it was converted.

    Returns:
        ``(piece, is_citation_run)`` pairs in document order. A unit with
        no run at either edge comes back whole as ``[(text, False)]``; a unit
        that is all one run comes back as ``[(text, True)]``.
    """
    blocks = split_blocks(text, kinds)
    if not blocks:
        return [(text, False)]
    flags = _entry_flags(blocks)
    leading, trailing = _edge_runs(blocks, flags)
    if leading + trailing >= len(blocks) or leading == len(blocks):
        return [(text, True)] if (leading or trailing) else [(text, False)]
    if not leading and not trailing:
        return [(text, False)]

    pieces: list[tuple[str, bool]] = []
    body_start = blocks[leading].start if leading else 0
    body_end = blocks[len(blocks) - trailing - 1].end if trailing else len(text)
    remainder = PLACEHOLDER.sub("", text[body_start:body_end])
    if len(" ".join(remainder.split())) < _MIN_ENTRY_CHARS:
        # Nothing but placeholders or a stray line beside the run: not prose.
        return [(text, True)]
    if leading:
        pieces.append((text[: blocks[leading - 1].end].strip(), True))
    pieces.append((text[body_start:body_end].strip(), False))
    if trailing:
        pieces.append((text[blocks[len(blocks) - trailing].start :].strip(), True))
    return [(piece, is_run) for piece, is_run in pieces if piece]


__all__ = [
    "CONTENT_ONLY_SHARE",
    "LABEL_AGREEMENT_SHARE",
    "MIN_LIST_ENTRIES",
    "MIN_TRIM_ENTRIES",
    "citation_counts",
    "citation_share",
    "is_citation_entry",
    "looks_like_bibliography",
    "trim_citation_runs",
]
