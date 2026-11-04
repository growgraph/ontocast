"""Reference-list detection at block level, and trimming of citation runs.

A unit is split into blocks -- paragraphs, list items and lines, with markdown
table rows as their own kind -- and each text block is tested as one cited
work. Units are judged by the share of their blocks that are citation entries,
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
from dataclasses import dataclass
from typing import Literal

from ontocast.tool.chunk.non_content import states_measurement

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

BlockKind = Literal["text", "heading", "table"]


@dataclass(frozen=True)
class Block:
    """One block of a unit, with offsets into the unit's text."""

    kind: BlockKind
    start: int
    end: int
    text: str


_BULLET = re.compile(r"^\s*[-*•·]\s+")

# One cited work at the start of a line: "[12] ...", "12. ...", "(12) ...".
_MARKER = re.compile(r"^\s*(?:\[\d{1,4}\]|\(\d{1,4}\)|\d{1,4}[.)])\s+\S")

# "Smith, J." / "Smith J" / "Smith, J. A." at the start of an entry.
_AUTHOR_SURNAME_FIRST = re.compile(
    r"^[A-Z][\w'’-]+,?\s+(?:[A-Z]\.?\s?-?){1,3}(?:[,;&]|and\b|\s)"
)
# "J. Smith" / "J.-P. Smith" at the start of an entry.
_AUTHOR_INITIALS_FIRST = re.compile(r"^(?:[A-Z]\.\s?-?){1,3}\s*[A-Z][\w'’-]+")
_ET_AL = re.compile(r"\bet\s+al\.?", re.IGNORECASE)

_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}[a-z]?(?!\d)")

_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+")
_VENUE_TOKEN = re.compile(
    r"\bvol\.\s*\d|\bpp?\.\s*\d|\bno\.\s*\d|\bdoi\b|\barXiv\b|\bISBN\b|\bProc\.",
    re.IGNORECASE,
)
# Volume and pages: "14, 193-198", "119, 3036", "30: 1806702".
_VOLUME_PAGES = re.compile(r"\b\d{1,4}\s*[,:]\s*\d{1,7}(?:\s*[–-]\s*\d{1,7})?\b")
# Two abbreviated title words in a row: "Phys. Rev.", "Nature Mater.".
_ABBREVIATED_VENUE = re.compile(r"\b[A-Z][a-z]{1,8}\.\s+[A-Z][a-z]{1,8}\.")

_HEADING = re.compile(r"^#{1,6}\s+\S")
# Converter placeholders (``<!-- image -->``) carry no text of their own.
_PLACEHOLDER = re.compile(r"<!--.*?-->", re.DOTALL)
_TABLE_ROW = re.compile(r"^\s*\|")


# A wrapped entry's tail: it opens mid-sentence, not with a capitalised word.
_CONTINUATION = re.compile(r"^\s*(?:[a-z0-9(\[]|doi\b|https?://)")


def _starts_entry(line: str) -> bool:
    return bool(_MARKER.match(_BULLET.sub("", line)) or _BULLET.match(line))


def split_blocks(text: str) -> list[Block]:
    """Split a unit into blocks: headings, table rows, and text blocks.

    Blank lines separate paragraphs. Inside a paragraph, a line that opens with
    a list bullet or a citation marker starts a new block, and lines without one
    continue the block before them, so a reference wrapped over two lines stays
    one entry. A paragraph that opens mid-sentence (lower case, a digit, a
    bracket, a DOI or URL) right after a block that opened with a marker
    continues that block too: converters split a wrapped entry that way.

    Args:
        text: Unit text (markdown or plain).

    Returns:
        Blocks in order, with offsets into ``text``.
    """
    blocks: list[Block] = []
    current: tuple[int, int] | None = None

    def close() -> None:
        nonlocal current
        if current is not None:
            start, end = current
            blocks.append(Block("text", start, end, text[start:end]))
            current = None

    offset = 0
    for line in text.splitlines(keepends=True):
        start, stripped = offset, line.strip()
        end = offset + len(line.rstrip("\r\n"))
        offset += len(line)
        if not stripped:
            close()
            continue
        if _HEADING.match(stripped) or _TABLE_ROW.match(line):
            close()
            kind: BlockKind = "heading" if _HEADING.match(stripped) else "table"
            blocks.append(Block(kind, start, end, text[start:end]))
            continue
        if current is None and _continues_entry(blocks, line):
            current = (blocks.pop().start, end)
            continue
        if current is not None and _starts_entry(line):
            close()
        current = (current[0], end) if current is not None else (start, end)
    close()
    return blocks


def _continues_entry(blocks: list[Block], line: str) -> bool:
    """Whether a new paragraph is the wrapped tail of the entry before it."""
    if not blocks or blocks[-1].kind != "text" or _starts_entry(line):
        return False
    return _starts_entry(blocks[-1].text) and bool(_CONTINUATION.match(line))


def is_citation_entry(text: str) -> bool:
    """Whether one block reads as a single cited work.

    It opens with a citation marker or an author shape, carries a publication
    year and a venue cue (DOI, venue token, volume and pages, or an abbreviated
    venue), has the length of one entry, and states no measurement. The last
    test separates "Smith et al. (2019) stored films at 85 °C ..." from a
    reference.

    Args:
        text: One block.

    Returns:
        True when the block is a citation entry.
    """
    body = " ".join(_BULLET.sub("", text).split())
    if not _MIN_ENTRY_CHARS <= len(body) <= _MAX_ENTRY_CHARS:
        return False
    opens_as_entry = bool(
        _MARKER.match(body)
        or _AUTHOR_SURNAME_FIRST.match(body)
        or _AUTHOR_INITIALS_FIRST.match(body)
        or _ET_AL.search(body[:160])
    )
    if not opens_as_entry or not _YEAR.search(body):
        return False
    has_venue = bool(
        _DOI.search(body)
        or _VENUE_TOKEN.search(body)
        or _VOLUME_PAGES.search(body)
        or _ABBREVIATED_VENUE.search(body)
    )
    return has_venue and not states_measurement(body)


def _entry_flags(blocks: list[Block]) -> list[bool]:
    return [block.kind == "text" and is_citation_entry(block.text) for block in blocks]


def citation_counts(text: str) -> tuple[int, int]:
    """``(entries, text_blocks)``: citation entries among the unit's text blocks.

    Headings and table rows are not text blocks, so a table never counts.
    """
    blocks = split_blocks(text)
    flags = _entry_flags(blocks)
    text_blocks = sum(1 for block in blocks if block.kind == "text")
    return sum(flags), text_blocks


def citation_share(text: str) -> float:
    """Citation entries divided by text blocks, in ``[0, 1]``; 0 with no text blocks."""
    entries, text_blocks = citation_counts(text)
    return entries / text_blocks if text_blocks else 0.0


def looks_like_bibliography(text: str) -> bool:
    """Whether the content alone makes a unit a reference list.

    Args:
        text: Unit text.

    Returns:
        True when at least :data:`MIN_LIST_ENTRIES` entries make up at least
        :data:`CONTENT_ONLY_SHARE` of the text blocks.
    """
    entries, text_blocks = citation_counts(text)
    return (
        entries >= MIN_LIST_ENTRIES
        and text_blocks > 0
        and entries / text_blocks >= CONTENT_ONLY_SHARE
    )


def _edge_runs(blocks: list[Block], flags: list[bool]) -> tuple[int, int]:
    """Block counts of the leading and trailing citation runs.

    Headings directly before a run belong to it ("## References" titles the
    list after it). A leading run may begin with headings too.
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
            elif blocks[index].kind == "heading":
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
        while index >= 0 and blocks[index].kind == "heading":
            index -= 1
        trailing = count - 1 - index
    return leading, trailing


def trim_citation_runs(text: str) -> list[tuple[str, bool]]:
    """Cut citation runs at a unit's start or end into pieces of their own.

    Args:
        text: Unit text.

    Returns:
        ``(piece, is_citation_run)`` pairs in document order. A unit with
        no run at either edge comes back whole as ``[(text, False)]``; a unit
        that is all one run comes back as ``[(text, True)]``.
    """
    blocks = split_blocks(text)
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
    remainder = _PLACEHOLDER.sub("", text[body_start:body_end])
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
    "Block",
    "citation_counts",
    "citation_share",
    "is_citation_entry",
    "looks_like_bibliography",
    "split_blocks",
    "trim_citation_runs",
]
