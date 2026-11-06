"""Tables of contents and lists of figures or tables, recognised from their content.

A contents entry is one line or one table row: a title, optional dotted
leaders, and a page number at the end (up to three arabic digits, or lower-case
roman). A table row is an entry only when its last cell is the page and every
other cell is text. A table whose header row is not an entry names its
columns, so it holds data, never contents.

A unit is a contents list when entries are at least
:data:`~ontocast.tool.chunk.bibliography.MIN_LIST_ENTRIES` lines, make up at
least :data:`~ontocast.tool.chunk.bibliography.CONTENT_ONLY_SHARE` of its
lines, and their arabic page numbers ascend in that share of steps.

English and Latin-script conventions; a contents list it misses is extracted,
never a section dropped.
"""

from __future__ import annotations

import re

from ontocast.tool.chunk.bibliography import CONTENT_ONLY_SHARE, MIN_LIST_ENTRIES
from ontocast.tool.chunk.structure import PLACEHOLDER

_PAGE = re.compile(r"\d{1,3}|[ivxlc]{1,7}")
_LINE_ENTRY = re.compile(r"^(?P<title>.*?)(?:\s*\.{2,}\s*|\s+)(?P<page>\S+)\s*$")
_TITLE = re.compile(r"[A-Za-z]{3,}")
_HEADING_MARK = re.compile(r"^#{1,6}\s+")
_SEPARATOR_ROW = re.compile(r"^\|?[\s:|-]+\|?$")
_NUMBER_CELL = re.compile(r"[\d\s.,:/%()$€£+-]+")

#: Longer lines are prose ending in a number, not a title and its page.
_MAX_ENTRY_CHARS = 250


def _page(token: str) -> str | None:
    return token if _PAGE.fullmatch(token) else None


def _line_page(line: str) -> str | None:
    """The page number a contents line ends with, or ``None``."""
    line = _HEADING_MARK.sub("", line)
    if len(line) > _MAX_ENTRY_CHARS:
        return None
    match = _LINE_ENTRY.match(line)
    if match is None or not _TITLE.search(match["title"]):
        return None
    return _page(match["page"])


def _row_page(row: str) -> str | None:
    """The page number a contents table row ends with, or ``None``."""
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    cells = [cell for cell in cells if cell]
    if len(cells) < 2:
        return None
    *titles, last = cells
    if any(_NUMBER_CELL.fullmatch(cell) for cell in titles):
        return None
    if not any(_TITLE.search(cell) for cell in titles):
        return None
    return _page(last)


def _entry_pages(text: str) -> list[str | None]:
    """Per counted line of ``text``, the page it ends with, or ``None``."""
    pages: list[str | None] = []
    lines = [
        line.strip() for line in PLACEHOLDER.sub("", text).splitlines() if line.strip()
    ]
    index = 0
    while index < len(lines):
        if not lines[index].startswith("|"):
            pages.append(_line_page(lines[index]))
            index += 1
            continue
        table = []
        while index < len(lines) and lines[index].startswith("|"):
            table.append(lines[index])
            index += 1
        rows = [row for row in table if not _SEPARATOR_ROW.fullmatch(row)]
        row_pages = [_row_page(row) for row in rows]
        has_header = len(table) > 1 and bool(_SEPARATOR_ROW.fullmatch(table[1]))
        if has_header and row_pages[0] is None:
            row_pages = [None] * len(rows)
        pages.extend(row_pages)
    return pages


def _ascending(pages: list[str]) -> bool:
    numbers = [int(page) for page in pages if page.isdigit()]
    steps = list(zip(numbers, numbers[1:], strict=False))
    if not steps:
        return True
    rising = sum(1 for before, after in steps if after >= before)
    return rising / len(steps) >= CONTENT_ONLY_SHARE


def looks_like_contents_list(text: str) -> bool:
    """Whether the content alone makes a unit a table of contents or a list of figures.

    Args:
        text: Unit text (markdown or plain).

    Returns:
        True when the unit is mostly contents entries with ascending pages.
    """
    pages = _entry_pages(text)
    entries = [page for page in pages if page is not None]
    return (
        len(entries) >= MIN_LIST_ENTRIES
        and len(entries) / len(pages) >= CONTENT_ONLY_SHARE
        and _ascending(entries)
    )


__all__ = ["looks_like_contents_list"]
