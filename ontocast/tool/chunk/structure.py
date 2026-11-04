"""Document structure: the blocks of the markdown export and their kinds.

The markdown export is the document text everything else reads, so blocks are
cut from it by markdown syntax: blank lines separate paragraphs, list bullets
and citation markers start blocks, and heading lines and table rows are blocks
of their own. A converted document adds what syntax cannot show: each docling
item is serialized as the export serializes it, and a block whose text equals
an item's text takes that item's kind and reference. So a caption, a footnote
or a formula is known for what it is, and a unit can name the items it holds.

A block no item matches -- plain-text input, a paragraph a unit boundary cut
through, text the converter joined differently -- keeps its syntax kind.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from ontocast.util.optional import require

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument, NodeItem


class BlockKind(StrEnum):
    """What a block of the export is."""

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    CODE = "code"
    PICTURE = "picture"


class ItemBlock(NamedTuple):
    """The kind and docling reference of an item, keyed by its block text."""

    kind: BlockKind
    ref: str


#: Normalised block text -> the docling item it came from.
BlockKinds = Mapping[str, ItemBlock]


@dataclass(frozen=True)
class Block:
    """One block of a text, with offsets into that text."""

    kind: BlockKind
    start: int
    end: int
    text: str
    ref: str | None = None


@dataclass(frozen=True)
class DocumentStructure:
    """The document text and the docling kinds of its blocks."""

    text: str
    block_kinds: BlockKinds = field(default_factory=dict)


#: A list bullet at the start of a line.
LIST_BULLET = re.compile(r"^\s*[-*•·]\s+")

#: One cited work at the start of a line: "[12] ...", "12. ...", "(12) ...".
ENTRY_MARKER = re.compile(r"^\s*(?:\[\d{1,4}\]|\(\d{1,4}\)|\d{1,4}[.)])\s+\S")

_HEADING = re.compile(r"^#{1,6}\s+\S")
_TABLE_ROW = re.compile(r"^\s*\|")
# Converter placeholders (``<!-- image -->``) carry no text of their own.
PLACEHOLDER = re.compile(r"<!--.*?-->", re.DOTALL)

# A wrapped entry's tail: it opens mid-sentence, not with a capitalised word.
_CONTINUATION = re.compile(r"^\s*(?:[a-z0-9(\[]|doi\b|https?://)")

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _starts_entry(line: str) -> bool:
    return bool(
        ENTRY_MARKER.match(LIST_BULLET.sub("", line)) or LIST_BULLET.match(line)
    )


def _continues_entry(blocks: list[Block], line: str) -> bool:
    """Whether a new paragraph is the wrapped tail of the entry before it."""
    if not blocks or blocks[-1].kind != BlockKind.TEXT or _starts_entry(line):
        return False
    return _starts_entry(blocks[-1].text) and bool(_CONTINUATION.match(line))


def _syntax_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    current: tuple[int, int] | None = None

    def close() -> None:
        nonlocal current
        if current is not None:
            start, end = current
            body = text[start:end]
            kind = (
                BlockKind.PICTURE
                if not PLACEHOLDER.sub("", body).strip()
                else BlockKind.TEXT
            )
            blocks.append(Block(kind, start, end, body))
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
            kind = BlockKind.HEADING if _HEADING.match(stripped) else BlockKind.TABLE
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


def split_blocks(text: str, kinds: BlockKinds | None = None) -> list[Block]:
    """Split a text into blocks, typed by docling items where they match.

    Blank lines separate paragraphs. Inside a paragraph, a line that opens with
    a list bullet or a citation marker starts a new block, and lines without one
    continue the block before them, so a reference wrapped over two lines stays
    one entry. A paragraph that opens mid-sentence (lower case, a digit, a
    bracket, a DOI or URL) right after a block that opened with a marker
    continues that block too: converters split a wrapped entry that way.
    Heading lines and table rows are blocks of their own; a paragraph of
    placeholders only is a picture.

    Args:
        text: Markdown or plain text.
        kinds: Docling kinds by block text (:func:`docling_block_kinds`). A
            matching text or picture block takes the item's kind; headings and
            table rows keep theirs. Every match records the item reference.

    Returns:
        Blocks in order, with offsets into ``text``.
    """
    blocks = _syntax_blocks(text)
    if not kinds:
        return blocks
    typed: list[Block] = []
    for block in blocks:
        item = kinds.get(_normalise(block.text))
        if item is None:
            typed.append(block)
            continue
        kind = (
            item.kind
            if block.kind in (BlockKind.TEXT, BlockKind.PICTURE)
            else block.kind
        )
        typed.append(Block(kind, block.start, block.end, block.text, item.ref))
    return typed


def item_refs(text: str, kinds: BlockKinds | None) -> tuple[str, ...]:
    """References of the docling items whose blocks ``text`` holds whole, in order."""
    refs: dict[str, None] = {}
    for block in split_blocks(text, kinds):
        if block.ref is not None:
            refs.setdefault(block.ref)
    return tuple(refs)


def _kind_of_label(label: str) -> BlockKind:
    return {
        "section_header": BlockKind.HEADING,
        "title": BlockKind.HEADING,
        "caption": BlockKind.CAPTION,
        "footnote": BlockKind.FOOTNOTE,
        "formula": BlockKind.FORMULA,
        "code": BlockKind.CODE,
        "picture": BlockKind.PICTURE,
        "chart": BlockKind.PICTURE,
        "table": BlockKind.TABLE,
    }.get(label, BlockKind.TEXT)


def _inside_inline_group(item: NodeItem, doc: DoclingDocument) -> bool:
    doc_types = require("docling_core.types.doc", feature="Document structure")
    parent = item.parent.resolve(doc) if item.parent else None
    while parent is not None:
        if (
            isinstance(parent, doc_types.GroupItem)
            and parent.label == doc_types.GroupLabel.INLINE
        ):
            return True
        parent = parent.parent.resolve(doc) if parent.parent else None
    return False


def docling_block_kinds(doc: DoclingDocument) -> dict[str, ItemBlock]:
    """Kind and reference of each body item, keyed by its normalised block text.

    Items are serialized one by one with the serializer the markdown export
    uses, so their paragraphs are the export's blocks. An inline group is one
    item (its parts serialize differently on their own); a picture or table
    contributes its captions. The first item wins when two share a text.
    """
    doc_types = require("docling_core.types.doc", feature="Document structure")
    markdown = require(
        "docling_core.transforms.serializer.markdown", feature="Document structure"
    )
    serializer = markdown.MarkdownDocSerializer(doc=doc)
    kinds: dict[str, ItemBlock] = {}

    def register(text: str, kind: BlockKind, ref: str) -> None:
        for paragraph in _PARAGRAPH_BREAK.split(text):
            key = _normalise(paragraph)
            if key:
                kinds.setdefault(key, ItemBlock(kind, ref))

    for item, _depth in doc.iterate_items(with_groups=True):
        if _inside_inline_group(item, doc):
            continue
        if isinstance(item, doc_types.GroupItem):
            if item.label != doc_types.GroupLabel.INLINE:
                continue
            kind = BlockKind.TEXT
        elif isinstance(item, doc_types.FloatingItem) and not isinstance(
            item, doc_types.TextItem
        ):
            for caption in item.captions:
                resolved = caption.resolve(doc)
                register(resolved.text, BlockKind.CAPTION, resolved.self_ref)
            continue
        else:
            kind = _kind_of_label(str(item.label))
        register(serializer.serialize(item=item).text, kind, item.self_ref)
    return kinds


def build_document_structure(doc: DoclingDocument) -> DocumentStructure:
    """The markdown export of ``doc`` and the docling kinds of its blocks."""
    return DocumentStructure(
        text=doc.export_to_markdown(), block_kinds=docling_block_kinds(doc)
    )


__all__ = [
    "ENTRY_MARKER",
    "LIST_BULLET",
    "PLACEHOLDER",
    "Block",
    "BlockKind",
    "BlockKinds",
    "DocumentStructure",
    "ItemBlock",
    "build_document_structure",
    "docling_block_kinds",
    "item_refs",
    "split_blocks",
]
