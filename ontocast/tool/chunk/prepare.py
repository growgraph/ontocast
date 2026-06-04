"""Prepare content units: segment, tag, filter, and size within section boundaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from docling_core.transforms.chunker.doc_chunk import DocMeta
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc import DoclingDocument

from ontocast.config import ChunkConfig
from ontocast.config.section_labels import (
    SectionLabelSchema,
    load_section_label_schema,
    resolve_section_schema_id,
)
from ontocast.onto.section_models import SectionSpan
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.section_llm import llm_backfill_section_labels
from ontocast.tool.chunk.sections import (
    detect_section_spans,
    document_text_for_section_tagging,
    label_from_headings,
    label_text_from_spans,
)
from ontocast.tool.chunk.segment import (
    PrepareSegment,
    coalesce_small_segments_right,
    merge_doc_item_refs,
    starts_with_section_heading,
)
from ontocast.tool.chunk.sizing import merge_small_parts

if TYPE_CHECKING:
    from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedChunk:
    """A prepared text chunk with optional structural metadata and section label."""

    text: str
    headings: list[str] | None
    doc_item_refs: tuple[str, ...] = ()
    section_label: str | None = None


# Backward-compatible alias
NormalizedChunk = PreparedChunk


@dataclass
class PrepareOptions:
    """Options for the chunk preparation pipeline."""

    section_schema_id: str | None = None
    document_type_hint: str | None = None
    target_sections: list[str] | None = None
    summarize_sections: list[str] | None = None

    def needs_section_prepare(self) -> bool:
        return self.target_sections is not None or self.summarize_sections is not None

    def filter_allowlist(self) -> list[str] | None:
        if self.target_sections is not None:
            return self.target_sections
        if (
            self.summarize_sections is not None
            and self.summarize_sections
            and "*" not in self.summarize_sections
        ):
            return self.summarize_sections
        return None


def _filter_segments(
    segments: list[PrepareSegment], allowlist: list[str] | None
) -> list[PrepareSegment]:
    if allowlist is None:
        return segments
    allowed = {section.strip().lower() for section in allowlist if section.strip()}
    if not allowed:
        return segments
    return [
        segment
        for segment in segments
        if segment.section_label is not None
        and segment.section_label.lower() in allowed
    ]


def _hybrid_segments(
    docling_doc: DoclingDocument, hybrid_chunker: HybridChunker
) -> list[PrepareSegment]:
    segments: list[PrepareSegment] = []
    for chunk in hybrid_chunker.chunk(docling_doc):
        text = chunk.text.strip()
        if not text:
            continue
        headings: list[str] | None = None
        doc_item_refs: tuple[str, ...] = ()
        meta = chunk.meta
        if isinstance(meta, DocMeta):
            headings = meta.headings
            doc_item_refs = tuple(item.self_ref for item in meta.doc_items)
        segments.append(
            PrepareSegment(
                text=text,
                headings=headings,
                doc_item_refs=doc_item_refs,
            )
        )
    return segments


def _semantic_full_doc_segments(
    document_text: str, splitter: ChunkerTool
) -> list[PrepareSegment]:
    text = document_text.strip()
    if not text:
        return []
    return [
        PrepareSegment(text=part.strip()) for part in splitter(text) if part.strip()
    ]


def _tag_segments(
    segments: list[PrepareSegment],
    document_text: str,
    spans: list[SectionSpan],
    schema: "SectionLabelSchema",
) -> None:
    """Assign section_label to each segment.

    Strategy (cheapest to most expensive):
    1. Docling heading breadcrumb — reliable structural metadata, no text search.
    2. Character-span overlap against the markdown export — catches segments
       whose text does match the markdown representation.

    Span-search cursor is preserved when a segment text is not found so
    subsequent segments are not mis-anchored to the document start.
    """
    search_from = 0
    for segment in segments:
        heading_label = label_from_headings(segment.headings, schema)
        if heading_label is not None:
            segment.section_label = heading_label
            # Still advance the search cursor so span-based tagging stays
            # ordered for segments that do need it.
            _, search_from = label_text_from_spans(
                segment.text, document_text, spans, search_from
            )
        else:
            label, search_from = label_text_from_spans(
                segment.text, document_text, spans, search_from
            )
            segment.section_label = label


def _forward_fill_section_labels(
    segments: list[PrepareSegment],
    schema: "SectionLabelSchema",
) -> None:
    """Propagate the nearest preceding label to unlabeled segments.

    Propagation is blocked when the unlabeled segment opens with a recognised
    section heading of its own — that signals a new section where the preceding
    label would be wrong.
    """
    last_label: str | None = None
    fill_count = 0
    for segment in segments:
        if segment.section_label is not None:
            last_label = segment.section_label
        elif last_label is not None and not starts_with_section_heading(
            segment, schema
        ):
            segment.section_label = last_label
            fill_count += 1
    if fill_count:
        logger.debug(
            "Forward-filled %s segment(s) with nearest preceding label", fill_count
        )


def _expand_segment(
    segment: PrepareSegment,
    splitter: ChunkerTool,
    config: ChunkConfig,
) -> list[PreparedChunk]:
    max_size = config.max_size
    if len(segment.text) <= max_size:
        return [
            PreparedChunk(
                text=segment.text,
                headings=segment.headings,
                doc_item_refs=segment.doc_item_refs,
                section_label=segment.section_label,
            )
        ]

    pieces: list[PreparedChunk] = []
    for sub_text in splitter(segment.text):
        sub_text = sub_text.strip()
        if not sub_text:
            continue
        sized_texts = (
            splitter.size_text(sub_text) if len(sub_text) > max_size else [sub_text]
        )
        for sized_text in sized_texts:
            pieces.append(
                PreparedChunk(
                    text=sized_text,
                    headings=segment.headings,
                    doc_item_refs=segment.doc_item_refs,
                    section_label=segment.section_label,
                )
            )
    return pieces


def _merge_prepared_chunks(
    chunks: list[PreparedChunk],
    min_size: int,
    max_size: int,
) -> list[PreparedChunk]:
    if not chunks:
        return []

    merged: list[PreparedChunk] = []
    index = 0
    while index < len(chunks):
        label = chunks[index].section_label
        run: list[PreparedChunk] = []
        while index < len(chunks) and chunks[index].section_label == label:
            run.append(chunks[index])
            index += 1

        texts = merge_small_parts(
            [chunk.text for chunk in run],
            min_size,
            max_size,
        )
        headings = next((chunk.headings for chunk in run if chunk.headings), None)
        refs: tuple[str, ...] = ()
        for chunk in run:
            refs = merge_doc_item_refs(refs, chunk.doc_item_refs)

        for text in texts:
            merged.append(
                PreparedChunk(
                    text=text,
                    headings=headings,
                    doc_item_refs=refs,
                    section_label=label,
                )
            )
    return merged


def _size_segments(
    segments: list[PrepareSegment],
    splitter: ChunkerTool,
    config: ChunkConfig,
) -> list[PreparedChunk]:
    expanded: list[PreparedChunk] = []
    for segment in segments:
        expanded.extend(_expand_segment(segment, splitter, config))
    return _merge_prepared_chunks(expanded, config.min_size, config.max_size)


def _simple_prepare(
    docling_doc: DoclingDocument,
    document_text: str,
    splitter: ChunkerTool,
    config: ChunkConfig,
    hybrid_chunker: HybridChunker,
) -> list[PreparedChunk]:
    segments = _hybrid_segments(docling_doc, hybrid_chunker)
    if not segments:
        text = document_text.strip()
        if not text:
            return []
        segments = [PrepareSegment(text=part) for part in splitter.size_text(text)]
    return _size_segments(segments, splitter, config)


async def prepare_content_units(
    docling_doc: DoclingDocument,
    splitter: ChunkerTool,
    config: ChunkConfig,
    options: PrepareOptions,
    tools: "ToolBox",
) -> list[PreparedChunk]:
    """Segment, tag, filter, and size document text into prepared chunks."""
    document_text = document_text_for_section_tagging(docling_doc)
    hybrid_chunker = HybridChunker()

    if not options.needs_section_prepare():
        return _simple_prepare(
            docling_doc, document_text, splitter, config, hybrid_chunker
        )

    schema_id = resolve_section_schema_id(
        section_schema_id=options.section_schema_id,
        document_type_hint=options.document_type_hint,
    )
    schema = load_section_label_schema(schema_id)
    spans = detect_section_spans(document_text, schema)

    segments = _hybrid_segments(docling_doc, hybrid_chunker)
    if not segments:
        segments = _semantic_full_doc_segments(document_text, splitter)

    if not segments:
        return []

    segments = coalesce_small_segments_right(
        segments,
        config.section_tag_min_chars,
        schema,
    )
    _tag_segments(segments, document_text, spans, schema)
    await llm_backfill_section_labels(
        segments,
        tools,
        section_schema_id=options.section_schema_id,
        document_type_hint=options.document_type_hint,
        section_tag_min_chars=config.section_tag_min_chars,
    )
    _forward_fill_section_labels(segments, schema)

    unlabeled = sum(1 for s in segments if s.section_label is None)
    if unlabeled:
        logger.warning(
            "%s segment(s) remain without section_label after LLM backfill",
            unlabeled,
        )

    allowlist = options.filter_allowlist()
    if allowlist is not None:
        before = len(segments)
        segments = _filter_segments(segments, allowlist)
        logger.info(
            "Section filter %s: kept %s/%s segments before sizing",
            allowlist,
            len(segments),
            before,
        )
        if before > 0 and not segments:
            logger.warning(
                "Section filter %s removed all segments; check headings or allowlist",
                allowlist,
            )

    return _size_segments(segments, splitter, config)
