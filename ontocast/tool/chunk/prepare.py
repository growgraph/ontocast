"""Prepare content units: segment, tag, filter, and size within section boundaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
from ontocast.util.optional import require

if TYPE_CHECKING:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.types.doc import DoclingDocument

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
    exclude_sections: list[str] | None = None

    def needs_section_prepare(self) -> bool:
        """True when a request option explicitly requires section labels.

        Section tagging itself is default-on (see ``CHUNK_SECTION_CLASSIFIER``);
        this only reports whether the request carries section-dependent options.
        """
        return (
            self.target_sections is not None
            or self.summarize_sections is not None
            or self.exclude_sections is not None
        )

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

    def filter_denylist(self, schema: SectionLabelSchema) -> list[str]:
        """Effective exclusion denylist.

        ``None`` means "use the resolved schema's default_exclude"; an explicit
        ``[]`` opts out of exclusion entirely; a non-empty list is used as-is.
        """
        if self.exclude_sections is not None:
            return list(self.exclude_sections)
        return list(schema.default_exclude)


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


def _filter_segments_excluding(
    segments: list[PrepareSegment], denylist: list[str]
) -> list[PrepareSegment]:
    """Drop labeled segments whose label is in the denylist; keep unlabeled ones."""
    denied = {section.strip().lower() for section in denylist if section.strip()}
    if not denied:
        return segments
    kept = [
        segment
        for segment in segments
        if segment.section_label is None or segment.section_label.lower() not in denied
    ]
    dropped = len(segments) - len(kept)
    if dropped:
        logger.info(
            "Section exclusion %s: dropped %s/%s segment(s) before sizing",
            sorted(denied),
            dropped,
            len(segments),
        )
    return kept


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
        doc_meta_cls = require(
            "docling_core.transforms.chunker.doc_chunk", feature="Hybrid chunking"
        ).DocMeta
        if isinstance(meta, doc_meta_cls):
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
    document_text: str, splitter: ChunkerTool, max_size: int
) -> list[PrepareSegment]:
    text = document_text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [PrepareSegment(text=text)]
    return [
        PrepareSegment(text=part.strip()) for part in splitter(text) if part.strip()
    ]


def _heading_breadcrumb(block_text: str) -> list[str] | None:
    """First line of the block when it is a markdown heading, as a breadcrumb."""
    for line in block_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return [stripped.lstrip("#").strip()]
        return None
    return None


def _section_blocks(
    document_text: str, spans: list[SectionSpan]
) -> list[tuple[str | None, str]]:
    """Cover the document with ordered ``(label, text)`` blocks.

    Detected section spans become labeled blocks; text between/around spans
    becomes unlabeled blocks (later labeled by the LLM classifier, when on).
    """
    blocks: list[tuple[str | None, str]] = []
    if not spans:
        return [(None, document_text)]
    cursor = 0
    for span in sorted(spans, key=lambda item: item.start):
        if span.start > cursor:
            blocks.append((None, document_text[cursor : span.start]))
        blocks.append((span.label, document_text[span.start : span.end]))
        cursor = max(cursor, span.end)
    if cursor < len(document_text):
        blocks.append((None, document_text[cursor:]))
    return blocks


def _semantic_section_segments(
    document_text: str,
    spans: list[SectionSpan],
    splitter: ChunkerTool,
    max_size: int,
) -> list[PrepareSegment]:
    """Sections-first segmentation: split at section boundaries, then chunk within.

    Chunking inside each section block means no chunk straddles a section
    boundary and every chunk from a detected section inherits its label
    deterministically — the LLM classifier only handles unheaded material.
    Blocks already within the chunk budget are kept whole (the semantic
    splitter needs enough sentences to embed and cluster).
    """
    if not document_text.strip():
        return []
    segments: list[PrepareSegment] = []
    for label, block_text in _section_blocks(document_text, spans):
        block_text = block_text.strip()
        if not block_text:
            continue
        headings = _heading_breadcrumb(block_text)
        parts = [block_text] if len(block_text) <= max_size else splitter(block_text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            segments.append(
                PrepareSegment(
                    text=part,
                    headings=headings,
                    section_label=label,
                )
            )
    return segments


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
        if segment.section_label is not None:
            # Sections-first segmentation labels at split time; keep the label
            # but advance the cursor so span search stays ordered.
            _, search_from = label_text_from_spans(
                segment.text, document_text, spans, search_from
            )
            continue
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


def _build_hybrid_chunker(config: ChunkConfig) -> HybridChunker:
    """Docling HybridChunker sized to our chunk budget.

    A bare ``HybridChunker()`` inherits the MiniLM tokenizer's 512-token limit
    while OntoCast re-merges segments up to ``config.max_size`` chars, which
    floods logs with "headers and captions … will be ignored" warnings. Budget
    the tokenizer from the configured chunk size instead (~4 chars per token).
    """
    hybrid_module = require(
        "docling_core.transforms.chunker.hybrid_chunker", feature="Hybrid chunking"
    )
    try:
        tokenizer_module = require(
            "docling_core.transforms.chunker.tokenizer.huggingface",
            feature="Hybrid chunking",
        )
        transformers_module = require("transformers", feature="Hybrid chunking")
        max_tokens = max(512, config.max_size // 4)
        hf_tokenizer = transformers_module.AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            model_max_length=max_tokens,
        )
        tokenizer = tokenizer_module.HuggingFaceTokenizer(
            tokenizer=hf_tokenizer, max_tokens=max_tokens
        )
        return hybrid_module.HybridChunker(tokenizer=tokenizer)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("Falling back to default HybridChunker tokenizer: %s", exc)
        return hybrid_module.HybridChunker()


def _primary_segments(
    docling_doc: DoclingDocument,
    document_text: str,
    spans: list[SectionSpan],
    splitter: ChunkerTool,
    config: ChunkConfig,
) -> list[PrepareSegment]:
    """Segment via the configured segmenter, falling back to the other one."""
    if config.segmenter == "docling":
        segments = _hybrid_segments(docling_doc, _build_hybrid_chunker(config))
        if not segments:
            segments = _semantic_section_segments(
                document_text, spans, splitter, config.max_size
            )
        return segments
    segments = _semantic_section_segments(
        document_text, spans, splitter, config.max_size
    )
    if not segments:
        segments = _hybrid_segments(docling_doc, _build_hybrid_chunker(config))
    return segments


def _simple_prepare(
    docling_doc: DoclingDocument,
    document_text: str,
    splitter: ChunkerTool,
    config: ChunkConfig,
) -> list[PreparedChunk]:
    """Untagged preparation (section classifier off)."""
    if config.segmenter == "docling":
        segments = _hybrid_segments(docling_doc, _build_hybrid_chunker(config))
    else:
        segments = _semantic_full_doc_segments(document_text, splitter, config.max_size)
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
    """Segment, tag, filter, and size document text into prepared chunks.

    Section tagging is default-on: the sections-first flow runs unless
    ``CHUNK_SECTION_CLASSIFIER=off`` (which also disables section filters and
    schema default exclusions; explicit section options are ignored with a
    warning in that case).
    """
    document_text = document_text_for_section_tagging(docling_doc)

    if config.section_classifier == "off":
        if options.needs_section_prepare():
            logger.warning(
                "Section options requested but CHUNK_SECTION_CLASSIFIER=off; "
                "section filters are ignored"
            )
        return _simple_prepare(docling_doc, document_text, splitter, config)

    schema_id = resolve_section_schema_id(
        section_schema_id=options.section_schema_id,
        document_type_hint=options.document_type_hint,
    )
    schema = load_section_label_schema(schema_id)
    spans = detect_section_spans(document_text, schema)

    segments = _primary_segments(docling_doc, document_text, spans, splitter, config)
    if not segments:
        return []

    segments = coalesce_small_segments_right(
        segments,
        config.section_tag_min_chars,
        schema,
    )
    _tag_segments(segments, document_text, spans, schema)
    if config.section_classifier == "llm":
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
            "%s segment(s) remain without section_label after classification",
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

    denylist = options.filter_denylist(schema)
    if denylist:
        segments = _filter_segments_excluding(segments, denylist)

    return _size_segments(segments, splitter, config)
