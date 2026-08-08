"""Prepare content units: segment, tag, filter, and size within section boundaries."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from ontocast.config import ChunkConfig
from ontocast.config.section_labels import (
    SectionLabelSchema,
    label_order,
    load_section_label_schema,
    resolve_section_schema_id,
    schema_id_from_hint,
)
from ontocast.onto.enum import SectionLabelSource
from ontocast.onto.section_models import SectionSpan
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.density import classify_by_density
from ontocast.tool.chunk.outline import markdown_headings
from ontocast.tool.chunk.schema_detect import SchemaDetection, detect_document_schema
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

# Relative trust in each classification tier, used when merging chunks: the
# merged chunk inherits the weakest evidence in the run.
_SOURCE_STRENGTH: dict[SectionLabelSource, int] = {
    SectionLabelSource.OUTLINE_UNRESOLVED: 0,
    SectionLabelSource.FORWARD_FILL: 1,
    SectionLabelSource.FRONT_MATTER: 2,
    SectionLabelSource.CONTENT_DENSITY: 3,
    SectionLabelSource.LLM: 4,
    SectionLabelSource.HEADING_INHERITED: 5,
    SectionLabelSource.SPAN_OVERLAP: 6,
    SectionLabelSource.HEADING_KEYWORD: 7,
    SectionLabelSource.HEADING_PATTERN: 8,
}


@dataclass(frozen=True)
class PreparedChunk:
    """A prepared text chunk with optional structural metadata and section label.

    ``section_label_source`` and ``section_label_confidence`` record which tier
    of the classification cascade decided the label, so a run can be audited
    and weak labels can be told from strong ones.
    """

    text: str
    headings: list[str] | None
    doc_item_refs: tuple[str, ...] = ()
    section_label: str | None = None
    section_label_source: SectionLabelSource | None = None
    section_label_confidence: float = 0.0


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
) -> list[tuple[SectionSpan, str]]:
    """Cover the document with ordered ``(span, text)`` blocks.

    Spans from the outline already partition the document, but any gap left by
    a caller-supplied span list is filled with an explicitly unresolved span so
    that no text is dropped and no neighbouring label is stretched over it.
    """
    blocks: list[tuple[SectionSpan, str]] = []
    if not spans:
        return [
            (
                SectionSpan(label=None, start=0, end=len(document_text)),
                document_text,
            )
        ]
    cursor = 0
    for span in sorted(spans, key=lambda item: item.start):
        if span.start > cursor:
            gap = SectionSpan(label=None, start=cursor, end=span.start)
            blocks.append((gap, document_text[cursor : span.start]))
        blocks.append((span, document_text[span.start : span.end]))
        cursor = max(cursor, span.end)
    if cursor < len(document_text):
        tail = SectionSpan(label=None, start=cursor, end=len(document_text))
        blocks.append((tail, document_text[cursor:]))
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
    for span, block_text in _section_blocks(document_text, spans):
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
                    section_label=span.label,
                    section_label_source=span.source,
                    section_label_confidence=span.confidence,
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
        if segment.section_label_source is SectionLabelSource.OUTLINE_UNRESOLVED:
            # The outline knows this region belongs to an unrecognised section.
            # Guessing from a neighbouring span would recreate label smearing;
            # leave it for the content and LLM tiers.
            _, search_from = label_text_from_spans(
                segment.text, document_text, spans, search_from
            )
            continue
        heading_label = label_from_headings(segment.headings, schema)
        if heading_label is not None:
            segment.section_label = heading_label
            segment.section_label_source = SectionLabelSource.HEADING_PATTERN
            segment.section_label_confidence = 0.95
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
            if label is not None:
                segment.section_label_source = SectionLabelSource.SPAN_OVERLAP
                segment.section_label_confidence = 0.8


def _forward_fill_section_labels(
    segments: list[PrepareSegment],
    schema: "SectionLabelSchema",
) -> None:
    """Propagate the nearest preceding label to unlabeled segments.

    Propagation is blocked in three cases:

    - the segment's section is *explicitly unresolved* — the outline saw a
      heading it could not name, so the preceding label is known to be wrong.
      This is the guard that keeps the span fix from being undone here;
    - the segment opens with a recognised section heading of its own;
    - the fill would run backwards against the schema's canonical section order
      (filling ``results`` into a region that precedes the ``introduction``).
    """
    last_label: str | None = None
    fill_count = 0
    for index, segment in enumerate(segments):
        if segment.section_label is not None:
            last_label = segment.section_label
            continue
        if segment.section_label_source is SectionLabelSource.OUTLINE_UNRESOLVED:
            last_label = None
            continue
        if last_label is None or starts_with_section_heading(segment, schema):
            continue
        if _fill_runs_backwards(last_label, segments, index, schema):
            continue
        segment.section_label = last_label
        segment.section_label_source = SectionLabelSource.FORWARD_FILL
        segment.section_label_confidence = 0.3
        fill_count += 1
    if fill_count:
        logger.debug(
            "Forward-filled %s segment(s) with nearest preceding label", fill_count
        )


@dataclass(frozen=True)
class SchemaDecision:
    """Which label schema a document is prepared against, and why.

    Recorded rather than recomputed: schema selection now depends on document
    text, and it used to be resolved independently in three places. If those
    disagreed, the deterministic tiers would tag against one schema while the
    LLM backfill validated against another, and ``normalise_llm_label`` drops
    labels absent from its schema -- silent label loss, not an error.
    """

    schema: SectionLabelSchema
    source: str
    detection: SchemaDetection | None = None

    @property
    def schema_id(self) -> str:
        return self.schema.id


def _sample_paragraphs(document_text: str, limit: int = 24) -> list[str]:
    """Evenly spaced body paragraphs, for the content detection tier."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", document_text)]
    blocks = [
        " ".join(block.split())
        for block in blocks
        if len(block) >= 300 and not block.lstrip().startswith(("#", "|"))
    ]
    if not blocks:
        return []
    step = max(1, len(blocks) // limit)
    return [block[:600] for block in blocks[::step]][:limit]


def resolve_prepare_schema(
    document_text: str,
    config: ChunkConfig,
    options: PrepareOptions,
    splitter: ChunkerTool | None = None,
) -> SchemaDecision:
    """Choose the section-label schema for one document.

    Precedence: an explicit ``section_schema_id``, then a ``document_type_hint``
    that maps to a schema, then automatic detection, then the manifest default.
    Caller-supplied intent is never overridden -- detection only fills the gap
    where the request said nothing.

    Args:
        document_text: Markdown export used for heading detection.
        config: Chunk configuration, including the detection tier.
        options: Per-request schema id and document type hint.
        splitter: Chunker, used only to reach the embedding model already loaded
            for semantic chunking. ``None`` restricts detection to the lexical
            tier.

    Returns:
        The chosen schema and how it was chosen.
    """
    if options.section_schema_id and options.section_schema_id.strip():
        schema_id = resolve_section_schema_id(
            section_schema_id=options.section_schema_id
        )
        return SchemaDecision(load_section_label_schema(schema_id), "explicit")

    from_hint = schema_id_from_hint(options.document_type_hint)
    if from_hint is not None:
        return SchemaDecision(load_section_label_schema(from_hint), "hint")

    detection = _detect_schema(document_text, config, splitter)
    if detection is not None:
        logger.info(
            "Detected document schema %r via %s tier (score %.1f, margin %.1fx): %s",
            detection.schema_id,
            detection.tier,
            detection.score,
            detection.margin,
            ", ".join(detection.evidence[0].examples[:3]),
        )
        return SchemaDecision(
            load_section_label_schema(detection.schema_id), "detected", detection
        )

    default_id = resolve_section_schema_id()
    logger.info("No schema evidence; falling back to default schema %r", default_id)
    return SchemaDecision(load_section_label_schema(default_id), "default")


def _detect_schema(
    document_text: str,
    config: ChunkConfig,
    splitter: ChunkerTool | None,
) -> SchemaDetection | None:
    if config.section_schema_detect == "off":
        return None
    headings = [node.text for node in markdown_headings(document_text)]
    embed = None
    if config.section_schema_detect in ("headings", "auto") and splitter is not None:
        embed = splitter.embed_texts
    allow_content = config.section_schema_detect == "auto"
    return detect_document_schema(
        headings,
        _sample_paragraphs(document_text) if allow_content else None,
        embed=embed,
        allow_content_tier=allow_content,
        min_score=config.section_schema_detect_min_score,
        min_margin=config.section_schema_detect_min_margin,
        content_min_margin=config.section_schema_detect_content_min_margin,
    )


def _density_label_segments(
    segments: list[PrepareSegment],
    schema: "SectionLabelSchema",
    config: ChunkConfig,
) -> None:
    """Label still-unlabeled segments from their content (mutates in place).

    This is the only tier that can classify a document with no headings at all.
    It runs after heading analysis so it never overrides a heading-derived
    label, and it declines to guess when the evidence is ambiguous.
    """
    if config.section_density == "off":
        return
    aggressive = config.section_density == "aggressive"
    labeled = 0
    for segment in segments:
        if segment.section_label is not None:
            continue
        result = classify_by_density(segment.text, schema, aggressive=aggressive)
        if result is None:
            continue
        segment.section_label, segment.section_label_confidence = result
        segment.section_label_source = SectionLabelSource.CONTENT_DENSITY
        labeled += 1
    if labeled:
        logger.info(
            "Content-density classification (%s) labeled %s segment(s)",
            config.section_density,
            labeled,
        )


def _fill_runs_backwards(
    label: str,
    segments: list[PrepareSegment],
    index: int,
    schema: "SectionLabelSchema",
) -> bool:
    """Whether filling ``label`` at ``index`` would violate canonical order."""
    order = label_order(label, schema)
    if order is None:
        return False
    for following in segments[index + 1 :]:
        if following.section_label is None:
            continue
        next_order = label_order(following.section_label, schema)
        if next_order is None:
            return False
        if next_order < order:
            logger.debug(
                "Refusing forward-fill of %r before %r (order %s > %s)",
                label,
                following.section_label,
                order,
                next_order,
            )
            return True
        return False
    return False


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
                section_label_source=segment.section_label_source,
                section_label_confidence=segment.section_label_confidence,
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
                    section_label_source=segment.section_label_source,
                    section_label_confidence=segment.section_label_confidence,
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
            # Unlabeled chunks are not known to share a section — they are
            # merely each unresolved. Merging a run of them would rebuild the
            # cross-section chunks the outline fix just eliminated.
            if label is None:
                break

        texts = merge_small_parts(
            [chunk.text for chunk in run],
            min_size,
            max_size,
        )
        headings = next((chunk.headings for chunk in run if chunk.headings), None)
        refs: tuple[str, ...] = ()
        for chunk in run:
            refs = merge_doc_item_refs(refs, chunk.doc_item_refs)
        # A merged chunk is only as trustworthy as its weakest constituent.
        source = min(
            (chunk.section_label_source for chunk in run if chunk.section_label_source),
            key=_SOURCE_STRENGTH.__getitem__,
            default=None,
        )
        confidence = min(chunk.section_label_confidence for chunk in run)

        for text in texts:
            merged.append(
                PreparedChunk(
                    text=text,
                    headings=headings,
                    doc_item_refs=refs,
                    section_label=label,
                    section_label_source=source,
                    section_label_confidence=confidence,
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
    return _hybrid_chunker_for_max_tokens(max(512, config.max_size // 4))


@lru_cache(maxsize=4)
def _hybrid_chunker_for_max_tokens(max_tokens: int) -> HybridChunker:
    """Build (and cache) a HybridChunker for one token budget.

    The chunker holds a HuggingFace tokenizer and carries no per-document
    state, so rebuilding it per document meant an ``AutoTokenizer`` load on
    every request for an object that only ever depends on ``max_tokens``.
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
    tools: "ToolBox | None" = None,
) -> list[PreparedChunk]:
    """Segment, tag, filter, and size document text into prepared chunks.

    Section tagging is default-on: the sections-first flow runs unless
    ``CHUNK_SECTION_CLASSIFIER=off`` (which also disables section filters and
    schema default exclusions; explicit section options are ignored with a
    warning in that case).

    Args:
        docling_doc: Converted source document.
        splitter: Chunker used to size oversized sections.
        config: Chunk configuration, including the classifier tier.
        options: Per-request section schema and filters.
        tools: ToolBox providing the LLM. Required only when
            ``config.section_classifier == "llm"``; the deterministic tiers
            need no LLM, so callers that only inspect sections may omit it.

    Raises:
        ValueError: ``section_classifier`` is ``"llm"`` but no ToolBox was
            supplied.
    """
    if config.section_classifier == "llm" and tools is None:
        raise ValueError(
            "CHUNK_SECTION_CLASSIFIER=llm requires a ToolBox providing an LLM"
        )
    # Segmentation is CPU-bound and runs local embedding models, but this
    # coroutine is awaited on the event loop -- inline, it would freeze every
    # concurrent document's in-flight provider sockets for its whole duration.
    document_text = await asyncio.to_thread(
        document_text_for_section_tagging, docling_doc
    )

    if config.section_classifier == "off":
        if options.needs_section_prepare():
            logger.warning(
                "Section options requested but CHUNK_SECTION_CLASSIFIER=off; "
                "section filters are ignored"
            )
        return await asyncio.to_thread(
            _simple_prepare, docling_doc, document_text, splitter, config
        )

    decision = await asyncio.to_thread(
        resolve_prepare_schema, document_text, config, options, splitter
    )
    schema = decision.schema
    spans = await asyncio.to_thread(
        detect_section_spans,
        document_text,
        schema,
        include_text_headings=config.section_text_headings,
    )

    segments = await asyncio.to_thread(
        _primary_segments, docling_doc, document_text, spans, splitter, config
    )
    if not segments:
        return []

    segments = await asyncio.to_thread(
        coalesce_small_segments_right,
        segments,
        config.section_tag_min_chars,
        schema,
    )
    await asyncio.to_thread(_tag_segments, segments, document_text, spans, schema)
    if config.section_classifier in ("heuristic", "llm"):
        await asyncio.to_thread(_density_label_segments, segments, schema, config)
    if config.section_classifier == "llm" and tools is not None:
        await llm_backfill_section_labels(
            segments,
            tools,
            # The resolved schema is threaded, not re-derived: re-resolving from
            # the raw request would ignore a text-based detection and validate
            # LLM labels against a different schema, silently dropping them.
            schema=schema,
            section_schema_id=options.section_schema_id,
            document_type_hint=options.document_type_hint,
            section_tag_min_chars=config.section_tag_min_chars,
            batch_size=config.section_llm_batch_size,
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

    return await asyncio.to_thread(_size_segments, segments, splitter, config)
