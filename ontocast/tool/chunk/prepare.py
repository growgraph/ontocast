"""Prepare content units: segment, label, select, size, trim and route.

Labelling and routing are separate. Every labelling tier adds evidence to a
segment and one rule decides its label (:mod:`ontocast.tool.chunk.assignment`);
the caller's ``target_sections`` selects among the labels; sizing builds units;
citation runs at a unit's edge are trimmed into units of their own; and
:func:`ontocast.tool.chunk.routing.route_unit` decides each unit's route from
its label, the label's strength and its content. Every unit leaves with a
route and a reason, kept or dropped.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ontocast.config import ChunkConfig
from ontocast.config.section_labels import (
    RouteClass,
    SectionLabelSchema,
    labels_of_class,
    load_section_label_schema,
    resolve_section_schema_id,
    schema_id_from_hint,
)
from ontocast.onto.enum import LabelStrength, SectionLabelSource
from ontocast.onto.section_models import LabelEvidence, SectionSpan
from ontocast.tool.chunk.assignment import assign, content_evidence, merged_evidence
from ontocast.tool.chunk.bibliography import trim_citation_runs
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.outline import markdown_headings
from ontocast.tool.chunk.routing import RouteReason, UnitRoute, route_unit
from ontocast.tool.chunk.schema_detect import SchemaDetection, detect_document_schema
from ontocast.tool.chunk.section_llm import llm_backfill_section_labels
from ontocast.tool.chunk.sections import detect_section_spans
from ontocast.tool.chunk.segment import (
    PrepareSegment,
    coalesce_small_segments_right,
)
from ontocast.tool.chunk.sizing import (
    DEFAULT_PART_SEPARATOR,
    merge_small_parts,
    split_by_measurement_density,
)
from ontocast.tool.chunk.structure import (
    BlockKinds,
    build_document_structure,
    item_refs,
)
from ontocast.util.measurement_lexicon import unit_adjacent_numbers

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

    from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedChunk:
    """A prepared text chunk with optional structural metadata, label and route.

    ``section_label_source`` and ``section_label_strength`` record which tier
    decided the label and whether it may take part in a drop;
    ``section_label_evidence`` holds every tier's vote. ``route`` and
    ``route_reason`` record what routing decided and why.
    """

    text: str
    headings: list[str] | None
    #: Docling items whose blocks the unit holds whole, set after trimming.
    doc_item_refs: tuple[str, ...] = ()
    section_label: str | None = None
    section_label_source: SectionLabelSource | None = None
    section_label_strength: LabelStrength | None = None
    section_label_evidence: tuple[LabelEvidence, ...] = ()
    route: UnitRoute = UnitRoute.EXTRACT
    route_reason: RouteReason = RouteReason.CONTENT


@dataclass(frozen=True)
class PreparedDocument:
    """A document's prepared units, each with its route and reason, in order.

    Attributes:
        routed: Every prepared unit in document order, kept or dropped.
        schema_id: The section schema the document was labelled against;
            ``None`` when section classification is off.
        schema_source: How the schema was chosen: ``explicit``, ``hint``,
            ``detected`` or ``default`` (a fallback, which applies no default
            exclusions); ``None`` when classification is off.
        allowlist_excluded_chars: Characters ``target_sections`` removed.
        allowlist_excluded_unlabeled_chars: The part of those that carried no
            label at all -- text a missing label, not the caller, kept out.
    """

    routed: list[PreparedChunk]
    schema_id: str | None = None
    schema_source: str | None = None
    allowlist_excluded_chars: int = 0
    allowlist_excluded_unlabeled_chars: int = 0

    @property
    def units(self) -> list[PreparedChunk]:
        """Units kept for extraction (routed extract, citations or non-content)."""
        return [unit for unit in self.routed if unit.route is not UnitRoute.SKIP]

    @property
    def dropped(self) -> list[PreparedChunk]:
        """Units routed to skip, each with its reason."""
        return [unit for unit in self.routed if unit.route is UnitRoute.SKIP]


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
        """The extraction allowlist: ``target_sections`` only.

        ``summarize_sections`` chooses which units are summarized and never
        what is extracted.
        """
        return self.target_sections


class SectionSelectionEmptyError(ValueError):
    """A section selection matched no segment in this document.

    Distinct from a malformed parameter: ``target_sections=["reslts"]`` is
    syntactically fine and only turns out to be wrong once *this* document has
    been classified. Under the default ``CHUNK_SECTION_FILTER_ON_EMPTY=warn``
    this is a log line and the run continues to an empty graph, which reads
    exactly like a document that genuinely had nothing to extract -- telling
    those two apart is the whole point of the ``error`` mode.

    Subclasses :class:`ValueError` so existing parameter guards keep their
    shape. Deliberately *not* an ``api``-layer error: nothing under ``tool/``
    imports ``ontocast.api``, and the parameter here is well-formed.
    """

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param


#: Share of a document's text an allowlist may remove before a warning: past
#: it, a selection that rests on missing or wrong labels is likely.
ALLOWLIST_WARN_SHARE = 0.5


def _guard_empty_selection(
    *,
    param: str,
    selection: Sequence[str],
    before: int,
    after: int,
    on_empty: str,
) -> None:
    """Warn or raise when a section selection left nothing behind.

    Args:
        param: Request option (or schema default) that produced ``selection``.
        selection: The effective allowlist or denylist.
        before: Segment count before filtering.
        after: Segment count after filtering.
        on_empty: ``CHUNK_SECTION_FILTER_ON_EMPTY``.

    Raises:
        SectionSelectionEmptyError: ``on_empty`` is ``"error"`` and every
            segment was removed.
    """
    if before == 0 or after > 0:
        return
    message = (
        f"{param}={sorted(selection)} removed all {before} part(s); the "
        "document's detected sections do not match the selection"
    )
    if on_empty == "error":
        raise SectionSelectionEmptyError(param, message)
    logger.warning("%s (CHUNK_SECTION_FILTER_ON_EMPTY=warn)", message)


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
            segment = PrepareSegment(
                text=part,
                headings=headings,
                section_label_source=span.source,
                evidence=(
                    [
                        LabelEvidence(
                            label=span.label, source=span.source, strength=span.strength
                        )
                    ]
                    if span.label is not None
                    else []
                ),
            )
            assign(segment)
            segments.append(segment)
    return segments


_PLACEHOLDER_LINE = re.compile(r"<!--.*-->")


def _is_bare_heading(text: str) -> bool:
    """True when ``text`` is heading lines (and image placeholders) with no body."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    body = [line for line in lines if not _PLACEHOLDER_LINE.fullmatch(line)]
    return bool(body) and len(markdown_headings(text)) == len(body)


def _attach_bare_headings(segments: list[PrepareSegment]) -> list[PrepareSegment]:
    """Prepend each body-less heading segment to the segment that follows it.

    A heading titles the text after it, so the joined segment keeps that text's
    label, headings and evidence: the heading lends it no label, and is
    filtered with its body instead of becoming a unit with nothing to extract.
    Trailing headings have no follower and are kept as they are.
    """
    attached: list[PrepareSegment] = []
    pending: list[PrepareSegment] = []
    for segment in segments:
        if _is_bare_heading(segment.text):
            pending.append(segment)
            continue
        if pending:
            segment = replace(
                segment,
                text=DEFAULT_PART_SEPARATOR.join(
                    part.text.strip() for part in (*pending, segment)
                ),
            )
            pending = []
        attached.append(segment)
    return attached + pending


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


def _add_content_evidence(
    segments: list[PrepareSegment],
    schema: SectionLabelSchema,
    config: ChunkConfig,
    block_kinds: BlockKinds | None,
) -> None:
    """Add the content tier's vote to every segment and decide again.

    Runs on labelled segments too: strong content evidence (a reference list,
    acknowledgements) overrides a weak heading label, and a disagreement with
    a strong heading is kept in the evidence. This is the only tier that can
    label a document with no headings at all.
    """
    if config.section_density == "off":
        return
    aggressive = config.section_density == "aggressive"
    voted = 0
    for segment in segments:
        evidence = content_evidence(
            segment.text, schema, aggressive=aggressive, block_kinds=block_kinds
        )
        if evidence is None:
            continue
        segment.evidence.append(evidence)
        assign(segment)
        voted += 1
    if voted:
        logger.info(
            "Content-density classification (%s) voted on %s segment(s)",
            config.section_density,
            voted,
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
                section_label=segment.section_label,
                section_label_source=segment.section_label_source,
                section_label_strength=segment.section_label_strength,
                section_label_evidence=tuple(segment.evidence),
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
                    section_label=segment.section_label,
                    section_label_source=segment.section_label_source,
                    section_label_strength=segment.section_label_strength,
                    section_label_evidence=tuple(segment.evidence),
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
        # A merged chunk is only as trustworthy as its weakest constituent.
        weakest = min(
            run,
            key=lambda chunk: chunk.section_label_strength is LabelStrength.STRONG,
        )
        evidence = merged_evidence(*(chunk.section_label_evidence for chunk in run))

        for text in texts:
            merged.append(
                PreparedChunk(
                    text=text,
                    headings=headings,
                    section_label=label,
                    section_label_source=weakest.section_label_source,
                    section_label_strength=weakest.section_label_strength,
                    section_label_evidence=evidence,
                )
            )
    return merged


def _split_dense_chunks(
    chunks: list[PreparedChunk],
    config: ChunkConfig,
) -> list[PreparedChunk]:
    """Split sized chunks that pack more measurements than the configured cap.

    Runs after merging, on final units, so it is the last word on unit
    boundaries: a piece inherits its parent's headings, refs and label. A
    split changes what each render is shown, so each one is logged.
    """
    cap = config.max_measurements_per_unit
    if cap <= 0:
        return chunks
    split: list[PreparedChunk] = []
    for chunk in chunks:
        pieces = split_by_measurement_density(
            chunk.text, max_measurements=cap, min_size=config.min_size
        )
        if len(pieces) > 1:
            logger.info(
                "Split chunk (%d chars, %d measurements, section_label=%r) into "
                "%d pieces of %s chars (CHUNK_MAX_MEASUREMENTS_PER_UNIT=%d)",
                len(chunk.text),
                len(unit_adjacent_numbers(chunk.text)),
                chunk.section_label,
                len(pieces),
                [len(piece) for piece in pieces],
                cap,
            )
        split.extend(replace(chunk, text=piece) for piece in pieces)
    return split


def _size_segments(
    segments: list[PrepareSegment],
    splitter: ChunkerTool,
    config: ChunkConfig,
) -> list[PreparedChunk]:
    expanded: list[PreparedChunk] = []
    for segment in segments:
        expanded.extend(_expand_segment(segment, splitter, config))
    merged = _merge_prepared_chunks(expanded, config.min_size, config.max_size)
    return _split_dense_chunks(merged, config)


def _simple_prepare(
    document_text: str,
    splitter: ChunkerTool,
    config: ChunkConfig,
) -> list[PreparedChunk]:
    """Untagged preparation (section classifier off)."""
    segments = _semantic_full_doc_segments(document_text, splitter, config.max_size)
    if not segments:
        text = document_text.strip()
        if not text:
            return []
        segments = [PrepareSegment(text=part) for part in splitter.size_text(text)]
    return _size_segments(segments, splitter, config)


def _allowlist_report(
    before: list[PrepareSegment],
    after: list[PrepareSegment],
    allowlist: list[str],
) -> tuple[int, int]:
    """Characters an allowlist removed, and how many of them carried no label.

    Warns when the allowlist removed most of the document: a strict selection
    resting on missing or wrong labels loses text without any other trace.
    """
    kept_ids = {id(segment) for segment in after}
    removed = [segment for segment in before if id(segment) not in kept_ids]
    removed_chars = sum(len(segment.text) for segment in removed)
    unlabeled_chars = sum(
        len(segment.text) for segment in removed if segment.section_label is None
    )
    total_chars = sum(len(segment.text) for segment in before)
    logger.info(
        "target_sections=%s kept %s/%s segment(s); removed %s chars, %s of them "
        "unlabeled",
        sorted(allowlist),
        len(after),
        len(before),
        removed_chars,
        unlabeled_chars,
    )
    if after and total_chars and removed_chars / total_chars > ALLOWLIST_WARN_SHARE:
        logger.warning(
            "target_sections=%s removed %.0f%% of the document's text (%s of "
            "%s chars unlabeled); check the labels with `ontocast sections`",
            sorted(allowlist),
            100 * removed_chars / total_chars,
            unlabeled_chars,
            removed_chars,
        )
    return removed_chars, unlabeled_chars


def _trim_citation_runs(
    chunks: list[PreparedChunk],
    schema: SectionLabelSchema | None,
    block_kinds: BlockKinds | None = None,
) -> list[PreparedChunk]:
    """Cut citation runs at a unit's start or end into units of their own.

    The cut-out run is labelled with the schema's citations-class label, by the
    content-block tier, as strong evidence of its own; the rest of the unit
    keeps its label. A unit that is all citations is left whole: routing reads
    its content directly.
    """
    citation_labels = labels_of_class(schema, RouteClass.CITATIONS) if schema else []
    citation_label = citation_labels[0] if citation_labels else None
    run_evidence = (
        (
            LabelEvidence(
                label=citation_label,
                source=SectionLabelSource.CONTENT_BLOCK,
                strength=LabelStrength.STRONG,
            ),
        )
        if citation_label is not None
        else ()
    )
    trimmed: list[PreparedChunk] = []
    for chunk in chunks:
        pieces = trim_citation_runs(chunk.text, block_kinds)
        if len(pieces) == 1:
            trimmed.append(chunk)
            continue
        for piece, is_run in pieces:
            if not is_run:
                trimmed.append(replace(chunk, text=piece))
                continue
            trimmed.append(
                replace(
                    chunk,
                    text=piece,
                    section_label=citation_label,
                    section_label_source=SectionLabelSource.CONTENT_BLOCK,
                    section_label_strength=run_evidence[0].strength
                    if run_evidence
                    else None,
                    section_label_evidence=run_evidence,
                )
            )
        logger.info(
            "Trimmed a citation run out of a unit (%d chars, section_label=%r): %s",
            len(chunk.text),
            chunk.section_label,
            [
                (len(piece), "citations" if is_run else "kept")
                for piece, is_run in pieces
            ],
        )
    return trimmed


def _route_chunks(
    chunks: list[PreparedChunk],
    *,
    schema: SectionLabelSchema | None,
    schema_fallback: bool,
    exclude_sections: list[str] | None,
    config: ChunkConfig,
    block_kinds: BlockKinds | None = None,
) -> list[PreparedChunk]:
    """Route every chunk, in order, recording its route, reason and item refs.

    Every decision other than a plain extract is logged with the unit's size,
    label and labelling tier: a wrong route changes what is extracted, and
    this line is where it shows without a dry run.
    """
    routed: list[PreparedChunk] = []
    for index, chunk in enumerate(chunks):
        decision = route_unit(
            chunk,
            schema=schema,
            schema_fallback=schema_fallback,
            exclude_sections=exclude_sections,
            config=config,
            block_kinds=block_kinds,
        )
        if decision.reason is not RouteReason.CONTENT:
            logger.info(
                "Unit %d (%d chars, section_label=%r, source=%s) routed %s: %s",
                index,
                len(chunk.text),
                chunk.section_label,
                chunk.section_label_source.value
                if chunk.section_label_source is not None
                else None,
                decision.route.value,
                decision.reason.value,
            )
        routed.append(
            replace(
                chunk,
                route=decision.route,
                route_reason=decision.reason,
                doc_item_refs=item_refs(chunk.text, block_kinds),
            )
        )
    return routed


_EXCLUSION_REASONS = frozenset(
    {RouteReason.EXCLUDED_BY_REQUEST, RouteReason.BOILERPLATE}
)


def _guard_excluded_everything(
    document: PreparedDocument,
    options: PrepareOptions,
    schema: SectionLabelSchema,
    on_empty: str,
) -> None:
    """Warn or raise when exclusions, not content, left nothing to extract."""
    dropped = document.dropped
    if document.units or not any(
        unit.route_reason in _EXCLUSION_REASONS for unit in dropped
    ):
        return
    _guard_empty_selection(
        param=(
            "exclude_sections"
            if options.exclude_sections is not None
            else f"exclude_sections (schema '{schema.id}' boilerplate)"
        ),
        selection=options.exclude_sections
        or labels_of_class(schema, RouteClass.BOILERPLATE),
        before=len(dropped),
        after=0,
        on_empty=on_empty,
    )


async def prepare_content_units(
    docling_doc: DoclingDocument,
    splitter: ChunkerTool,
    config: ChunkConfig,
    options: PrepareOptions,
    tools: "ToolBox | None" = None,
) -> PreparedDocument:
    """Segment, tag, select, size, trim and route document text into units.

    Section tagging is default-on: the sections-first flow runs unless
    ``CHUNK_SECTION_CLASSIFIER=off`` (which leaves only the content rules of
    routing; explicit section options are ignored with a warning in that
    case).

    Args:
        docling_doc: Converted source document.
        splitter: Chunker used to size oversized sections.
        config: Chunk configuration, including the classifier tier.
        options: Per-request section schema and filters.
        tools: ToolBox providing the LLM. Required only when
            ``config.section_classifier == "llm"``; the deterministic tiers
            need no LLM, so callers that only inspect sections may omit it.

    Returns:
        The kept and dropped units, each with its route and reason, and how the
        schema was chosen.

    Raises:
        ValueError: ``section_classifier`` is ``"llm"`` but no ToolBox was
            supplied.
        SectionSelectionEmptyError: ``target_sections`` or the exclusions
            removed every unit and ``CHUNK_SECTION_FILTER_ON_EMPTY=error``.
    """
    if config.section_classifier == "llm" and tools is None:
        raise ValueError(
            "CHUNK_SECTION_CLASSIFIER=llm requires a ToolBox providing an LLM"
        )
    # Segmentation is CPU-bound and runs local embedding models, but this
    # coroutine is awaited on the event loop -- inline, it would freeze every
    # concurrent document's in-flight provider sockets for its whole duration.
    structure = await asyncio.to_thread(build_document_structure, docling_doc)
    document_text = structure.text
    block_kinds = structure.block_kinds

    if config.section_classifier == "off":
        if options.needs_section_prepare():
            logger.warning(
                "Section options requested but CHUNK_SECTION_CLASSIFIER=off; "
                "section filters are ignored"
            )
        chunks = await asyncio.to_thread(
            _simple_prepare, document_text, splitter, config
        )
        routed = _route_chunks(
            _trim_citation_runs(chunks, None, block_kinds),
            schema=None,
            schema_fallback=True,
            exclude_sections=None,
            config=config,
            block_kinds=block_kinds,
        )
        return PreparedDocument(routed=routed)

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
        _semantic_section_segments, document_text, spans, splitter, config.max_size
    )
    if not segments:
        return PreparedDocument(
            routed=[], schema_id=schema.id, schema_source=decision.source
        )

    segments = await asyncio.to_thread(
        coalesce_small_segments_right,
        segments,
        config.section_tag_min_chars,
        schema,
    )
    if config.section_classifier in ("heuristic", "llm"):
        await asyncio.to_thread(
            _add_content_evidence, segments, schema, config, block_kinds
        )
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
    segments = _attach_bare_headings(segments)

    unlabeled = sum(1 for s in segments if s.section_label is None)
    if unlabeled:
        # Normal for text with no headings (plain-text and JSON inputs); the
        # manifest's section_label_histogram is the record.
        logger.info(
            "%s segment(s) remain without section_label after classification",
            unlabeled,
        )

    on_empty = config.section_filter_on_empty

    excluded_chars = 0
    excluded_unlabeled_chars = 0
    allowlist = options.filter_allowlist()
    if allowlist is not None:
        before = len(segments)
        kept = _filter_segments(segments, allowlist)
        excluded_chars, excluded_unlabeled_chars = _allowlist_report(
            segments, kept, allowlist
        )
        segments = kept
        _guard_empty_selection(
            param="target_sections",
            selection=allowlist,
            before=before,
            after=len(segments),
            on_empty=on_empty,
        )

    chunks = await asyncio.to_thread(_size_segments, segments, splitter, config)
    chunks = _trim_citation_runs(chunks, schema, block_kinds)
    routed = _route_chunks(
        chunks,
        schema=schema,
        schema_fallback=decision.source == "default",
        exclude_sections=options.exclude_sections,
        config=config,
        block_kinds=block_kinds,
    )
    document = PreparedDocument(
        routed=routed,
        schema_id=schema.id,
        schema_source=decision.source,
        allowlist_excluded_chars=excluded_chars,
        allowlist_excluded_unlabeled_chars=excluded_unlabeled_chars,
    )
    _guard_excluded_everything(document, options, schema, on_empty)
    return document
