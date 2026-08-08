"""LLM section-label backfill for chunk preparation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field

from ontocast.config.section_labels import (
    SectionLabelSchema,
    canonical_labels,
    load_section_label_schema,
    normalise_llm_label,
    resolve_section_schema_id,
)
from ontocast.onto.enum import SectionLabelSource
from ontocast.onto.model import BasePydanticModel
from ontocast.prompt.section_classification import (
    CHUNK_SECTION_BATCH_CLASSIFICATION_PROMPT,
    CHUNK_SECTION_CLASSIFICATION_PROMPT,
    document_type_context,
    format_batch_items,
)
from ontocast.tool.llm import record_active_span

if TYPE_CHECKING:
    from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

_FRAGMENT_MAX_CHARS = 500
_BATCH_FRAGMENT_MAX_CHARS = 300


class ChunkSectionClassification(BasePydanticModel):
    """LLM output mapping one excerpt to a canonical section label."""

    label: str | None = Field(
        default=None,
        description="Canonical section label or null if not classifiable",
    )


class SectionLabelAssignment(BasePydanticModel):
    """One excerpt index and the section label assigned to it."""

    index: int = Field(description="Index of the excerpt, as given in the prompt")
    label: str | None = Field(
        default=None,
        description="Canonical section label or null if not classifiable",
    )


class BatchSectionClassification(BasePydanticModel):
    """LLM output assigning a section label to each numbered excerpt."""

    assignments: list[SectionLabelAssignment] = Field(default_factory=list)


def fragment_for_text(text: str) -> str:
    """Return a short excerpt suitable for LLM section classification."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped[:_FRAGMENT_MAX_CHARS]
    snippet = text.strip()
    return snippet[:_FRAGMENT_MAX_CHARS]


async def classify_section_with_llm(
    text: str,
    tools: "ToolBox",
    schema: SectionLabelSchema,
    *,
    document_type_hint: str | None = None,
) -> str | None:
    """Classify a text fragment with the section-label LLM prompt."""
    fragment = fragment_for_text(text)
    if not fragment:
        return None
    parser = PydanticOutputParser(pydantic_object=ChunkSectionClassification)
    allowed = ", ".join(canonical_labels(schema))
    prompt = CHUNK_SECTION_CLASSIFICATION_PROMPT.format_prompt(
        allowed_labels=allowed,
        format_instructions=parser.get_format_instructions(),
        document_context=document_type_context(document_type_hint),
        fragment=fragment,
    )
    response = await tools.llm(prompt)
    parsed = parser.parse(response.content or "")
    return normalise_llm_label(parsed.label, schema)


async def classify_sections_batched(
    items: list[tuple[int, str]],
    tools: "ToolBox",
    schema: SectionLabelSchema,
    *,
    document_type_hint: str | None = None,
    batch_size: int = 40,
) -> dict[int, str | None] | None:
    """Classify many excerpts in as few LLM calls as possible.

    One call covers up to ``batch_size`` excerpts, versus one call per excerpt
    for :func:`classify_section_with_llm`. Passing the excerpts together also
    gives the model the document's shape, which a single fragment cannot show.

    Args:
        items: ``(index, fragment)`` pairs in document order.
        tools: ToolBox providing the LLM.
        schema: Active section label schema.
        document_type_hint: Optional free-text document type.
        batch_size: Maximum excerpts per LLM call.

    Returns:
        Mapping of index to label (``None`` where unclassifiable), or ``None``
        when the model's response could not be used, so the caller can fall
        back to per-excerpt classification.
    """
    if not items:
        return {}
    parser = PydanticOutputParser(pydantic_object=BatchSectionClassification)
    allowed = ", ".join(canonical_labels(schema))
    resolved: dict[int, str | None] = {}
    size = max(1, batch_size)

    for start in range(0, len(items), size):
        batch = items[start : start + size]
        prompt = CHUNK_SECTION_BATCH_CLASSIFICATION_PROMPT.format_prompt(
            allowed_labels=allowed,
            format_instructions=parser.get_format_instructions(),
            document_context=document_type_context(document_type_hint),
            items=format_batch_items(batch),
        )
        try:
            response = await tools.llm(prompt)
            parsed = parser.parse(response.content or "")
        except Exception as exc:
            logger.warning(
                "Batched section classification failed for %s excerpt(s): %s",
                len(batch),
                exc,
            )
            return None
        known = {index for index, _ in batch}
        for assignment in parsed.assignments:
            if assignment.index in known:
                resolved[assignment.index] = normalise_llm_label(
                    assignment.label, schema
                )
    return resolved


async def llm_backfill_section_labels(
    segments: list,
    tools: "ToolBox",
    *,
    section_schema_id: str | None = None,
    document_type_hint: str | None = None,
    section_tag_min_chars: int = 80,
    batch_size: int = 40,
    schema: SectionLabelSchema | None = None,
) -> None:
    """Set ``section_label`` on segments that are still unlabeled (mutates in place).

    Classifies in batches when ``batch_size`` is positive, falling back to one
    call per segment if the batched response cannot be used.

    Args:
        segments: Prepare segments, mutated in place.
        tools: ToolBox providing the LLM.
        section_schema_id: Raw request value; used only when ``schema`` is not
            given.
        document_type_hint: Free-text document type, also passed to the prompt.
        section_tag_min_chars: Minimum segment length to be worth classifying.
        batch_size: Excerpts per LLM call; 0 restores one call per segment.
        schema: Already-resolved schema. Callers that resolved it themselves
            **must** pass it: re-deriving from the raw request would ignore a
            text-based schema detection, and labels outside the re-derived
            schema are silently discarded.
    """
    if schema is None:
        schema = load_section_label_schema(
            resolve_section_schema_id(
                section_schema_id=section_schema_id,
                document_type_hint=document_type_hint,
            )
        )
    min_chars = max(0, section_tag_min_chars)

    def _needs_llm_backfill(index: int) -> bool:
        segment = segments[index]
        if segment.section_label is not None:
            return False
        text = segment.text.strip()
        fragment = fragment_for_text(segment.text)
        if not fragment:
            return False
        if len(text) >= min_chars:
            return True
        if fragment.lstrip().startswith("#"):
            return True
        return bool(segment.headings)

    unlabeled_indices = [
        index for index in range(len(segments)) if _needs_llm_backfill(index)
    ]
    if not unlabeled_indices:
        return

    if batch_size > 0:
        batched = await classify_sections_batched(
            [
                (
                    index,
                    fragment_for_text(segments[index].text)[:_BATCH_FRAGMENT_MAX_CHARS],
                )
                for index in unlabeled_indices
            ],
            tools,
            schema,
            document_type_hint=document_type_hint,
            batch_size=batch_size,
        )
        if batched is not None:
            _apply_llm_labels(segments, batched)
            return
        logger.info(
            "Falling back to per-segment section classification for %s segment(s)",
            len(unlabeled_indices),
        )

    worker_limit = max(1, tools.config.server.parallel_workers)
    semaphore = asyncio.Semaphore(worker_limit)

    async def classify_index(index: int) -> tuple[int, str | None]:
        wait_start = time.perf_counter()
        async with semaphore:
            record_active_span(
                "chunk section classify/worker_wait", time.perf_counter() - wait_start
            )
            segment = segments[index]
            try:
                label = await classify_section_with_llm(
                    segment.text,
                    tools,
                    schema,
                    document_type_hint=document_type_hint,
                )
                return index, label
            except Exception as exc:
                logger.warning(
                    "LLM section classification failed for segment %s: %s",
                    index,
                    exc,
                )
                return index, None

    # classify_index catches its own errors, but the gather must not abort the
    # whole backfill if one slips through -- an unlabeled segment is survivable,
    # an unchunked document is not.
    results = await asyncio.gather(
        *[classify_index(index) for index in unlabeled_indices],
        return_exceptions=True,
    )
    labels = {
        index: label
        for index, label in (
            item for item in results if not isinstance(item, BaseException)
        )
    }
    for item in results:
        if isinstance(item, BaseException):
            logger.warning("Section classification task failed: %s", item)
    _apply_llm_labels(segments, labels)


def _apply_llm_labels(segments: list, labels: dict[int, str | None]) -> None:
    """Write LLM-decided labels onto segments, recording the source."""
    applied = 0
    for index, label in labels.items():
        if label is None:
            continue
        segments[index].section_label = label
        segments[index].section_label_source = SectionLabelSource.LLM
        segments[index].section_label_confidence = 0.7
        applied += 1
    if applied:
        logger.debug("LLM classified %s segment(s)", applied)


__all__ = [
    "BatchSectionClassification",
    "ChunkSectionClassification",
    "SectionLabelAssignment",
    "classify_section_with_llm",
    "classify_sections_batched",
    "fragment_for_text",
    "llm_backfill_section_labels",
]
