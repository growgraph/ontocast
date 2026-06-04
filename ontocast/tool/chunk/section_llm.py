"""LLM section-label backfill for chunk preparation."""

from __future__ import annotations

import asyncio
import logging
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
from ontocast.onto.model import BasePydanticModel
from ontocast.prompt.section_classification import (
    CHUNK_SECTION_CLASSIFICATION_PROMPT,
    document_type_context,
)

if TYPE_CHECKING:
    from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

_FRAGMENT_MAX_CHARS = 500


class ChunkSectionClassification(BasePydanticModel):
    """LLM output mapping one excerpt to a canonical section label."""

    label: str | None = Field(
        default=None,
        description="Canonical section label or null if not classifiable",
    )


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


async def llm_backfill_section_labels(
    segments: list,
    tools: "ToolBox",
    *,
    section_schema_id: str | None,
    document_type_hint: str | None = None,
    section_tag_min_chars: int = 80,
) -> None:
    """Set ``section_label`` on segments that are still unlabeled (mutates in place)."""
    schema_id = resolve_section_schema_id(
        section_schema_id=section_schema_id,
        document_type_hint=document_type_hint,
    )
    schema = load_section_label_schema(schema_id)
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

    worker_limit = max(1, tools.config.server.parallel_workers)
    semaphore = asyncio.Semaphore(worker_limit)

    async def classify_index(index: int) -> tuple[int, str | None]:
        async with semaphore:
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

    results = await asyncio.gather(
        *[classify_index(index) for index in unlabeled_indices]
    )
    for index, label in results:
        if label is not None:
            segments[index].section_label = label


__all__ = [
    "ChunkSectionClassification",
    "classify_section_with_llm",
    "fragment_for_text",
    "llm_backfill_section_labels",
]
