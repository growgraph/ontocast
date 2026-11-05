"""LLM section tier: one call labels the document's outline, as weak votes.

The model is shown the document as an ordered list of regions: each headed
section once (its heading line and the opening of its text), and each segment
of unheaded text on its own. It returns one label per region, chosen from the
schema's labels and their descriptions. Every segment of a region receives
that label as weak evidence, so the model decides only where no other tier
spoke; where it disagrees with another tier, the disagreement is kept in the
evidence. A long outline is sent in consecutive windows. A failed call costs
the votes of its window and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field

from ontocast.config.section_labels import SectionLabelSchema, normalise_llm_label
from ontocast.onto.enum import LabelStrength, LLMPhase, SectionLabelSource
from ontocast.onto.model import BasePydanticModel
from ontocast.onto.section_models import LabelEvidence, OutlineLabelReport
from ontocast.prompt.section_classification import (
    OUTLINE_LABELING_PROMPT,
    document_type_context,
    format_label_catalog,
    format_regions,
)
from ontocast.tool.chunk.assignment import assign
from ontocast.tool.llm import RUN_STOPPING_ERRORS, record_active_count, use_llm_phase

if TYPE_CHECKING:
    from ontocast.tool.chunk.segment import PrepareSegment
    from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: Characters of a region shown to the model: the heading and its opening text.
EXCERPT_MAX_CHARS = 300


class RegionLabel(BasePydanticModel):
    """The section label the model gives one numbered region."""

    index: int = Field(description="Number of the region, as given in the prompt")
    label: str | None = Field(
        default=None,
        description="One of the listed section labels, or null if none fits",
    )


class OutlineLabels(BasePydanticModel):
    """The model's labels for the numbered regions of one outline window."""

    labels: list[RegionLabel] = Field(default_factory=list)


@dataclass(frozen=True)
class OutlineRegion:
    """One region shown to the model and the segments its label applies to."""

    excerpt: str
    segments: tuple[int, ...]


def text_excerpt(text: str, limit: int = EXCERPT_MAX_CHARS) -> str:
    """The opening of ``text``: its first line kept, the rest on one line.

    A heading keeps its own line, so the model sees what names the region and
    what follows it as distinct.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    first, _, rest = stripped.partition("\n")
    body = " ".join(rest.split())
    excerpt = f"{first.strip()}\n{body}" if body else first.strip()
    return excerpt[:limit].rstrip()


def outline_regions(segments: list[PrepareSegment]) -> list[OutlineRegion]:
    """The document's regions in order: each headed section once, unheaded segments each."""
    regions: list[OutlineRegion] = []
    position: dict[int, int] = {}
    for index, segment in enumerate(segments):
        if segment.region is None:
            excerpt = text_excerpt(segment.text)
            if excerpt:
                regions.append(OutlineRegion(excerpt=excerpt, segments=(index,)))
            continue
        at = position.get(segment.region.index)
        if at is None:
            position[segment.region.index] = len(regions)
            regions.append(
                OutlineRegion(excerpt=segment.region.excerpt, segments=(index,))
            )
            continue
        region = regions[at]
        regions[at] = OutlineRegion(
            excerpt=region.excerpt, segments=(*region.segments, index)
        )
    return regions


async def _label_window(
    window: list[OutlineRegion],
    tools: ToolBox,
    schema: SectionLabelSchema,
    document_type_hint: str | None,
) -> dict[int, str] | Exception:
    """The model's labels for one window, by position, or the error it raised."""
    parser = PydanticOutputParser(pydantic_object=OutlineLabels)
    prompt = OUTLINE_LABELING_PROMPT.format_prompt(
        label_catalog=format_label_catalog(
            [(label.id, label.description) for label in schema.labels]
        ),
        format_instructions=parser.get_format_instructions(),
        document_context=document_type_context(document_type_hint),
        regions=format_regions([region.excerpt for region in window]),
    )
    try:
        with use_llm_phase(LLMPhase.SECTION_CLASSIFICATION):
            response = await tools.llm(prompt)
        parsed = parser.parse(response.content or "")
    except RUN_STOPPING_ERRORS:
        raise
    except Exception as exc:
        # The tier is weak: its votes are missed, the document is not.
        logger.warning(
            "Outline labeling failed for %s region(s): %s: %s",
            len(window),
            type(exc).__name__,
            exc,
        )
        record_active_count("chunk/outline_windows_failed")
        return exc
    labels: dict[int, str] = {}
    for item in parsed.labels:
        if not 0 <= item.index < len(window):
            continue
        label = normalise_llm_label(item.label, schema)
        if label is not None:
            labels[item.index] = label
    return labels


async def llm_label_outline(
    segments: list[PrepareSegment],
    tools: ToolBox,
    schema: SectionLabelSchema,
    *,
    document_type_hint: str | None = None,
    window: int = 40,
) -> OutlineLabelReport:
    """Add the model's vote to every segment and decide again (mutates in place).

    Args:
        segments: Prepare segments, in document order.
        tools: ToolBox providing the LLM.
        schema: The schema the document is typed against; labels outside it
            are discarded.
        document_type_hint: Free-text document type, passed to the prompt.
        window: Regions per call.

    Returns:
        How many windows were sent and which failed.
    """
    regions = outline_regions(segments)
    size = max(1, window)
    voted = 0
    report = OutlineLabelReport()
    for start in range(0, len(regions), size):
        chunk = regions[start : start + size]
        report.windows += 1
        labels = await _label_window(chunk, tools, schema, document_type_hint)
        if isinstance(labels, Exception):
            report.windows_failed += 1
            name = type(labels).__name__
            report.errors[name] = report.errors.get(name, 0) + 1
            continue
        for position, label in labels.items():
            for index in chunk[position].segments:
                segment = segments[index]
                segment.evidence.append(
                    LabelEvidence(
                        label=label,
                        source=SectionLabelSource.LLM,
                        strength=LabelStrength.WEAK,
                    )
                )
                assign(segment)
                voted += 1
    if voted:
        logger.debug(
            "Outline labeling voted on %s segment(s) in %s region(s)",
            voted,
            len(regions),
        )
    return report


__all__ = [
    "EXCERPT_MAX_CHARS",
    "OutlineLabels",
    "OutlineRegion",
    "RegionLabel",
    "llm_label_outline",
    "outline_regions",
    "text_excerpt",
]
