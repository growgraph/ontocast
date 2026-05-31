"""Section tagging agent for structured documents."""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field

from ontocast.onto.enum import Status
from ontocast.onto.model import BasePydanticModel
from ontocast.onto.section import (
    CANONICAL_SECTION_LABELS,
    build_section_spans_from_labels,
    iter_heading_lines,
)
from ontocast.onto.state import AgentState
from ontocast.prompt.tag_sections import (
    HEADING_CLASSIFICATION_PROMPT,
    document_type_context,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


class HeadingClassification(BasePydanticModel):
    """Single heading mapped to a canonical section label."""

    heading: str = Field(description="Original heading text")
    label: str | None = Field(
        default=None,
        description="Canonical section label or null if not a section",
    )


class HeadingClassifications(BasePydanticModel):
    """Batch LLM output for section heading classification."""

    classifications: list[HeadingClassification] = Field(default_factory=list)


async def _classify_headings_with_llm(
    headings: list[str],
    tools: ToolBox,
    *,
    document_type_hint: str | None = None,
) -> dict[str, str | None]:
    if not headings:
        return {}
    parser = PydanticOutputParser(pydantic_object=HeadingClassifications)
    allowed = ", ".join(CANONICAL_SECTION_LABELS)
    prompt = HEADING_CLASSIFICATION_PROMPT.format_prompt(
        allowed_labels=allowed,
        format_instructions=parser.get_format_instructions(),
        document_context=document_type_context(document_type_hint),
        headings="\n".join(f"- {heading}" for heading in headings),
    )
    response = await tools.llm(prompt)
    parsed = parser.parse(response.content or "")
    result: dict[str, str | None] = {}
    for item in parsed.classifications:
        result[item.heading.strip()] = tools.section_classifier.normalise_llm_label(
            item.label
        )
    return result


async def tag_sections(state: AgentState, tools: ToolBox) -> AgentState:
    """Detect section headings in converted document text."""
    if not state.use_section_tagging:
        return state

    if not state.input_text:
        state.status = Status.FAILED
        return state

    embed_fn = tools.embedding_tool.embed
    labeled_headings: list[tuple[int, str]] = []
    llm_queue: list[str] = []
    llm_offsets: dict[str, int] = {}

    for offset, normalised, regex_label in iter_heading_lines(state.input_text):
        if regex_label is not None:
            labeled_headings.append((offset, regex_label))
            continue

        label, score, needs_llm = (
            tools.section_classifier.classify_heading_with_confidence(
                normalised, embed_fn
            )
        )
        if label is not None and not needs_llm:
            labeled_headings.append((offset, label))
            logger.debug(
                "Embedding classified heading %r -> %s (score=%.3f)",
                normalised,
                label,
                score,
            )
            continue

        if label is not None:
            logger.debug(
                "Low-confidence embedding for %r -> %s (score=%.3f); queuing LLM",
                normalised,
                label,
                score,
            )
        llm_queue.append(normalised)
        llm_offsets[normalised] = offset

    if llm_queue:
        try:
            llm_labels = await _classify_headings_with_llm(
                llm_queue,
                tools,
                document_type_hint=state.document_type_hint,
            )
            for heading in llm_queue:
                label = llm_labels.get(heading)
                if label is None:
                    label = llm_labels.get(heading.strip())
                if label is not None:
                    labeled_headings.append((llm_offsets[heading], label))
        except Exception as exc:
            logger.warning(
                "LLM section heading classification failed: %s",
                exc,
            )

    spans = build_section_spans_from_labels(state.input_text, labeled_headings)
    state.section_spans = spans
    if spans:
        logger.info(
            "Tagged %s section(s): %s",
            len(spans),
            [span.label for span in spans],
        )
    else:
        logger.info("No section headings detected; section_spans left empty")
    state.status = Status.SUCCESS
    return state
