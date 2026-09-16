"""LLM summarization of content units before extraction."""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ontocast.onto.content_unit import ContentUnit
from ontocast.tool.llm import LLMConfigurationError, use_budget_tracker
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def should_summarize_unit(
    unit: ContentUnit,
    summarize_sections: list[str] | None,
) -> bool:
    """Whether a unit should be passed through the summarization node."""
    if summarize_sections is None:
        return False
    if not summarize_sections or "*" in summarize_sections:
        return True
    if unit.section_label is None:
        return False
    allowed = {section.strip().lower() for section in summarize_sections}
    return unit.section_label.lower() in allowed


async def ensure_unit_summary(
    state: Any,
    unit_index: int,
    tools: ToolBox,
    budget_tracker: Any = None,
) -> None:
    """Summarise one content unit in place, if it is due one and lacks one.

    Called from inside the extraction fan-outs rather than from a preceding
    node. A unit's summary depends only on that unit, so a document-level
    summarize stage made every unit wait for the *slowest* summary before any
    extraction could start, for no dependency.

    Idempotent, so the facts fan-out is a no-op when the ontology fan-out
    already summarised the unit. Failures are logged and leave ``summary`` as
    ``None``: extraction then falls back to the unit's full text.

    Args:
        state: Document state; ``content_units[unit_index]`` is mutated.
        unit_index: Index of the unit to summarise.
        tools: Tool container providing the LLM.
        budget_tracker: Charged for the call.
    """
    unit = state.content_units[unit_index]
    if unit.summary is not None:
        return
    if not state.use_summarization:
        return
    if not should_summarize_unit(unit, state.summarize_sections):
        return
    try:
        unit.summary = await summarize_chunk(
            unit,
            tools,
            max_sentences=state.summary_max_sentences,
            budget_tracker=budget_tracker,
        )
    except LLMConfigurationError:
        # A unit without a summary is survivable; a run whose every call
        # is rejected is not, and it would surface here first.
        raise
    except Exception as exc:
        logger.warning("Summarization failed for unit %s: %s", unit_index, exc)


_SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledge extraction assistant. Compress the user's text to at "
            "most {max_sentences} sentences. Retain all facts, named entities, and "
            "epistemic markers (hedging words, attribution phrases, modal verbs). "
            "Do not interpret or infer — only compress. Output plain text only.",
        ),
        (
            "human",
            "Section: {section_label}\n\n{text}",
        ),
    ]
)


async def summarize_chunk(
    unit: ContentUnit,
    tools: ToolBox,
    *,
    max_sentences: int,
    budget_tracker: Any = None,
) -> str:
    """Compress a content unit for downstream extraction.

    Args:
        unit: The content unit to summarize.
        tools: Tool container.
        max_sentences: Upper bound on summary length.
        budget_tracker: Charged for this call. Summarization used to call the
            shared LLM tool directly, so its tokens landed on whichever
            tracker another unit happened to have bound.
    """
    section_label = unit.section_label or "unclassified"
    prompt = _SUMMARIZE_PROMPT.format_prompt(
        max_sentences=max_sentences,
        section_label=section_label,
        text=unit.text,
    )
    with use_budget_tracker(budget_tracker):
        response = await tools.llm(prompt)
    summary = (response.content or "").strip()
    if not summary:
        raise ValueError("Summarization returned empty text")
    logger.debug(
        "Summarized unit %s (%s): %s -> %s chars",
        unit.index,
        section_label,
        len(unit.text),
        len(summary),
    )
    return summary
