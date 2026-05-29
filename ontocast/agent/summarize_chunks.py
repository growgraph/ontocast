"""LLM summarization of content units before extraction."""

import logging

from langchain_core.prompts import ChatPromptTemplate

from ontocast.onto.content_unit import ContentUnit
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

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
) -> str:
    """Compress a content unit for downstream extraction."""
    section_label = unit.section_label or "unclassified"
    prompt = _SUMMARIZE_PROMPT.format_prompt(
        max_sentences=max_sentences,
        section_label=section_label,
        text=unit.text,
    )
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
