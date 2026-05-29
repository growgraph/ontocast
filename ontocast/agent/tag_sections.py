"""Section tagging agent for structured documents."""

import logging

from ontocast.onto.enum import Status
from ontocast.onto.section import detect_section_spans
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def tag_sections(state: AgentState, tools: ToolBox) -> AgentState:
    """Detect section headings in converted document text."""
    _ = tools
    if not state.use_section_tagging:
        return state

    if not state.input_text:
        state.status = Status.FAILED
        return state

    spans = detect_section_spans(state.input_text)
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
