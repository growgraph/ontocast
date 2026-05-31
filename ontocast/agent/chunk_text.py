"""Text chunking agent for OntoCast.

This module provides functionality for splitting text into manageable chunks
that can be processed independently, ensuring optimal processing of large
documents.
"""

import logging

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.section import assign_section_labels, filter_units_by_target_sections
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


async def chunk_text(state: AgentState, tools: ToolBox) -> AgentState:
    """Split text into manageable chunks.

    This function takes the converted document text and splits it into smaller,
    manageable chunks that can be processed independently.

    Args:
        state: The current agent state containing the text to chunk.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with text chunks.
    """
    logger.info("Chunking the text")
    if state.input_text is not None:
        state.content_units = []
        chunks_txt: list[str] = tools.chunker(state.input_text)
        logger.info(
            f"Created {len(chunks_txt)} chunks for processing: {[len(c) for c in chunks_txt]}"
        )

        for i, chunk_txt in enumerate(chunks_txt):
            state.content_units.append(
                ContentUnit(
                    text=chunk_txt,
                    index=i,
                    doc_iri=state.doc_iri,
                )
            )

        had_spans = bool(state.section_spans)
        if state.section_spans:
            assign_section_labels(
                state.content_units,
                state.input_text,
                state.section_spans,
            )

        if not had_spans and state.target_sections is not None and state.content_units:
            embed_fn = tools.embedding_tool.embed
            for unit in state.content_units:
                if unit.section_label is not None:
                    continue
                label = tools.section_classifier.classify_chunk(
                    unit.text,
                    embed_fn,
                    state.target_sections,
                )
                if label is not None:
                    unit.section_label = label
                    logger.debug(
                        "Content embedding classified unit %s -> %s",
                        unit.index,
                        label,
                    )

        if state.target_sections is not None:
            before = len(state.content_units)
            state.content_units = filter_units_by_target_sections(
                state.content_units,
                state.target_sections,
            )
            logger.info(
                "Section filter %s: kept %s/%s chunks",
                state.target_sections,
                len(state.content_units),
                before,
            )
            if before > 0 and len(state.content_units) == 0:
                logger.warning(
                    "Section filter %s removed all %s chunk(s); "
                    "section_spans_detected=%s. "
                    "Check heading vocabulary or target_sections.",
                    state.target_sections,
                    before,
                    had_spans,
                )
            for index, unit in enumerate(state.content_units):
                unit.index = index

        if (
            state.summarize_sections is not None
            and state.summarize_sections
            and "*" not in state.summarize_sections
            and state.target_sections is None
        ):
            before = len(state.content_units)
            state.content_units = filter_units_by_target_sections(
                state.content_units,
                state.summarize_sections,
            )
            logger.info(
                "summarize_sections implicit filter: kept %s/%s chunks",
                len(state.content_units),
                before,
            )
            for index, unit in enumerate(state.content_units):
                unit.index = index

        if state.max_chunks is not None:
            logger.info(f"Selecting {state.max_chunks} chunks after section filter")
            state.content_units = state.content_units[: state.max_chunks]
            for index, unit in enumerate(state.content_units):
                unit.index = index

        logger.info(
            "Created "
            f"{len(state.content_units)} content units for processing: "
            f"{[len(c.text) for c in state.content_units]}"
        )
        state.status = Status.SUCCESS
    else:
        state.status = Status.FAILED

    return state
