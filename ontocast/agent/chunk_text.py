"""Text chunking agent for OntoCast.

This module provides functionality for splitting text into manageable chunks
that can be processed independently, ensuring optimal processing of large
documents.
"""

import logging

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def chunk_text(state: AgentState, tools: ToolBox) -> AgentState:
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
        chunks_txt: list[str] = tools.chunker(state.input_text)
        logger.info(
            f"Created {len(chunks_txt)} chunks for processing: {[len(c) for c in chunks_txt]}"
        )

        if state.max_chunks is not None:
            logger.info(f"Selecting {state.max_chunks} chunks")

            chunks_txt = chunks_txt[: state.max_chunks]

        for i, chunk_txt in enumerate(chunks_txt):
            state.content_units.append(
                ContentUnit(
                    text=chunk_txt,
                    index=i,
                    doc_iri=state.doc_iri,
                )
            )

        logger.info(
            "Created "
            f"{len(state.content_units)} content units for processing: "
            f"{[len(c) for c in state.content_units]}"
        )
        state.status = Status.SUCCESS
    else:
        state.status = Status.FAILED

    return state
