"""Text chunking agent for OntoCast.

This module provides functionality for splitting text into manageable chunks
that can be processed independently, ensuring optimal processing of large
documents.
"""

import logging

from ontocast.onto.chunk import Chunk
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox
from ontocast.util import render_text_hash

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

        if state.max_chunks is not None:
            chunks_txt = chunks_txt[: state.max_chunks]

        for chunk_txt in chunks_txt:
            state.chunks.append(
                Chunk(
                    text=chunk_txt,
                    hid=render_text_hash(chunk_txt),
                    doc_iri=state.doc_iri,
                )
            )
        state.status = Status.SUCCESS
    else:
        state.status = Status.FAILED

    return state
