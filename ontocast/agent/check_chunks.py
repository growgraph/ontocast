"""Chunk management agent for OntoCast.

This module provides functionality for managing document chunks in the
OntoCast workflow. It handles chunk processing state transitions and
ensures proper workflow progression through the chunk processing pipeline.

The module supports:
- Checking if chunks are available for processing
- Managing chunk state transitions
- Updating processing status based on chunk availability
- Logging chunk processing progress
"""

import logging
from collections import defaultdict

from ontocast.onto.constants import DEFAULT_CHUNK_IRI
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState

logger = logging.getLogger(__name__)


def check_chunks_empty(state: AgentState) -> AgentState:
    """Check if chunks are available and manage chunk processing state.

    This function checks if there are remaining chunks to process and
    manages the state transitions accordingly. If chunks are available,
    it sets up the next chunk for processing. If no chunks remain,
    it signals completion of the workflow.

    The function performs the following operations:
    1. Adds the current chunk to the processed list if it exists
    2. Checks if there are remaining chunks to process
    3. Sets up the next chunk and resets node visits if chunks remain
    4. Sets appropriate status for workflow routing

    Args:
        state: The current agent state containing chunks and processing status.

    Returns:
        AgentState: Updated agent state with chunk processing information.

    Example:
        >>> state = AgentState(chunks=[chunk1, chunk2], current_chunk=None)
        >>> updated_state = check_chunks_empty(state)
        >>> print(updated_state.current_chunk)  # chunk1
        >>> print(updated_state.status)  # Status.FAILED
    """
    logger.info(
        f"Chunks (rem): {len(state.chunks)}, "
        f"chunks proc: {len(state.chunks_processed)}. "
        f"Setting up current chunk"
    )

    if state.current_chunk is not None:
        state.current_chunk.graph.remap_namespaces(
            old_namespace=DEFAULT_CHUNK_IRI, new_namespace=state.current_chunk.namespace
        )
        state.chunks_processed.append(state.current_chunk)

    if state.chunks:
        state.current_chunk = state.chunks.pop(0)
        state.node_visits = defaultdict(int)
        state.status = Status.FAILED
        logger.info(
            "Chunk available, setting status to FAILED"
            " and proceeding to SELECT_ONTOLOGY"
        )
    else:
        state.current_chunk = None
        state.status = Status.SUCCESS
        logger.info(
            "No more chunks, setting status to SUCCESS "
            "and proceeding to AGGREGATE_FACTS"
        )

    return state
