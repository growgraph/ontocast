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

from rdflib import URIRef

from ontocast.onto.constants import CHUNK_NULL_IRI, DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
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
        >>> print(updated_state.current_content_unit)  # first content unit
        >>> print(updated_state.status)  # Status.FAILED
    """

    if CHUNK_NULL_IRI not in state.current_content_unit.iri:
        state.current_content_unit.processed = True
        state.current_content_unit.graph.remap_namespaces(
            old_namespace=DEFAULT_IRI,
            new_namespace=state.current_content_unit.namespace,
        )
        state.processed_content_units.append(state.current_content_unit)

    if state.content_units:
        state.current_content_unit = state.content_units.pop(0)
        state.node_visits = defaultdict(int)
        state.status = Status.FAILED
        # TODO use method for easier tracing
    else:
        state.current_content_unit = ContentUnit(
            text="",
            index=0,
            hid="default",
            doc_iri=URIRef(CHUNK_NULL_IRI),
        )
        state.status = Status.SUCCESS
        logger.info(
            f"All content units processed ({len(state.processed_content_units)} total), "
            "setting status to SUCCESS and proceeding to AGGREGATE_FACTS"
        )

    return state
