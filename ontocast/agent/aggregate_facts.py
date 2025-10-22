"""Fact aggregation agent for OntoCast.

This module provides functionality for aggregating and serializing facts from
multiple chunks into a single RDF graph, handling entity and predicate
disambiguation.
"""

import logging

from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def aggregate_serialize(state: AgentState, tools: ToolBox) -> AgentState:
    """Create a node that saves the knowledge graph."""

    for c in state.chunks_processed:
        c.sanitize()

    state.aggregated_facts = tools.aggregator.aggregate_graphs(
        chunks=state.chunks_processed, doc_namespace=state.doc_namespace
    )
    logger.info(
        f"chunks proc: {len(state.chunks_processed)}; "
        f"ontology {len(state.current_ontology.graph)} triples; "
        f"facts graph: {len(state.aggregated_facts)} triples"
    )
    tools.serialize(state)
    return state
