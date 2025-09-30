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
        state.chunks_processed, state.doc_namespace
    )
    logger.info(
        f"chunks proc: {len(state.chunks_processed)}\n"
        f"facts graph: {len(state.aggregated_facts)} triples\n"
        f"onto graph {len(state.current_ontology.graph)} triples"
    )
    tools.serialize(state)
    return state
