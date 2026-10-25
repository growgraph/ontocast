"""Serialization agent for OntoCast.

This module provides functionality for serializing the knowledge graph
(ontology and facts) to the triple store.
"""

import logging

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def serialize(state: AgentState, tools: ToolBox) -> AgentState:
    """Serialize the knowledge graph to the triple store.

    This function:
    - Handles version management for updated ontologies
    - Tracks budget usage
    - Serializes both ontology and facts to the triple store

    Args:
        state: Current agent state with ontology and facts
        tools: ToolBox containing serialization tools

    Returns:
        Updated agent state after serialization
    """
    # Initialize empty facts graph if not set (for ontology-only render mode)
    if state.aggregated_facts is None:
        state.aggregated_facts = RDFGraph()
        logger.info("No facts to serialize (ontology-only render mode)")

    # Report LLM budget usage
    if state.budget_tracker:
        logger.info(state.budget_tracker.get_summary())
        duration_summary = state.budget_tracker.get_duration_summary()
        if duration_summary:
            logger.info(duration_summary)
        parallelism_summary = state.budget_tracker.get_parallelism_summary()
        if parallelism_summary:
            logger.info(parallelism_summary)

    provenance_graph_uri = f"{str(state.graph_uri).rstrip('/')}/ontology-provenance"
    if len(state.ontology_provenance_artifact) > 0:
        logger.info(
            "Persisting ontology provenance artifact (%d triples) to graph %s",
            len(state.ontology_provenance_artifact),
            provenance_graph_uri,
        )
        if tools.triple_store_manager is not None:
            tools.triple_store_manager.serialize(
                state.ontology_provenance_artifact,
                graph_uri=provenance_graph_uri,
            )

    tools.serialize(state)
    return state
