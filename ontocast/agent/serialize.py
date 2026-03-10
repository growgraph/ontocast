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

    # Ontology versioning: reduce_ontology sets ontology_updates_applied with the
    # merged GraphUpdate when aggregating parallel ontology units.
    if state.ontology_updates_applied:
        logger.info(
            f"Ontology was updated during processing ({len(state.ontology_updates_applied)} update operations). "
            f"Analyzing changes to determine version increment..."
        )
        state.current_ontology.mark_as_updated(state.ontology_updates_applied)
        state.current_ontology.sync_properties_to_graph()
    elif state.ontology_units:
        logger.debug("Ontology from EmbeddingBasedAggregator; skipping version bump")
    else:
        logger.debug(
            f"Ontology unchanged during processing (version: {state.current_ontology.version})"
        )

    # Report LLM budget usage
    if state.budget_tracker:
        logger.info(state.budget_tracker.get_summary())

    provenance_graph_uri = f"{str(state.graph_uri).rstrip('/')}/ontology-provenance"
    if len(state.ontology_provenance_artifact) > 0:
        logger.info(
            "Persisting ontology provenance artifact (%d triples) to graph %s",
            len(state.ontology_provenance_artifact),
            provenance_graph_uri,
        )
        if tools.filesystem_manager is not None:
            tools.filesystem_manager.serialize(
                state.ontology_provenance_artifact,
                graph_uri=provenance_graph_uri,
            )
        if (
            tools.triple_store_manager is not None
            and tools.triple_store_manager != tools.filesystem_manager
        ):
            tools.triple_store_manager.serialize(
                state.ontology_provenance_artifact,
                graph_uri=provenance_graph_uri,
            )

    tools.serialize(state)
    return state
