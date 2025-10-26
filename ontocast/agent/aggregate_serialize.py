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
    total_chunks = len(state.chunks_processed)
    logger.info(
        f"Aggregating {total_chunks} processed chunks: "
        f"ontology {len(state.current_ontology.graph)} triples; "
        f"facts graph: {len(state.aggregated_facts)} triples"
    )

    # Check if the ontology was updated during processing
    # If there were updates applied, increment the version (MAJOR/MINOR/PATCH)
    if state.ontology_updates_applied:
        logger.info(
            f"Ontology was updated during processing ({len(state.ontology_updates_applied)} update operations). "
            f"Analyzing changes to determine version increment..."
        )
        # Pass the updates to analyze and increment version appropriately
        state.current_ontology.mark_as_updated(state.ontology_updates_applied)
        # Sync the updated properties (version and updated_at) to the graph
        state.current_ontology.sync_properties_to_graph()
    else:
        logger.debug(
            f"Ontology unchanged during processing (version: {state.current_ontology.version})"
        )

    # Report LLM budget usage
    if state.budget_tracker:
        logger.info(state.budget_tracker.get_summary())
    tools.serialize(state)
    return state
