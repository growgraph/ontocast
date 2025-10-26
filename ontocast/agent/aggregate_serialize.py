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

    # Check if ontology version was updated by the LLM (via GraphUpdate)
    # and increment patch version if it changed, but only if it hasn't already been updated
    if state.current_ontology.initial_version is not None:
        initial_version = state.current_ontology.initial_version
        current_version = state.current_ontology.version

        # If the version changed (was updated by LLM), increment the patch version
        if initial_version != current_version:
            logger.info(
                f"Ontology version changed from {initial_version} to {current_version} "
                f"(updated by LLM). Incrementing patch version..."
            )
            state.current_ontology.mark_as_updated()
            # Sync the updated properties (version and updated_at) to the graph
            state.current_ontology.sync_properties_to_graph()
        else:
            # No version change, so the ontology wasn't updated
            logger.debug(f"Ontology version unchanged: {current_version}")

    # Report LLM budget usage
    if state.llm_budget_tracker:
        logger.info(state.llm_budget_tracker.get_summary())
    tools.serialize(state)
    return state
