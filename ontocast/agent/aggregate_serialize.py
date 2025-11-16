"""Fact aggregation agent for OntoCast.

This module provides functionality for aggregating and serializing facts from
multiple chunks into a single RDF graph, handling entity and predicate
disambiguation.
"""

import logging

from rdflib import DCTERMS, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def aggregate(state: AgentState, tools: ToolBox) -> AgentState:
    """Aggregate facts from multiple processed chunks into a single RDF graph.

    Args:
        state: Current agent state with processed chunks
        tools: ToolBox containing aggregation tools

    Returns:
        Updated agent state with aggregated facts
    """
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

    # Add provenance information if source URL is available
    if state.source_url and state.doc_namespace:
        doc_iri = URIRef(state.doc_namespace)
        source_url_uri = URIRef(state.source_url)
        state.aggregated_facts.add((doc_iri, DCTERMS.source, source_url_uri))
        logger.info(
            f"Added provenance: {state.doc_namespace} dcterms:source {state.source_url}"
        )

    return state


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
    # Initialize empty facts graph if not set (for skip_facts_rendering case)
    if not hasattr(state, "aggregated_facts") or state.aggregated_facts is None:
        state.aggregated_facts = RDFGraph()
        logger.info("No facts to serialize (skip_facts_rendering mode)")

    # Check if the ontology was updated during processing
    # If there were updates applied, increment the version (MAJOR/MINOR/PATCH)
    if state.ontology_updates_applied:
        logger.info(
            f"Ontology was updated during processing ({len(state.ontology_updates_applied)} update operations). "
            f"Analyzing changes to determine version increment..."
        )
        # Pass the updates to analyze and increment version appropriately
        state.current_ontology.mark_as_updated(state.ontology_updates_applied)
        # Sync the updated properties (version and created_at) to the graph
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
