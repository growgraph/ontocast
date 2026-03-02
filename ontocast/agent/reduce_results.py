"""Reducers for parallel map/reduce workflow outputs."""

import logging
from copy import deepcopy

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox
from ontocast.util import render_text_hash

logger = logging.getLogger(__name__)


def _update_fingerprint(update: GraphUpdate) -> str:
    """Build a stable fingerprint from generated SPARQL queries."""
    queries = update.generate_sparql_queries()
    payload = "\n".join(queries)
    return render_text_hash(payload, digits=None)


def _canonicalize_updates(updates: list[GraphUpdate]) -> list[GraphUpdate]:
    """Deduplicate and deterministically order ontology updates."""
    seen: dict[str, GraphUpdate] = {}
    for update in updates:
        fp = _update_fingerprint(update)
        if fp not in seen:
            seen[fp] = update
    return [seen[key] for key in sorted(seen.keys())]


def reduce_facts_units(units: list[ContentUnit], tools: ToolBox) -> RDFGraph:
    """Aggregate facts graphs from successful unit outputs."""
    if not units:
        return RDFGraph()

    for unit in units:
        unit.sanitize()
    return tools.aggregator.aggregate_graphs(units=units)


def reduce_ontology_updates(
    base_ontology: Ontology,
    updates: list[GraphUpdate],
    ontology_max_triples: int | None,
) -> Ontology:
    """Apply ontology updates deterministically on top of a baseline ontology."""
    if not updates:
        return base_ontology

    canonical_updates = _canonicalize_updates(updates)
    logger.info(
        f"Reducing ontology updates: {len(updates)} input, "
        f"{len(canonical_updates)} canonical"
    )

    updated = deepcopy(base_ontology)
    updated_graph, was_applied = AgentState.render_updated_graph(
        graph=updated.graph,
        updates=canonical_updates,
        max_triples=ontology_max_triples,
    )
    if not was_applied:
        logger.warning("Ontology reducer skipped updates due to max_triples limit")
        return base_ontology

    updated.graph = updated_graph
    updated.sync_properties_to_graph()
    return updated
