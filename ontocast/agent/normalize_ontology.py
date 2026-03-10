"""Reducers for parallel map/reduce workflow outputs."""

import logging
from typing import Protocol

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


class AggregatorProtocol(Protocol):
    def aggregate_graphs(self, units: list[ContentUnit]) -> RDFGraph: ...


class NormalizeToolsProtocol(Protocol):
    @property
    def aggregator(self) -> AggregatorProtocol: ...


def normalize_ontology_units(
    units: list[ContentUnit],
    tools: ToolBox | NormalizeToolsProtocol,
    base_ontology: Ontology | None = None,
    require_base: bool = False,
) -> tuple[Ontology, list[GraphUpdate]]:
    """Aggregate delta graphs from unit outputs, then apply merged delta to base.

    Units contain delta graphs (insert triples only). EmbeddingBasedAggregator
    disambiguates and merges them. The merged delta is applied to the base
    ontology graph.

    Args:
        units: ContentUnits with type=ONTOLOGIES and delta graph from each unit.
        tools: ToolBox with aggregator.
        base_ontology: Optional ontology to use as base; merged delta is applied to it.
        require_base: Whether map/reduce caller expects a base ontology.

    Returns:
        Tuple of (Ontology with aggregated graph, list of applied GraphUpdates for versioning).
    """
    if not units:
        if base_ontology is not None:
            return base_ontology, []
        return Ontology(graph=RDFGraph()), []

    for unit in units:
        unit.sanitize()
    aggregated_delta = tools.aggregator.aggregate_graphs(units=units)

    if require_base and (base_ontology is None or base_ontology.is_null()):
        logger.warning(
            "normalize_ontology_units expected a base ontology but none was available; "
            "continuing with merged aggregated ontology output."
        )

    merged_update: GraphUpdate | None = None
    if len(aggregated_delta) > 0:
        merged_update = GraphUpdate(
            triple_operations=[TripleOp(type="insert", graph=aggregated_delta)]
        )

    if base_ontology is not None and not base_ontology.is_null():
        base_graph = base_ontology.graph
        if merged_update is not None:
            updated_graph, _ = AgentState.render_updated_graph(
                base_graph, [merged_update], max_triples=None
            )
            graph_changed = set(updated_graph) != set(base_graph)
            if graph_changed:
                result = base_ontology.derive_updated_version(updated_graph)
            else:
                result = base_ontology.model_copy(deep=True)
                result.graph = updated_graph
        else:
            result = base_ontology.model_copy(deep=True)
        result.sync_properties_to_graph()
        applied = [merged_update] if merged_update else []
        return result, applied

    result = Ontology(
        graph=aggregated_delta,
        ontology_id=base_ontology.ontology_id if base_ontology else None,
        title=base_ontology.title if base_ontology else None,
        description=base_ontology.description if base_ontology else None,
    )
    applied = [merged_update] if merged_update else []
    return result, applied
