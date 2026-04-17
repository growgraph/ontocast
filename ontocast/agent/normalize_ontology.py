"""Reducers for parallel map/reduce workflow outputs."""

import logging

from rdflib import OWL, RDF, BNode, Node, URIRef

from ontocast.onto.constants import PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def split_ontology_and_provenance_graph(
    graph: RDFGraph,
) -> tuple[RDFGraph, RDFGraph]:
    """Split normalized ontology graph into clean ontology + provenance artifact.

    Provenance/reification and normalization-time alignment artifacts are moved
    to a side graph so downstream consolidation works with a clean ontology graph.
    """
    clean_graph = RDFGraph()
    provenance_graph = RDFGraph()

    for prefix, namespace in graph.namespaces():
        if prefix:
            clean_graph.bind(prefix, namespace)
            provenance_graph.bind(prefix, namespace)

    reifier_nodes: set[BNode] = {
        subject
        for subject, _, _ in graph.triples((None, RDF_REIFIES, None))
        if isinstance(subject, BNode)
    }
    chunk_nodes: set[Node] = set()

    def is_schema_chunk_metadata(predicate: Node) -> bool:
        predicate_str = str(predicate)
        return predicate_str in {
            str(SCHEMA.identifier),
            str(SCHEMA.position),
            "http://schema.org/identifier",
            "http://schema.org/position",
        }

    for subject, predicate, obj in graph:
        if is_schema_chunk_metadata(predicate) or predicate == PROV.generatedAtTime:
            chunk_nodes.add(subject)
        if predicate == RDF.type and str(obj) in {
            str(PROV.Entity),
            str(SCHEMA.text),
            "http://schema.org/text",
        }:
            chunk_nodes.add(subject)

    def is_provenance_or_alignment_triple(
        subject: Node, predicate: Node, obj: Node
    ) -> bool:
        if predicate == RDF_REIFIES:
            return True
        if predicate == PROV.wasDerivedFrom:
            # Keep ontology lineage hashes in the clean ontology graph.
            if isinstance(obj, URIRef) and str(obj).startswith("urn:hash:"):
                return False
            return True
        if predicate == PROV.generatedAtTime or is_schema_chunk_metadata(predicate):
            return True
        if predicate == OWL.sameAs:
            return True
        if subject in reifier_nodes or obj in reifier_nodes:
            return True
        if subject in chunk_nodes or obj in chunk_nodes:
            return True
        if predicate == RDF.type and str(obj) in {
            str(PROV.Entity),
            str(SCHEMA.text),
            "http://schema.org/text",
        }:
            return True
        return False

    for triple in graph:
        if is_provenance_or_alignment_triple(*triple):
            provenance_graph.add(triple)
        else:
            clean_graph.add(triple)

    return clean_graph, provenance_graph


def normalize_ontology_units(
    units: list[ContentUnit],
    tools: ToolBox,
    base_ontology: Ontology | None = None,
    require_base: bool = False,
) -> tuple[Ontology, list[GraphUpdate], RDFGraph]:
    """Merge ontology unit deltas as TripleOps, then apply to base ontology.

    Units contain ontology delta graphs (insert triples only). To preserve the
    exact unit output shape (and avoid ontology/facts aggregation rewrites), we
    convert each unit graph into an ``insert`` TripleOp and apply them as one
    GraphUpdate.

    Args:
        units: ContentUnits with type=ONTOLOGIES and delta graph from each unit.
        tools: ToolBox instance.
        base_ontology: Optional ontology to use as base; merged delta is applied to it.
        require_base: Whether map/reduce caller expects a base ontology.

    Returns:
        Tuple of (
            ontology with cleaned graph,
            list of applied GraphUpdates for versioning,
            provenance artifact graph stripped from ontology output,
        ).
    """
    if not units:
        if base_ontology is not None:
            return base_ontology, [], RDFGraph()
        return Ontology(graph=RDFGraph()), [], RDFGraph()

    for unit in units:
        unit.sanitize()
    _ = tools

    if require_base and (base_ontology is None or base_ontology.is_null()):
        logger.warning(
            "normalize_ontology_units expected a base ontology but none was available; "
            "continuing with merged aggregated ontology output."
        )

    # Unit delta graphs contain insert-only triples produced by build_ontology_delta_graph.
    # Delete operations are intentionally excluded at the map stage and are not
    # represented here. This is a deliberate policy: parallel unit deletes cannot
    # be safely reconciled without a consensus pass, which is not yet implemented.
    merged_update = GraphUpdate(
        triple_operations=[
            TripleOp(type="insert", graph=unit.graph)
            for unit in units
            if len(unit.graph) > 0
        ]
    )
    if not merged_update.triple_operations:
        merged_update = None

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
        cleaned_graph, provenance_graph = split_ontology_and_provenance_graph(
            result.graph
        )
        result.graph = cleaned_graph
        result.sync_properties_to_graph()
        applied = [merged_update] if merged_update else []
        return result, applied, provenance_graph

    aggregated_delta = RDFGraph()
    for unit in units:
        for triple in unit.graph:
            aggregated_delta.add(triple)
        for prefix, namespace in unit.graph.namespaces():
            if prefix:
                aggregated_delta.bind(prefix, namespace)

    cleaned_graph, provenance_graph = split_ontology_and_provenance_graph(
        aggregated_delta
    )
    result = Ontology(
        graph=cleaned_graph,
        ontology_id=base_ontology.ontology_id if base_ontology else None,
        title=base_ontology.title if base_ontology else None,
        description=base_ontology.description if base_ontology else None,
    )
    applied = [merged_update] if merged_update else []
    return result, applied, provenance_graph
