"""Reducers for parallel map/reduce workflow outputs."""

import logging

from rdflib import OWL, Node

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.identity import validate_ontology_iri
from ontocast.onto.namespace_merge import merge_namespace_bindings
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import is_scaffolding, scaffolding_nodes
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.state import AgentState
from ontocast.onto.util import is_rdflib_default_namespace
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def _working_anchor_from_graph(graph: RDFGraph) -> str | None:
    """Pick a stable working-context IRI from the first non-standard namespace.

    Used when aggregating unit deltas without a catalog base ontology. Does not
    set ``ontology_id`` from the prefix name. A namespace that is not a valid
    ontology IRI is passed over: a prefix-shaped one (``fro:``, an undeclared
    JSON-LD prefix parsed as a scheme) would otherwise become an ontology
    identity that collides at serialization.
    """
    for prefix, namespace in graph.namespaces():
        if not prefix:
            continue
        ns = str(namespace)
        if is_rdflib_default_namespace(ns):
            continue
        try:
            return validate_ontology_iri(ns.rstrip("/#"))
        except ValueError:
            continue
    return None


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

    scaffolding = scaffolding_nodes(graph)

    def is_provenance_or_alignment_triple(
        subject: Node, predicate: Node, obj: Node
    ) -> bool:
        # The header's source-document link is not scaffolding: it stays with
        # the ontology. owl:sameAs is the aggregator's alignment bookkeeping.
        return predicate == OWL.sameAs or is_scaffolding(
            (subject, predicate, obj), scaffolding
        )

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
    delete_graph: RDFGraph | None = None,
) -> tuple[Ontology, list[GraphUpdate], RDFGraph]:
    """Merge ontology unit deltas as TripleOps, then apply to base ontology.

    Units contain ontology insert delta graphs; ``delete_graph`` carries the
    reconciled delete delta (triples to remove from the base). Deletes execute
    first, then inserts, as one ordered GraphUpdate — so the applied update
    list feeds version-bump analysis with true operation types.

    Args:
        units: ContentUnits with type=ONTOLOGIES and delta graph from each unit.
        tools: ToolBox instance.
        base_ontology: Optional ontology to use as base; merged delta is applied to it.
        require_base: Whether map/reduce caller expects a base ontology.
        delete_graph: Optional triples to delete from the base before inserts.
            Requires a catalog base; ignored (with a warning) otherwise.

    Returns:
        Tuple of (
            ontology with cleaned graph,
            list of applied GraphUpdates for versioning,
            provenance artifact graph stripped from ontology output,
        ).
    """
    has_deletes = delete_graph is not None and len(delete_graph) > 0
    if not units and not has_deletes:
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

    operations: list[TripleOp] = []
    if has_deletes:
        if base_ontology is None or base_ontology.is_null():
            logger.warning(
                "normalize_ontology_units received %d delete triple(s) without a "
                "catalog base; deletes are dropped (nothing to delete from).",
                len(delete_graph) if delete_graph is not None else 0,
            )
        else:
            operations.append(TripleOp(type="delete", graph=delete_graph))
    operations.extend(
        TripleOp(type="insert", graph=unit.graph)
        for unit in units
        if len(unit.graph) > 0
    )
    merged_update = GraphUpdate(triple_operations=operations)
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
                # Nothing changed: the base stays the artifact, and the update
                # is not reported as applied, so no version is stamped for it.
                result = base_ontology.model_copy(deep=True)
                merged_update = None
        else:
            result = base_ontology.model_copy(deep=True)
        cleaned_graph, provenance_graph = split_ontology_and_provenance_graph(
            result.graph
        )
        result.graph = cleaned_graph
        applied = [merged_update] if merged_update else []
        return result, applied, provenance_graph

    aggregated_delta = RDFGraph()
    bindings: dict[str, str] = {}
    for unit in units:
        for triple in unit.graph:
            aggregated_delta.add(triple)
        incoming = {
            prefix: str(namespace)
            for prefix, namespace in unit.graph.namespaces()
            if prefix
        }
        bindings = merge_namespace_bindings(bindings, incoming)
    for prefix, namespace in bindings.items():
        aggregated_delta.bind(prefix, namespace)

    cleaned_graph, provenance_graph = split_ontology_and_provenance_graph(
        aggregated_delta
    )
    if base_ontology is not None and not base_ontology.is_null():
        result = Ontology(
            graph=cleaned_graph,
            ontology_id=base_ontology.ontology_id,
            title=base_ontology.title,
            description=base_ontology.description,
            iri=base_ontology.iri,
            namespace=base_ontology.namespace,
        )
    else:
        # No catalog base: provisional Ontology from first domain namespace stem.
        anchor = _working_anchor_from_graph(cleaned_graph) or NULL_ONTOLOGY.iri
        result = Ontology(
            graph=cleaned_graph,
            ontology_id=None,
            iri=anchor if anchor != NULL_ONTOLOGY.iri else NULL_ONTOLOGY.iri,
            title=None,
            description=None,
        )
    applied = [merged_update] if merged_update else []
    return result, applied, provenance_graph
