"""The link pass, after the validation gate: links the text states for isolated instances.

Shared by the document graph (``LINK_FACTS``) and the single-unit path, like
:mod:`ontocast.stategraph.facts_gate`. It must run after the gate's last
re-aggregation: a later one would rebuild the graph from the units and lose
the links. Values still isolated after it are dropped, whether the pass ran
or not.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from rdflib import RDFS, Literal, URIRef

from ontocast.agent import link_facts
from ontocast.onto.annotation import is_annotation
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import FactsIsolatedInstance, FactsValidationFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.facts_gate import drop_detached_values
from ontocast.tool.agg.rewriter import add_reified_provenance
from ontocast.tool.facts_validation.connectivity import (
    facts_connectivity,
    value_parts,
)
from ontocast.tool.facts_validation.linking import (
    select_link_candidates,
    validate_link_proposals,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: Links kept on the state for audit; the full count is ``facts_link_added``.
_LINKS_STATE_CAP = 200
#: Isolated instances listed in the manifest; the count is ``facts_isolated``.
_ISOLATED_STATE_CAP = 200


async def run_link_pass(
    state: AgentState,
    tools: ToolBox,
    ontology_graph: RDFGraph,
    units: Sequence[ContentUnit],
) -> None:
    """Link the gate's isolated instances, then drop the values still isolated.

    The link pass is one model call, only when ``FACTS_LINK_PASS`` is on and
    the gate left an isolated instance a source unit names. Every accepted link
    is added with a reifier deriving it from the unit whose text holds its
    evidence. Then every isolated instance typed as a value is removed
    (:func:`~ontocast.stategraph.facts_gate.drop_detached_values`), counted in
    ``facts_unlinked_dropped`` with records of kind ``unlinked_dropped``.
    ``facts_isolated`` and ``facts_components`` are rewritten for the served
    graph; ``facts_link_isolated`` keeps the gate's count. Validation findings
    describe the graph the gate validated.

    Args:
        state: Document state after the gate, carrying ``aggregated_facts`` and
            ``facts_isolated_instances``.
        tools: Dependency container.
        ontology_graph: The ontology context the facts were validated against.
        units: The document's content units.
    """
    value_classes = tools.get_atomic_tools().value_classes()
    await _link_isolated(state, tools, ontology_graph, units, value_classes)
    _drop_detached_values(state, value_classes)
    state.facts_isolated_report = isolated_report(
        state.aggregated_facts, state.facts_isolated_instances
    )[:_ISOLATED_STATE_CAP]


def _fact_namespaces(state: AgentState) -> list[str]:
    return [DEFAULT_IRI, str(state.doc_iri), state.doc_namespace or ""]


def isolated_report(
    graph: RDFGraph, isolated: Sequence[str]
) -> list[FactsIsolatedInstance]:
    """The instances still linked to nothing, each with its label and kind.

    ``holds_value`` when it carries a value (a blank node, or a part named
    under it) or a non-annotation literal: a statement whose subject the
    graph never names. ``other`` when it carries neither.
    """
    report: list[FactsIsolatedInstance] = []
    for iri in isolated:
        node = URIRef(iri)
        label = graph.value(node, RDFS.label)
        holds = bool(value_parts(graph, node)) or any(
            isinstance(obj, Literal) and not is_annotation(predicate)
            for predicate, obj in graph.predicate_objects(node)
        )
        report.append(
            FactsIsolatedInstance(
                iri=iri,
                label=str(label) if label is not None else "",
                kind="holds_value" if holds else "other",
            )
        )
    return report


def _drop_detached_values(state: AgentState, value_classes: frozenset[str]) -> None:
    """Drop the values the gate and the link pass left isolated, and recount."""
    if not state.facts_isolated_instances:
        return
    records = drop_detached_values(
        state.aggregated_facts,
        state.facts_isolated_instances,
        value_classes,
        _fact_namespaces(state),
    )
    if not records:
        return
    dropped = [record.source for record in records]
    gone = set(dropped)
    state.facts_isolated_instances = [
        iri for iri in state.facts_isolated_instances if iri not in gone
    ]
    metrics = state.retrieval_metrics
    unlinked = metrics.get(RetrievalMetric.FACTS_UNLINKED_DROPPED)
    metrics[RetrievalMetric.FACTS_UNLINKED_DROPPED] = (
        unlinked if isinstance(unlinked, int) else 0
    ) + len(dropped)
    metrics[RetrievalMetric.FACTS_ISOLATED] = len(state.facts_isolated_instances)
    # Each dropped value was a component of its own.
    components = metrics.get(RetrievalMetric.FACTS_COMPONENTS)
    if isinstance(components, int):
        metrics[RetrievalMetric.FACTS_COMPONENTS] = components - len(dropped)
    state.facts_gate_repairs = [*state.facts_gate_repairs, *records]


async def _link_isolated(
    state: AgentState,
    tools: ToolBox,
    ontology_graph: RDFGraph,
    units: Sequence[ContentUnit],
    value_classes: frozenset[str],
) -> None:
    """The link pass proper: links the text states for isolated instances."""
    config = tools.config.get_tool_config().facts_validation
    graph = state.aggregated_facts
    if not config.link_pass or not state.facts_isolated_instances or not len(graph):
        return
    fact_namespaces = _fact_namespaces(state)
    # Property names are listed, and answered, under the catalog's prefixes.
    # The facts context is a prompt graph merged from copies of the catalog,
    # so binding it in place never touches a stored ontology.
    tools.prefix_table().normalize(ontology_graph)
    # The term a non-catalog finding names is its predicate field.
    refused = {
        finding.predicate
        for finding in state.facts_validation_findings
        if finding.kind == FactsValidationFindingKind.NON_CATALOG_VOCABULARY
    }
    candidates = select_link_candidates(
        graph,
        state.facts_isolated_instances,
        units,
        ontology_graph,
        fact_namespaces,
        max_isolated=config.link_pass_max_isolated,
        excluded_predicates=refused,
    )
    if candidates is None:
        return
    proposals = await link_facts.propose_links(
        state, tools, candidates, graph, ontology_graph
    )
    accepted, rejected = validate_link_proposals(
        proposals, candidates, graph, ontology_graph, units, value_classes=value_classes
    )
    metrics = state.retrieval_metrics
    metrics[RetrievalMetric.FACTS_LINK_ISOLATED] = len(state.facts_isolated_instances)
    metrics[RetrievalMetric.FACTS_LINK_PROPOSED] = len(proposals)
    metrics[RetrievalMetric.FACTS_LINK_ADDED] = len(accepted)
    metrics[RetrievalMetric.FACTS_LINK_REJECTIONS_BY_REASON] = {
        str(reason): count for reason, count in sorted(rejected.items())
    }
    if not accepted:
        return

    before = facts_connectivity(graph, fact_namespaces, document=state.doc_iri)
    for link in accepted:
        subject, predicate, obj = (
            URIRef(link.subject),
            URIRef(link.predicate),
            URIRef(link.object),
        )
        graph.add((subject, predicate, obj))
        add_reified_provenance(graph, subject, predicate, obj, URIRef(link.unit))
    after = facts_connectivity(graph, fact_namespaces, document=state.doc_iri)

    linked = {end for link in accepted for end in (link.subject, link.object)}
    state.facts_isolated_instances = [
        iri for iri in state.facts_isolated_instances if iri not in linked
    ]
    metrics[RetrievalMetric.FACTS_ISOLATED] = len(state.facts_isolated_instances)
    # The gate counted before naming values; the difference across the pass is
    # what the links joined, whatever the naming did to the absolute count.
    gate_components = metrics.get(RetrievalMetric.FACTS_COMPONENTS)
    if isinstance(gate_components, int):
        metrics[RetrievalMetric.FACTS_COMPONENTS] = gate_components - (
            before.components - after.components
        )
    state.facts_links_added = [*state.facts_links_added, *accepted][:_LINKS_STATE_CAP]
    logger.info(
        "Link pass: %d of %d proposed link(s) added for %d isolated instance(s)",
        len(accepted),
        len(proposals),
        len(candidates.isolated),
    )


__all__ = ["run_link_pass"]
