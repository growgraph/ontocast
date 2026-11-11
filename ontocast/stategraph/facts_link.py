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

from rdflib import URIRef

from ontocast.agent import link_facts
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import (
    FactsGateRepairKind,
    FactsValidationFindingKind,
    GraphRepairRecord,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.facts_gate import drop_detached_values
from ontocast.tool.agg.rewriter import add_reified_provenance
from ontocast.tool.facts_validation.connectivity import facts_connectivity
from ontocast.tool.facts_validation.linking import (
    select_link_candidates,
    validate_link_proposals,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: Links kept on the state for audit; the full count is ``facts_link_added``.
_LINKS_STATE_CAP = 200


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
    await _link_isolated(state, tools, ontology_graph, units)
    _drop_detached_values(state, ontology_graph)


def _fact_namespaces(state: AgentState) -> list[str]:
    return [DEFAULT_IRI, str(state.doc_iri), state.doc_namespace or ""]


def _drop_detached_values(state: AgentState, ontology_graph: RDFGraph) -> None:
    """Drop the values the gate and the link pass left isolated, and recount."""
    if not state.facts_isolated_instances:
        return
    dropped = drop_detached_values(
        state.aggregated_facts,
        state.facts_isolated_instances,
        ontology_graph,
        _fact_namespaces(state),
    )
    if not dropped:
        return
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
    state.facts_gate_repairs = [
        *state.facts_gate_repairs,
        *(
            GraphRepairRecord(
                kind=FactsGateRepairKind.UNLINKED_DROPPED, source=iri, target=""
            )
            for iri in dropped
        ),
    ]


async def _link_isolated(
    state: AgentState,
    tools: ToolBox,
    ontology_graph: RDFGraph,
    units: Sequence[ContentUnit],
) -> None:
    """The link pass proper: links the text states for isolated instances."""
    config = tools.config.get_tool_config().facts_validation
    graph = state.aggregated_facts
    if not config.link_pass or not state.facts_isolated_instances or not len(graph):
        return
    fact_namespaces = _fact_namespaces(state)
    # Property names are listed, and answered, under the catalog's prefixes.
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
        proposals, candidates, graph, ontology_graph, units
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
