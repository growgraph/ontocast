"""Reduce-time decisions on the terms a document minted.

Runs inside the ontology reduce, after unit deltas are merged and before the
update is partitioned and written. Two carriers hold minted terms: the merged
insert delta (units that extended a catalog ontology) and the fresh ontology
graphs (units that started the document's new home ontology). Both are decided
together and patched in place.

Order: list the minted terms; write the plan's parents, domains and ranges
over the render's; split off the candidates the plan did not name;
flag the terms that need a decision; ask consolidation about the flagged terms
only (when on, and only when any is flagged); compile the actions, defaults
included, onto each carrier.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence

from rdflib import URIRef

from ontocast.agent.consolidate_ontology import (
    build_term_contexts,
    consolidate_terms,
    format_other_terms,
)
from ontocast.onto.model import OntologyUnitFinding, OntologyUnitFindingKind
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool.ontology_validation.axiom_hygiene import (
    has_generic_parent,
    property_side_reasons,
)
from ontocast.tool.ontology_validation.catalog_terms import catalog_prefixes
from ontocast.tool.ontology_validation.consolidate import (
    add_flag,
    compile_actions,
    resolve_actions,
    structural_flags,
)
from ontocast.tool.ontology_validation.plan import (
    apply_plan_axioms,
    minted_terms,
    plan_candidates,
)
from ontocast.tool.ontology_validation.reconcile import MintedDuplicate
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: Findings that question a term's level or identity, so a term carrying one
#: is put to consolidation. Hygiene findings (namespace, label, cycle, role,
#: restriction, delete) are not: keep, merge or drop cannot repair them, and
#: they have their own checks.
REVIEW_FINDING_KINDS = frozenset(
    {
        OntologyUnitFindingKind.SPECIFIC_LABEL_LEAF,
        OntologyUnitFindingKind.VALUE_SET_SIBLINGS,
        OntologyUnitFindingKind.CATALOG_DUPLICATE,
        OntologyUnitFindingKind.LABEL_COLLISION,
        OntologyUnitFindingKind.HIERARCHY_TOO_DEEP,
    }
)


def _side_counts(
    minted: Mapping[str, str], carriers: Sequence[RDFGraph]
) -> tuple[int, int]:
    """Generic and missing domains and ranges over the minted properties."""
    generic = missing = 0
    for iri, role in minted.items():
        if role != "property":
            continue
        for reason in property_side_reasons(URIRef(iri), carriers):
            if reason.startswith("generic_"):
                generic += 1
            else:
                missing += 1
    return generic, missing


async def decide_minted_terms(
    state: AgentState,
    tools: ToolBox,
    *,
    merged_inserts: RDFGraph,
    fresh_ontologies: Sequence[Ontology],
    findings: Sequence[OntologyUnitFinding],
    term_units: Mapping[str, set[int]],
    duplicates: Sequence[MintedDuplicate],
) -> None:
    """Decide and apply an action for every minted term, in place.

    Nothing changes when there is no plan and consolidation is off, or when no
    term was minted. Metrics go to ``state.ontology_reduce_metrics``.
    """
    config = tools.config.get_tool_config().ontology_validation
    plan = state.ontology_plan
    consolidate = config.consolidation == "on"
    if plan is None and not consolidate:
        return
    catalog_index = tools.ontology_manager.catalog_surface_index()
    carriers: list[RDFGraph] = [merged_inserts, *(o.graph for o in fresh_ontologies)]
    minted = minted_terms(carriers, catalog_index)
    if not minted:
        return
    metrics = state.ontology_reduce_metrics
    if plan is not None:
        # The plan resolved each term's parent and sides against the whole
        # catalog; a render saw one unit, so its values give way.
        written = apply_plan_axioms(carriers, plan, minted)
        metrics["plan_sides_written"] = written.sides
        metrics["plan_parents_written"] = written.parents
    candidates = plan_candidates(carriers, plan, catalog_index)
    metrics["minted_terms_before_decisions"] = len(minted)
    metrics["plan_candidates"] = len(candidates)
    if candidates:
        metrics["plan_candidate_iris"] = sorted(candidates)

    prefixes = (
        plan.prefixes
        if plan is not None and plan.prefixes
        else catalog_prefixes(
            list(tools.ontology_manager.ontologies),
            state.home_ontology,
            tools.prefix_table(),
        )
    )
    flagged = structural_flags(minted, carriers, catalog_index)
    for iri in candidates:
        add_flag(flagged, iri, "candidate")
    for finding in findings:
        if finding.kind in REVIEW_FINDING_KINDS and finding.subject in minted:
            add_flag(flagged, finding.subject, str(finding.kind))
    for duplicate in duplicates:
        if duplicate.minted_iri in minted:
            add_flag(
                flagged,
                duplicate.minted_iri,
                "catalog_duplicate",
                (duplicate.catalog_iri,),
            )
    metrics["consolidation_flagged"] = len(flagged)

    actions = []
    if consolidate and not flagged:
        # Every term is planned and nothing flags it: the defaults keep them
        # all, so a call could only confirm them.
        metrics["consolidation_skipped"] = 1
    elif consolidate:
        contexts = build_term_contexts(
            state,
            carriers=carriers,
            minted=minted,
            flagged=flagged,
            candidates=candidates,
            findings=findings,
            term_units=term_units,
            plan=plan,
        )
        actions, failed = await consolidate_terms(
            state,
            tools,
            contexts,
            prefixes=prefixes,
            batch_size=config.consolidation_max_terms,
            other_terms=format_other_terms(minted, flagged, carriers, prefixes),
        )
        metrics["consolidation_calls_failed"] = failed

    decided, ignored = resolve_actions(
        actions,
        minted=minted,
        candidates=candidates,
        catalog_index=catalog_index,
        prefixes=prefixes,
        reviewed=flagged,
    )
    changed = sum(
        compile_actions(decided, carrier, minted=minted).apply_to(carrier)
        for carrier in carriers
    )
    counts = dict(Counter(action.action for action in decided.values()))
    for ontology in fresh_ontologies:
        ontology.content_hash()
    for action in (
        "keep",
        "merge_into",
        "generalise_to",
        "demote_to_property_value",
        "drop",
    ):
        metrics[f"consolidation_{action}"] = counts.get(action, 0)
    surviving = {
        iri: role
        for iri, role in minted.items()
        if decided[iri].action not in ("drop", "merge_into")
    }
    metrics["generic_property_sides"], metrics["missing_property_sides"] = _side_counts(
        surviving, carriers
    )
    metrics["generic_parents"] = sum(
        has_generic_parent(URIRef(iri), carriers) for iri in surviving
    )
    metrics["consolidation_actions_ignored"] = len(ignored)
    metrics["consolidation_changed_triples"] = changed
    metrics["plan_candidates_dropped"] = sum(
        1 for iri in candidates if decided[iri].action == "drop"
    )
    for term, reason in ignored:
        logger.debug("Consolidation action on %s ignored: %s", term, reason)
    logger.info(
        "Minted terms decided: %s (%d candidates, %d actions ignored)",
        counts,
        len(candidates),
        len(ignored),
    )
