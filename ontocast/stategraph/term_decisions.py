"""Reduce-time decisions on the terms a document minted.

Runs inside the ontology reduce, after unit deltas are merged and before the
update is partitioned and written. Two carriers hold minted terms: the merged
insert delta (units that extended a catalog ontology) and the fresh ontology
graphs (units that started the document's new home ontology). Both are decided
together and patched in place.

Order: list the minted terms; split off the candidates the plan did not name;
flag the terms that need a decision; ask consolidation about the flagged terms
only (when on, and only when any is flagged); compile the actions, defaults
included, onto each carrier.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence

from ontocast.agent.consolidate_ontology import (
    build_term_contexts,
    consolidate_terms,
    format_other_terms,
)
from ontocast.onto.model import OntologyUnitFinding, OntologyUnitFindingKind
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool.ontology_validation.catalog_terms import catalog_prefixes
from ontocast.tool.ontology_validation.consolidate import (
    compile_actions,
    resolve_actions,
    structural_flags,
)
from ontocast.tool.ontology_validation.plan import minted_terms, plan_candidates
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
    candidates = plan_candidates(carriers, plan, catalog_index)
    metrics = state.ontology_reduce_metrics
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
        flagged.setdefault(iri, set()).add("candidate")
    for finding in findings:
        if finding.kind in REVIEW_FINDING_KINDS and finding.subject in minted:
            flagged.setdefault(finding.subject, set()).add(str(finding.kind))
    for duplicate in duplicates:
        if duplicate.minted_iri in minted:
            flagged.setdefault(duplicate.minted_iri, set()).add("catalog_duplicate")
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
            duplicates=duplicates,
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
