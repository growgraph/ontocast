"""Reusable per-unit render/critic retry loops.

These loops are designed for map/reduce execution where each content unit
is processed independently. They deep-copy the incoming unit state, then run
render -> critic until success or retry exhaustion. After the last allowed
render succeeds, the critic is skipped: no further extract exists for feedback
to inform.

Ontology context assembly (``resolve_unit_ontology_context``) runs at the
start of both ``ontology_loop`` and ``facts_loop`` so each unit chooses its
own ontology context according to mode/policy.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from rdflib import URIRef

from ontocast.agent.complete_facts import complete_facts
from ontocast.agent.criticise_facts import criticise_facts
from ontocast.agent.criticise_ontology import criticise_ontology
from ontocast.agent.external_evidence import (
    fetch_external_evidence_for_node,
    plan_external_evidence_for_node,
)
from ontocast.agent.render_facts import render_facts
from ontocast.agent.render_ontology import render_ontology
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import (
    ExternalEvidenceCacheEntry,
    ExternalEvidenceRequest,
    FactsUnitFinding,
    FactsUnitFindingKind,
    GraphRepairRecord,
    LoopAttempt,
    OntologyUnitFinding,
    RolledBackFix,
    Suggestions,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import OntologyContextConfigError
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.stategraph.context_resolver import (
    UnitOntologyContext,
    resolve_unit_ontology_context,
)
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import (
    FactsAcceptancePolicy,
    ValidationPolicy,
    collect_unit_findings,
    material_defects,
    unit_numeric_inventory,
)
from ontocast.tool.facts_validation.critic_patch import (
    CriticPatchPolicy,
    FixPatch,
    compile_critic_fixes,
)
from ontocast.tool.llm import LLMConfigurationError
from ontocast.tool.ontology_validation import collect_ontology_unit_findings
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: The two unit states the loop is generic over. Constrained rather than bound:
#: the driver hands its state to helpers that are themselves generic, and a
#: bound would widen those to the union at every call.
UnitStateT = TypeVar("UnitStateT", UnitFactsState, UnitOntologyState)


def _document_supplemental_ontologies(context: UnitLoopContext) -> list[Ontology]:
    """Non-null reduced ontology artifacts for LLM ingest prefix repair."""
    return [
        ontology for ontology in context.reduced_artifacts() if not ontology.is_null()
    ]


def _catalog_ontologies_for_patch_sources(
    tools: ToolBox,
    patch_sources: list[str],
) -> list[Ontology]:
    """Freshest catalog terminals for each working-context source IRI."""
    if not patch_sources:
        return []
    mgr = tools.ontology_manager
    result: list[Ontology] = []
    seen: set[str] = set()
    for ref in patch_sources:
        iri = mgr.resolve_ontology_ref(ref) or ref
        if iri in seen:
            continue
        onto = mgr.get_freshest_terminal_ontology_by_iri(iri)
        if onto is None or onto.is_null():
            continue
        seen.add(onto.iri)
        result.append(onto)
    return result


def _supplemental_ontologies_for_unit(
    context: UnitLoopContext,
    unit_state: UnitOntologyState | UnitFactsState,
    tools: ToolBox,
) -> list[Ontology]:
    """Document artifacts plus catalog entries for the unit's patch sources."""
    merged: list[Ontology] = []
    seen: set[str] = set()
    for ontology in (
        *_document_supplemental_ontologies(context),
        *_catalog_ontologies_for_patch_sources(
            tools, list(unit_state.ontology_patch_sources)
        ),
    ):
        if ontology.iri in seen:
            continue
        seen.add(ontology.iri)
        merged.append(ontology)
    return merged


def _resolve_max_visits_limit(state_visits: int, override: int | None) -> int:
    """Return a safe visit limit while respecting explicit overrides."""
    visits = state_visits if override is None else override
    return max(1, visits)


def _collect_facts_findings(
    unit_state: UnitFactsState,
    atomic: AtomicToolBox | None = None,
) -> list[FactsUnitFinding]:
    """Run the deterministic per-unit validator against the current graph.

    The toolbox supplies the deployment's namespace exemptions and quantity
    fallback vocabulary; ``None`` (tests) means no exemptions. Terms the
    shapes-derived conformance chapter requires join the exempt set per
    unit -- they come from the tenancy's shapes catalog, which the
    tenancy-independent toolbox policy cannot carry.
    """
    policy = atomic.validation_policy if atomic is not None else None
    if unit_state.shapes_contract_terms:
        base = policy if policy is not None else ValidationPolicy()
        policy = base.model_copy(
            update={
                "contract_exempt_terms": (
                    *base.contract_exempt_terms,
                    *unit_state.shapes_contract_terms,
                )
            }
        )
    coverage_limit = atomic.numeric_coverage_limit if atomic is not None else 30
    coverage_mandatory: str | bool = (
        atomic.numeric_coverage_mandatory if atomic is not None else "off"
    )
    # The whole catalog, not the unit's retrieved snapshot: a term the
    # snapshot omitted is still a term, and must not be reported unknown with
    # a look-alike from the snapshot offered as its replacement.
    full_catalog_terms = atomic.catalog_terms() if atomic is not None else None
    return collect_unit_findings(
        graph=unit_state.content_unit.graph,
        ontology_graph=unit_state.ontology_snapshot.graph,
        quarantined=unit_state.quarantined_literal_triples,
        extraction_text=unit_state.content_unit.extraction_text,
        fact_namespaces=[DEFAULT_IRI, str(unit_state.content_unit.doc_iri)],
        full_catalog_terms=full_catalog_terms,
        # Citation numerics (pages, years, volume numbers) are not extractable
        # quantities — never push coverage repair on bibliography units.
        coverage_limit=(
            0 if unit_state.content_unit.is_citation_metadata else coverage_limit
        ),
        coverage_mandatory=coverage_mandatory,
        policy=policy,
        is_citation_metadata=unit_state.content_unit.is_citation_metadata,
        is_non_content=unit_state.content_unit.is_non_content,
    )


def _collect_ontology_findings(
    unit_state: UnitOntologyState,
    atomic: AtomicToolBox | None = None,
) -> list[OntologyUnitFinding]:
    """Run the deterministic delta validator against the unit's current state.

    Validates the net insert/delete delta (never the shared snapshot), passing
    the already-materialised ``working_graph`` as the merged view so the only
    graph copy is the one inside ``build_delta``. Budget-timed because that
    copy is exactly the cost the shared-by-reference snapshot exists to avoid.
    """
    started = time.perf_counter()
    delta = unit_state.build_delta()
    snapshot_graph = (
        None
        if unit_state.ontology_snapshot.is_empty()
        else unit_state.ontology_snapshot.graph
    )
    findings = collect_ontology_unit_findings(
        inserts=delta.inserts,
        deletes=delta.deletes,
        snapshot_graph=snapshot_graph,
        merged_graph=unit_state.working_graph,
        fact_namespaces=[DEFAULT_IRI, str(unit_state.content_unit.doc_iri)],
        policy=atomic.validation_policy if atomic is not None else None,
    )
    unit_state.budget_tracker.add_duration(
        "ontology_validation/unit_findings", time.perf_counter() - started
    )
    return findings


def _record_attempt(
    unit_state: UnitStateT,
    *,
    kind: Literal[
        "render", "critic", "critic_patch", "critic_skipped", "completion", "llm_repair"
    ],
    render_attempt: int,
    critic_attempt: int = 0,
    n_findings: int = 0,
    n_mandatory: int = 0,
    repair_failed: bool = False,
    n_fixes_applied: int = 0,
    n_fixes_noop: int = 0,
    n_triples_deleted: int = 0,
    n_triples_inserted: int = 0,
    patch_rolled_back: bool = False,
    patch_delete_capped: bool = False,
    accept_reason: str = "",
    rolled_back_fixes: list[RolledBackFix] | None = None,
    n_fixes_junk_refused: int = 0,
    n_fixes_unresolved_prefix: int = 0,
    n_measurements_recovered: int = 0,
) -> None:
    """Append one telemetry record for the current loop attempt.

    ``triple_count`` is the phase's own product measure, so an ontology record
    reports what the unit contributes rather than the size of its scratchpad --
    the working graph is the snapshot plus a small delta and barely moves.
    """
    undone = list(rolled_back_fixes or ())
    unit_state.attempt_log.append(
        LoopAttempt(
            render_attempt=render_attempt,
            critic_attempt=critic_attempt,
            kind=kind,
            success=unit_state.status == Status.SUCCESS,
            accept_reason=accept_reason,
            n_deterministic_findings=n_findings,
            n_mandatory_findings=n_mandatory,
            repair_failed=repair_failed,
            failure_stage=(str(unit_state.failure_stage) if repair_failed else None),
            failure_reason=unit_state.failure_reason if repair_failed else None,
            triple_count=unit_state.product_triple_count(),
            n_fixes_applied=n_fixes_applied,
            n_fixes_noop=n_fixes_noop,
            n_triples_deleted=n_triples_deleted,
            n_triples_inserted=n_triples_inserted,
            patch_rolled_back=patch_rolled_back or bool(undone),
            n_fixes_rolled_back=len(undone),
            rolled_back_fixes=undone,
            patch_delete_capped=patch_delete_capped,
            n_fixes_junk_refused=n_fixes_junk_refused,
            n_fixes_unresolved_prefix=n_fixes_unresolved_prefix,
            n_measurements_recovered=n_measurements_recovered,
        )
    )


@dataclass(frozen=True)
class PatchOutcome:
    """What one critic pass changed, and whether the loop should run another."""

    applied: int = 0
    residual: int = 0
    noop: int = 0
    #: Fixes applied and undone on their own; the rest of the pass stands.
    rolled_back: int = 0
    mandatory_after: int = 0

    @property
    def converged(self) -> bool:
        """True when another pass has nothing left to work with.

        A pass that changed nothing -- every fix rolled back, or nothing to
        apply and nothing mandatory left -- counts as converged: the next pass
        would be handed the same graph and the same findings, so repeating it
        buys a second identical answer at full price.
        """
        return self.applied == 0 and (self.rolled_back > 0 or self.mandatory_after == 0)


def _regression_reason(
    *,
    graph_before: RDFGraph,
    graph_after: RDFGraph,
    product_before: int,
    product_after: int,
    mandatory_before: int,
    mandatory_after: int,
) -> str | None:
    """Why a fix left the unit worse than it found it, or ``None``.

    Three signals, each learned from a different way a repair went wrong:

    1. ``delete_only``: it deleted and wrote nothing. The finding is gone
       because the data is gone, which is the outcome the repair contract
       exists to forbid.
    2. ``no_progress``: it shrank the product without resolving anything.
       Counting findings alone cannot see this -- deleting the flagged
       statement drops the count, so the dominant failure mode scored as a
       success.
    3. ``new_mandatory``: it *created* mandatory findings. A fix that
       manufactures new defects is strictly worse than no fix, however much
       else it changed.

    Judged per fix against the running baseline, so one bad correction is
    undone alone instead of taking the whole critique with it.
    """
    wrote_nothing = not (graph_after - graph_before)
    deleted_something = bool(graph_before - graph_after)
    if deleted_something and wrote_nothing:
        return "delete_only"
    if mandatory_after > mandatory_before:
        return "new_mandatory"
    if product_after < product_before and mandatory_after >= mandatory_before:
        return "no_progress"
    return None


def _patch_regressed(
    *,
    graph_before: RDFGraph,
    graph_after: RDFGraph,
    product_before: int,
    product_after: int,
    mandatory_before: int,
    mandatory_after: int,
) -> bool:
    """Whether a fix left the unit worse; see :func:`_regression_reason`."""
    return (
        _regression_reason(
            graph_before=graph_before,
            graph_after=graph_after,
            product_before=product_before,
            product_after=product_after,
            mandatory_before=mandatory_before,
            mandatory_after=mandatory_after,
        )
        is not None
    )


@dataclass
class _PatchRun:
    """What applying a list of per-fix patches one at a time did."""

    applied: list[FixPatch] = field(default_factory=list)
    rolled_back: list[RolledBackFix] = field(default_factory=list)
    deleted: int = 0
    inserted: int = 0
    #: Findings against the graph as the run left it.
    findings: list = field(default_factory=list)
    mandatory_after: int = 0


def _timed_findings(
    unit_state: UnitStateT, atomic: AtomicToolBox, phase: "LoopPhase"
) -> list:
    """The phase's findings, charged to the deterministic repair budget.

    Applying fixes one at a time runs the validator once per fix instead of
    once per pass; the walk is LLM-free but not free, so it is timed where
    the other deterministic repairs are.
    """
    started = time.perf_counter()
    findings = phase.collect_findings(unit_state, atomic)
    unit_state.budget_tracker.add_duration(
        "repair/deterministic", time.perf_counter() - started
    )
    return findings


def _apply_patches(
    unit_state: UnitStateT,
    atomic: AtomicToolBox,
    phase: "LoopPhase",
    patches: list[FixPatch],
    *,
    mandatory_before: int,
) -> _PatchRun:
    """Apply patches one at a time, undoing each that leaves the unit worse.

    The baseline a fix is judged against is the graph as the previous kept
    fix left it, so a fix that resolves a finding lowers the bar for the ones
    after it and a fix that manufactures one is caught on its own. A whole
    pass used to be undone for the one fix that regressed, which is how a
    critique lost its good corrections to one bad one.
    """
    run = _PatchRun()
    baseline = mandatory_before
    current_findings: list | None = None
    for patch in patches:
        token = unit_state.snapshot_for_rollback()
        graph_before = unit_state.patch_target_graph().copy()
        product_before = unit_state.product_triple_count()
        if not unit_state.apply_patch(patch.update):
            logger.warning("Critic fix refused by the phase; graph unchanged")
            run.rolled_back.append(
                RolledBackFix(
                    triple_ids=list(patch.fix.triple_ids),
                    correct_value=patch.fix.correct_value or "",
                    reason="refused",
                )
            )
            continue
        findings = _timed_findings(unit_state, atomic, phase)
        mandatory_after = sum(1 for finding in findings if finding.mandatory)
        reason = _regression_reason(
            graph_before=graph_before,
            graph_after=unit_state.patch_target_graph(),
            product_before=product_before,
            product_after=unit_state.product_triple_count(),
            mandatory_before=baseline,
            mandatory_after=mandatory_after,
        )
        if reason is not None:
            logger.warning(
                "Critic fix left the unit worse (%s; -%d/+%d triples, "
                "mandatory %d -> %d) — rolling it back",
                reason,
                patch.deletes,
                patch.inserts,
                baseline,
                mandatory_after,
            )
            unit_state.restore(token)
            run.rolled_back.append(
                RolledBackFix(
                    triple_ids=list(patch.fix.triple_ids),
                    correct_value=patch.fix.correct_value or "",
                    reason=reason,
                    mandatory_delta=mandatory_after - baseline,
                )
            )
            continue
        run.applied.append(patch)
        run.deleted += patch.deletes
        run.inserted += patch.inserts
        baseline = mandatory_after
        current_findings = findings
    if current_findings is None:
        # Nothing stayed, so the graph is as it was; the findings are still
        # collected here because the caller records and re-evaluates on them.
        current_findings = _timed_findings(unit_state, atomic, phase)
        baseline = sum(1 for finding in current_findings if finding.mandatory)
    run.findings = current_findings
    run.mandatory_after = baseline
    return run


def _reevaluate_unit_status(
    unit_state: UnitStateT,
    phase: "LoopPhase",
    atomic: AtomicToolBox,
    findings: list,
) -> None:
    """Re-run acceptance on the unit's *current* graph and findings.

    The critic decides acceptance before its patch is applied, so a unit
    whose patch then resolved every defect still carried ``FAILED`` out of
    the loop -- and was counted as "salvaged from a non-converged loop" by
    the reduce, overstating non-convergence. Acceptance is re-derived here
    from the post-patch findings and the still-outstanding critic fixes,
    with the same ``material_defects`` rule the critic itself used.
    """
    defects = material_defects(
        findings,
        unit_state.suggestions.actionable_fixes,
        phase.acceptance_policy(atomic),
    )
    if not defects:
        if unit_state.status != Status.SUCCESS:
            logger.info("%s unit clean after critic patch; marking SUCCESS", phase.name)
        unit_state.status = Status.SUCCESS
        unit_state.clear_failure()
    else:
        unit_state.set_failure(
            phase.critic_stage,
            f"{phase.name} unit has {len(defects)} material defect(s) "
            "after critic patch",
        )


def _apply_critic_patch(
    unit_state: UnitStateT,
    atomic: AtomicToolBox,
    phase: "LoopPhase",
    *,
    render_attempt: int,
    pass_index: int,
    mandatory_before: int,
) -> PatchOutcome:
    """Compile the critique into per-fix patches and apply them one at a time.

    This is where a critique stops being a description and becomes a change.
    The critic cites statement ids, so the delete side resolves by lookup rather
    than by matching text the model retyped from memory -- which is what made
    the previous contract lose most of its own removals. Screening then withholds
    what the deployment does not allow a pass to destroy, and each fix that
    survives is applied on its own and undone on its own if it leaves the unit
    worse.
    """
    graph = unit_state.patch_target_graph()
    compiled = compile_critic_fixes(
        unit_state.suggestions.actionable_fixes,
        graph,
        index=unit_state.prompt_triple_index,
        policy=phase.patch_policy(atomic),
    )
    # An index is paired with the critique it was built for. Clearing it here
    # means a later compile reaching for a previous pass's table resolves
    # nothing rather than resolving to the wrong statement.
    unit_state.prompt_triple_index = None

    unit_state.critic_fixes_residual = len(compiled.residual)
    unit_state.critic_fixes_noop += len(compiled.noop)
    unit_state.critic_fixes_junk_refused += compiled.junk_refused
    unit_state.critic_fixes_unresolved_prefix += compiled.unresolved_prefix
    # An applied fix must stop existing as a request: `suggestions` is what the
    # next pass sees as outstanding work, and a fix already carried out would
    # be asked for again against a graph that no longer matches it. A rolled
    # back fix is not re-requested either: the next pass would repeat it.
    unit_state.suggestions = Suggestions(
        actionable_fixes=list(compiled.residual),
        systemic_critique_summary=unit_state.suggestions.systemic_critique_summary,
    )

    run = _apply_patches(
        unit_state, atomic, phase, compiled.patches, mandatory_before=mandatory_before
    )
    unit_state.critic_fixes_applied += len(run.applied)
    unit_state.critic_fixes_rolled_back += len(run.rolled_back)
    unit_state.deterministic_findings = run.findings
    _reevaluate_unit_status(unit_state, phase, atomic, run.findings)
    if isinstance(unit_state, UnitFactsState):
        unit_state.applied_repairs.extend(
            GraphRepairRecord(
                kind=FactsUnitFindingKind.CRITIC_FIX,
                source=f"ids={patch.fix.triple_ids}" if patch.fix.triple_ids else "",
                target=patch.fix.correct_value or "",
            )
            for patch in run.applied
        )
    _record_attempt(
        unit_state,
        kind="critic_patch",
        render_attempt=render_attempt,
        critic_attempt=pass_index,
        n_findings=len(run.findings),
        n_mandatory=run.mandatory_after,
        n_fixes_applied=len(run.applied),
        n_fixes_noop=len(compiled.noop),
        n_triples_deleted=run.deleted,
        n_triples_inserted=run.inserted,
        rolled_back_fixes=run.rolled_back,
        patch_delete_capped=compiled.delete_capped,
        n_fixes_junk_refused=compiled.junk_refused,
        n_fixes_unresolved_prefix=compiled.unresolved_prefix,
    )
    return PatchOutcome(
        applied=len(run.applied),
        residual=len(compiled.residual),
        noop=len(compiled.noop),
        rolled_back=len(run.rolled_back),
        mandatory_after=run.mandatory_after,
    )


async def _run_completion_passes(
    unit_state: UnitFactsState,
    atomic: AtomicToolBox,
    phase: "LoopPhase",
    *,
    render_attempt: int,
) -> None:
    """Insert-only completion passes, run once the critic loop is done.

    Each pass asks :func:`ontocast.agent.complete_facts.complete_facts` for
    subjects recovering measurements the numeric inventory still lists as
    missing, then applies the proposal through :func:`_apply_patches` -- the
    same per-subject regression check a critic fix goes through, so an
    insert that leaves the unit worse is rolled back on its own. Stops early
    once the inventory is empty; the caller has already checked
    ``atomic.facts_completion_passes > 0`` and that this is the facts phase.
    """
    for pass_index in range(1, atomic.facts_completion_passes + 1):
        inventory = unit_numeric_inventory(
            graph=unit_state.content_unit.graph,
            ontology_graph=unit_state.ontology_snapshot.graph,
            extraction_text=unit_state.content_unit.extraction_text,
            policy=atomic.validation_policy,
            limit=atomic.numeric_coverage_limit,
        )
        if not inventory.measurements:
            break

        findings = phase.collect_findings(unit_state, atomic)
        unit_state.deterministic_findings = findings
        mandatory_before = sum(1 for finding in findings if finding.mandatory)

        fixes = await complete_facts(unit_state, atomic, inventory)
        compiled = compile_critic_fixes(
            fixes, unit_state.patch_target_graph(), policy=atomic.facts_patch_policy
        )
        run = _apply_patches(
            unit_state,
            atomic,
            phase,
            compiled.patches,
            mandatory_before=mandatory_before,
        )
        unit_state.deterministic_findings = run.findings
        _reevaluate_unit_status(unit_state, phase, atomic, run.findings)

        post_inventory = unit_numeric_inventory(
            graph=unit_state.content_unit.graph,
            ontology_graph=unit_state.ontology_snapshot.graph,
            extraction_text=unit_state.content_unit.extraction_text,
            policy=atomic.validation_policy,
            limit=atomic.numeric_coverage_limit,
        )
        recovered = max(
            0, len(inventory.measurements) - len(post_inventory.measurements)
        )
        logger.info(
            "Unit facts completion pass %s/%s: %d applied, %d rolled back, "
            "%d measurement(s) recovered",
            pass_index,
            atomic.facts_completion_passes,
            len(run.applied),
            len(run.rolled_back),
            recovered,
        )
        _record_attempt(
            unit_state,
            kind="completion",
            render_attempt=render_attempt,
            critic_attempt=pass_index,
            n_findings=len(run.findings),
            n_mandatory=run.mandatory_after,
            n_fixes_applied=len(run.applied),
            n_fixes_noop=len(compiled.noop),
            n_triples_deleted=run.deleted,
            n_triples_inserted=run.inserted,
            rolled_back_fixes=run.rolled_back,
            patch_delete_capped=compiled.delete_capped,
            n_fixes_junk_refused=compiled.junk_refused,
            n_fixes_unresolved_prefix=compiled.unresolved_prefix,
            n_measurements_recovered=recovered,
        )
        if not post_inventory.measurements:
            break


def _reset_node_evidence_context(
    state: UnitFactsState | UnitOntologyState, node: WorkflowNode
) -> None:
    """Start node execution in no-search mode with empty evidence context."""
    state.set_external_evidence_request(node, ExternalEvidenceRequest())
    state.set_external_evidence_cache_entry(node, ExternalEvidenceCacheEntry())
    state.load_external_evidence_for_node(node)


def _apply_unit_ontology_context(
    unit_state: UnitFactsState | UnitOntologyState,
    ctx: UnitOntologyContext,
) -> None:
    """Point unit state at the assembled context (snapshot + writable + sources).

    The snapshot is shared by reference, not copied. Every consumer in both unit
    loops treats it as read-only schema -- the facts loop mutates only the
    rendered facts graph, and the ontology loop edits ``working_graph``, keeping
    the snapshot as its pristine baseline for ``working_graph_changed()``.
    Deep-copying it per unit cost a full rdflib graph copy each time, on the
    event loop, for a value that is identical across the whole fan-out.
    """
    unit_state.ontology_snapshot = ctx.snapshot
    unit_state.ontology_patch_sources = list(ctx.patch_sources)
    unit_state.writable_iris = list(ctx.writable_iris)
    unit_state.assembly_anchor_iri = ctx.primary_writable_iri
    unit_state.assembly_mode_used = ctx.assembly_mode


async def _apply_facts_ontology_context(
    unit_state: UnitFactsState,
    context: UnitLoopContext,
    tools: ToolBox,
) -> None:
    """Set ontology_snapshot for facts from the per-unit context resolver.

    Only reached when the caller has no merged document context to hand down
    (single-unit pipelines, or facts-only runs with no ontology stage).
    """
    ctx = await resolve_unit_ontology_context(context, tools, unit_state.content_unit)
    logger.info(
        "Ontology context for mode %s: sources=%s writable=%s",
        context.ontology_context_mode,
        ctx.patch_sources,
        ctx.writable_iris,
    )
    _apply_unit_ontology_context(unit_state, ctx)


def _select_conformance_chapter(unit_state: UnitFactsState, tools: ToolBox) -> None:
    """Join the shapes contract on this unit's resolved ontology context.

    Runs once per unit, right after context resolution: the fan-out could
    not select earlier because the snapshot did not exist yet. The join key
    is every IRI of the snapshot graph -- subjects, predicates and objects,
    because the schema closure carries superclass IRIs as objects, which is
    how a shape targeting a superclass reaches a unit typed with the
    subclass -- plus the writable IRIs. An empty snapshot selects nothing:
    a unit with no ontology context has no classes to hold to their rules.
    """
    context_terms: set[str] = set(unit_state.writable_iris or ())
    for triple in unit_state.ontology_snapshot.graph:
        for term in triple:
            if isinstance(term, URIRef):
                context_terms.add(str(term))
    unit_state.conformance_chapter = tools.shapes_chapter_for_context(context_terms)
    unit_state.conformance_selection_pending = False
    logger.debug(
        "Conformance chapter selected for unit %s: %d chars from %d context terms",
        unit_state.content_unit.index,
        len(unit_state.conformance_chapter),
        len(context_terms),
    )


# --- phase adapters ----------------------------------------------------------
#
# The two unit loops were parallel implementations of the same control flow,
# and they had drifted: only one recorded render attempts, only one stopped
# escalating a rejection into a fresh extraction, and only one kept the
# critique it had just paid for. Rather than fix each divergence twice, the
# flow is written once and the genuine differences -- which agent, which graph,
# which validator, which budget -- are declared here.
#
# The callables are module-level wrappers rather than direct references on
# purpose: a frozen adapter capturing ``render_facts`` at import time would
# defeat every test that monkeypatches this module's globals, and those tests
# would go on passing while silently exercising the real agents.


def _same_state(returned: object, given: object) -> None:
    """The agents mutate the state they are handed and return it.

    The driver relies on that: it keeps its own reference so the loop stays
    generic over the two unit types rather than widening to their union at every
    call. If an agent ever starts returning a copy, the loop would carry on with
    the stale object and quietly lose that step's work -- so it fails here
    instead.
    """
    assert returned is given, "unit-loop agents must mutate the state in place"


async def _render_facts_phase(state, atomic, supplemental) -> None:
    _same_state(
        await render_facts(state, atomic, supplemental_ontologies=supplemental), state
    )


async def _render_ontology_phase(state, atomic, supplemental) -> None:
    _same_state(
        await render_ontology(state, atomic, supplemental_ontologies=supplemental),
        state,
    )


async def _criticise_facts_phase(state, atomic) -> None:
    _same_state(await criticise_facts(state, atomic), state)


async def _criticise_ontology_phase(state, atomic) -> None:
    _same_state(await criticise_ontology(state, atomic), state)


def _prepare_facts(unit_state: UnitFactsState) -> None:
    """Nothing to stage: the facts graph is the unit's product in place."""


def _facts_critic_skip_reason(
    unit_state: UnitFactsState, atomic: AtomicToolBox
) -> str | None:
    """Why the facts critic should not be spent on this unit, or ``None``.

    A citation-metadata unit is rendered under the bibliographic instruction
    and has no domain facts to review. A render below the triple floor is,
    at the default, an empty graph: the critic scores one perfect and bills
    a call for it.
    """
    if unit_state.content_unit.is_citation_metadata:
        return "citation_metadata"
    if len(unit_state.content_unit.graph) < atomic.facts_critic_min_triples:
        return "empty_render"
    return None


def _ontology_critic_skip_reason(
    unit_state: UnitOntologyState, atomic: AtomicToolBox
) -> str | None:
    """The ontology critic has no skip rule; its budget alone decides."""
    return None


def _critic_unavailable(unit_state: UnitFactsState | UnitOntologyState) -> bool:
    return (
        isinstance(unit_state, UnitFactsState)
        and unit_state.critic_outcome == "unavailable"
    )


def _prepare_ontology(unit_state: UnitOntologyState) -> None:
    """Copy the snapshot into the scratchpad the loop renders against."""
    started = time.perf_counter()
    unit_state.working_graph = unit_state.ontology_snapshot.graph.copy()
    unit_state.budget_tracker.add_duration(
        "ctx/working_graph_copy", time.perf_counter() - started
    )


@dataclass(frozen=True)
class LoopPhase:
    """Everything that differs between the facts and ontology unit loops."""

    name: Literal["facts", "ontology"]
    render_node: WorkflowNode
    critic_node: WorkflowNode
    render_stage: FailureStage
    critic_stage: FailureStage
    # The two agent hooks are side-effecting: they mutate the state the driver
    # holds. That is what the agents already do, and saying so here is what lets
    # the driver stay generic over the two unit types instead of widening to
    # their union at every call. `_same_state` keeps the assumption honest.
    prepare: Callable[..., None]
    render: Callable[..., Awaitable[None]]
    criticise: Callable[..., Awaitable[None]]
    collect_findings: Callable[..., list]
    critic_passes: Callable[[AtomicToolBox], int]
    patch_policy: Callable[[AtomicToolBox], CriticPatchPolicy]
    acceptance_policy: Callable[[AtomicToolBox], FactsAcceptancePolicy]
    #: Why a critic pass should not be spent on the unit, or ``None`` to run
    #: it. Decided per pass, on the current graph.
    critic_skip_reason: Callable[..., str | None]


FACTS_PHASE = LoopPhase(
    name="facts",
    render_node=WorkflowNode.TEXT_TO_FACTS,
    critic_node=WorkflowNode.CRITICISE_FACTS,
    render_stage=FailureStage.GENERATE_GRAPH_UPDATE_FOR_FACTS,
    critic_stage=FailureStage.FACTS_CRITIQUE,
    prepare=_prepare_facts,
    render=_render_facts_phase,
    criticise=_criticise_facts_phase,
    collect_findings=_collect_facts_findings,
    critic_passes=lambda atomic: atomic.facts_critic_passes,
    patch_policy=lambda atomic: atomic.facts_patch_policy,
    acceptance_policy=lambda atomic: atomic.acceptance_policy,
    critic_skip_reason=_facts_critic_skip_reason,
)

ONTOLOGY_PHASE = LoopPhase(
    name="ontology",
    render_node=WorkflowNode.TEXT_TO_ONTOLOGY,
    critic_node=WorkflowNode.CRITICISE_ONTOLOGY,
    render_stage=FailureStage.GENERATE_GRAPH_UPDATE_FOR_ONTOLOGY,
    critic_stage=FailureStage.ONTOLOGY_CRITIQUE,
    prepare=_prepare_ontology,
    render=_render_ontology_phase,
    criticise=_criticise_ontology_phase,
    collect_findings=_collect_ontology_findings,
    critic_passes=lambda atomic: atomic.ontology_critic_passes,
    patch_policy=lambda atomic: atomic.ontology_patch_policy,
    acceptance_policy=lambda atomic: atomic.ontology_acceptance_policy,
    critic_skip_reason=_ontology_critic_skip_reason,
)


async def _render_with_evidence(
    unit_state: UnitStateT,
    atomic: AtomicToolBox,
    phase: LoopPhase,
    supplemental: list[Ontology],
    *,
    render_attempt: int,
) -> bool:
    """One render attempt, with the evidence-backed retry. True on success."""
    unit_state.node_visits[phase.render_node] += 1
    _reset_node_evidence_context(unit_state, phase.render_node)
    await phase.render(unit_state, atomic, supplemental)
    _record_attempt(unit_state, kind="render", render_attempt=render_attempt)
    if unit_state.status == Status.SUCCESS:
        return True

    request = unit_state.get_external_evidence_request(phase.render_node)
    if not request.initiate_search:
        logger.info(
            "Unit %s render failed at attempt %s (no search request)",
            phase.name,
            render_attempt,
        )
        return False

    await plan_external_evidence_for_node(unit_state, atomic, phase.render_node)
    await fetch_external_evidence_for_node(unit_state, atomic, phase.render_node)
    await phase.render(unit_state, atomic, supplemental)
    _record_attempt(unit_state, kind="render", render_attempt=render_attempt)
    logger.info(
        "Unit %s render %s at attempt %s (with search)",
        phase.name,
        "recovered" if unit_state.status == Status.SUCCESS else "failed",
        render_attempt,
    )
    return unit_state.status == Status.SUCCESS


async def run_unit_loop(
    state: UnitStateT,
    tools: ToolBox,
    document_context: UnitLoopContext,
    phase: LoopPhase,
    max_visits_per_node: int | None = None,
    pre_resolved_context: UnitOntologyContext | None = None,
) -> UnitStateT:
    """Extract a unit once, then review and patch it a bounded number of times.

    The two budgets mean different things and no longer trade against each
    other. ``MAX_VISITS`` retries a render that *failed*; a render that
    succeeded is never repeated, because re-extracting a whole unit is the
    expensive answer to a local defect and reliably introduces new ones. The
    critic passes then improve what came back, each one re-running the
    deterministic checks for free before paying for the critique.

    Args:
        state: Unit state to run the loop over.
        tools: Tool container.
        document_context: Document-level inputs, shared read-only.
        phase: Which of the two loops this is.
        max_visits_per_node: Override for the render-failure bound.
        pre_resolved_context: Ontology context resolved once by the caller. The
            merged document ontology depends only on document-level state, so
            the fan-out builds it once and hands the *same object* to every
            unit; resolving it per unit cost a full rdflib merge and two graph
            copies each time.
    """
    atomic = tools.get_atomic_tools()
    unit_state = state.model_copy(deep=True)
    # Charge resolver LLM calls (e.g. ontology selection) to this unit's
    # tracker -- the copy that survives the loop and is merged by the caller.
    # Shallow copy: retrieval_metrics stays shared with the caller's context.
    document_context = document_context.model_copy(
        update={"budget_tracker": unit_state.budget_tracker}
    )
    # The stage the loop is currently in, so an unhandled exception is
    # attributed to where it happened rather than always naming the critique.
    stage = phase.render_stage
    try:
        if pre_resolved_context is not None:
            _apply_unit_ontology_context(unit_state, pre_resolved_context)
        elif isinstance(unit_state, UnitFactsState):
            await _apply_facts_ontology_context(unit_state, document_context, tools)
        else:
            # The ontology loop is the caller that can answer an empty context
            # by inventing vocabulary: ``render_ontology`` branches on an empty
            # seed into ``render_ontology_fresh``.
            resolved = await resolve_unit_ontology_context(
                document_context,
                tools,
                unit_state.content_unit,
                can_create_vocabulary=True,
            )
            _apply_unit_ontology_context(unit_state, resolved)
        if (
            isinstance(unit_state, UnitFactsState)
            and unit_state.conformance_selection_pending
        ):
            _select_conformance_chapter(unit_state, tools)
        phase.prepare(unit_state)

        max_visits = _resolve_max_visits_limit(
            unit_state.max_visits_per_node, max_visits_per_node
        )
        unit_state.max_visits_per_node = max_visits

        rendered = False
        render_attempt = 0
        supplemental: list[Ontology] = []
        for render_attempt in range(1, max_visits + 1):
            stage = phase.render_stage
            supplemental = _supplemental_ontologies_for_unit(
                document_context, unit_state, tools
            )
            rendered = await _render_with_evidence(
                unit_state,
                atomic,
                phase,
                supplemental,
                render_attempt=render_attempt,
            )
            if rendered:
                break

        if not rendered:
            logger.info("Unit %s loop exhausted render retries", phase.name)
            unit_state.deterministic_findings = phase.collect_findings(
                unit_state, atomic
            )
            return unit_state

        stage = phase.critic_stage
        for pass_index in range(1, phase.critic_passes(atomic) + 1):
            findings = phase.collect_findings(unit_state, atomic)
            unit_state.deterministic_findings = findings
            mandatory_before = sum(1 for finding in findings if finding.mandatory)

            skip_reason = phase.critic_skip_reason(unit_state, atomic)
            if skip_reason is not None:
                # No call is billed, and the record says why: an empty render
                # reviewed by the critic scores perfect for nothing, and a
                # citation-metadata unit has no domain facts to review.
                logger.info(
                    "Unit %s critic pass %s skipped (%s)",
                    phase.name,
                    pass_index,
                    skip_reason,
                )
                if isinstance(unit_state, UnitFactsState):
                    unit_state.critic_outcome = "skipped"
                _record_attempt(
                    unit_state,
                    kind="critic_skipped",
                    render_attempt=render_attempt,
                    critic_attempt=pass_index,
                    n_findings=len(findings),
                    n_mandatory=mandatory_before,
                    accept_reason=skip_reason,
                )
                break

            unit_state.node_visits[phase.critic_node] += 1
            _reset_node_evidence_context(unit_state, phase.critic_node)
            await phase.criticise(unit_state, atomic)
            if unit_state.status != Status.SUCCESS and not _critic_unavailable(
                unit_state
            ):
                request = unit_state.get_external_evidence_request(phase.critic_node)
                if request.initiate_search:
                    await plan_external_evidence_for_node(
                        unit_state, atomic, phase.critic_node
                    )
                    await fetch_external_evidence_for_node(
                        unit_state, atomic, phase.critic_node
                    )
                    await phase.criticise(unit_state, atomic)
            if _critic_unavailable(unit_state):
                # The critic produced no critique, so there is nothing to
                # compile and nothing to re-evaluate acceptance from. The
                # render is kept as it was and the unit leaves FAILED at the
                # critique stage: unreviewed, not accepted. Applying the
                # previous pass's suggestions here would patch against a
                # critique of a graph that has since changed.
                logger.warning(
                    "Unit %s critic unavailable on pass %s; render kept unreviewed",
                    phase.name,
                    pass_index,
                )
                break

            outcome = _apply_critic_patch(
                unit_state,
                atomic,
                phase,
                render_attempt=render_attempt,
                pass_index=pass_index,
                mandatory_before=mandatory_before,
            )
            logger.info(
                "Unit %s critic pass %s/%s: %d applied, %d residual, %d no-op, "
                "%d rolled back",
                phase.name,
                pass_index,
                phase.critic_passes(atomic),
                outcome.applied,
                outcome.residual,
                outcome.noop,
                outcome.rolled_back,
            )
            if outcome.converged:
                break
        else:
            if phase.critic_passes(atomic) == 0:
                # No pass ran, so nothing collected the residual the document
                # metric sums. Without this the denominator silently counts
                # only units that happened to be criticised.
                unit_state.deterministic_findings = phase.collect_findings(
                    unit_state, atomic
                )

        if (
            phase.name == "facts"
            and atomic.facts_completion_passes > 0
            and isinstance(unit_state, UnitFactsState)
        ):
            await _run_completion_passes(
                unit_state, atomic, phase, render_attempt=render_attempt
            )

        mandatory = sum(
            1 for finding in unit_state.deterministic_findings if finding.mandatory
        )
        if mandatory:
            logger.warning(
                "%d mandatory deterministic finding(s) remain unresolved", mandatory
            )
        return unit_state
    except (OntologyContextConfigError, LLMConfigurationError):
        # Not a unit failure. Both describe the deployment, not the unit, and
        # are identical for every other unit in the document: an unresolvable
        # ontology context (the whole point of ONTOLOGY_CONTEXT_REQUIRED is
        # that the run stops rather than emitting an ungrounded graph), or a
        # request the provider refuses outright. Recording either per unit
        # instead let the fan-out finish, write a zero-triple manifest and exit
        # successfully -- the vacuous pass these errors exist to prevent, now
        # with a traceback per unit to bury the cause.
        logger.error(
            "Deployment fault while running %s units; stopping the run",
            phase.name,
        )
        raise
    except Exception as exc:
        logger.exception("Unhandled exception in %s unit loop", phase.name)
        unit_state.set_failure(stage, str(exc))
        return unit_state


async def facts_loop(
    state: UnitFactsState,
    tools: ToolBox,
    document_context: UnitLoopContext,
    max_visits_per_node: int | None = None,
    pre_resolved_context: UnitOntologyContext | None = None,
) -> UnitFactsState:
    """Run the render/critic loop for one content unit's facts."""
    return await run_unit_loop(
        state,
        tools,
        document_context,
        FACTS_PHASE,
        max_visits_per_node=max_visits_per_node,
        pre_resolved_context=pre_resolved_context,
    )


async def ontology_loop(
    state: UnitOntologyState,
    tools: ToolBox,
    document_context: UnitLoopContext,
    max_visits_per_node: int | None = None,
) -> UnitOntologyState:
    """Run the render/critic loop for one content unit's ontology delta."""
    return await run_unit_loop(
        state,
        tools,
        document_context,
        ONTOLOGY_PHASE,
        max_visits_per_node=max_visits_per_node,
    )
