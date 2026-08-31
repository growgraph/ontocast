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
from dataclasses import dataclass
from typing import Literal, TypeVar

from rdflib import URIRef

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
)
from ontocast.tool.facts_validation.critic_patch import (
    CriticPatchPolicy,
    compile_critic_fixes,
)
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
    coverage_mandatory = (
        atomic.numeric_coverage_mandatory if atomic is not None else False
    )
    return collect_unit_findings(
        graph=unit_state.content_unit.graph,
        ontology_graph=unit_state.ontology_snapshot.graph,
        quarantined=unit_state.quarantined_literal_triples,
        extraction_text=unit_state.content_unit.extraction_text,
        fact_namespaces=[DEFAULT_IRI, str(unit_state.content_unit.doc_iri)],
        # Citation numerics (pages, years, volume numbers) are not extractable
        # quantities — never push coverage repair on bibliography units.
        coverage_limit=(
            0 if unit_state.content_unit.is_citation_metadata else coverage_limit
        ),
        coverage_mandatory=coverage_mandatory,
        policy=policy,
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
    kind: Literal["render", "critic", "critic_patch", "llm_repair"],
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
) -> None:
    """Append one telemetry record for the current loop attempt.

    ``triple_count`` is the phase's own product measure, so an ontology record
    reports what the unit contributes rather than the size of its scratchpad --
    the working graph is the snapshot plus a small delta and barely moves.
    """
    unit_state.attempt_log.append(
        LoopAttempt(
            render_attempt=render_attempt,
            critic_attempt=critic_attempt,
            kind=kind,
            success=unit_state.status == Status.SUCCESS,
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
            patch_rolled_back=patch_rolled_back,
            patch_delete_capped=patch_delete_capped,
        )
    )


@dataclass(frozen=True)
class PatchOutcome:
    """What one critic pass changed, and whether the loop should run another."""

    applied: int = 0
    residual: int = 0
    noop: int = 0
    rolled_back: bool = False
    mandatory_after: int = 0

    @property
    def converged(self) -> bool:
        """True when another pass has nothing left to work with.

        A rolled-back pass counts as converged: it was handed the same graph and
        the same findings the next one would see, so repeating it buys a second
        identical answer at full price.
        """
        return self.rolled_back or (self.applied == 0 and self.mandatory_after == 0)


def _patch_regressed(
    *,
    graph_before: RDFGraph,
    graph_after: RDFGraph,
    product_before: int,
    product_after: int,
    mandatory_before: int,
    mandatory_after: int,
) -> bool:
    """Whether a pass left the unit worse than it found it.

    Three signals, each learned from a different way a repair went wrong:

    1. It deleted and wrote nothing. The finding is gone because the data is
       gone, which is the outcome the repair contract exists to forbid.
    2. It shrank the product without resolving anything. Counting findings alone
       cannot see this -- deleting the flagged statement drops the count, so the
       dominant failure mode scored as a success.
    3. It *created* mandatory findings. A pass that manufactures new defects is
       strictly worse than no pass, however much else it fixed.
    """
    wrote_nothing = not (graph_after - graph_before)
    deleted_something = bool(graph_before - graph_after)
    no_progress = product_after < product_before and mandatory_after >= mandatory_before
    return (
        (deleted_something and wrote_nothing)
        or no_progress
        or mandatory_after > mandatory_before
    )


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
    """Compile the critique into a patch and apply it, or undo it.

    This is where a critique stops being a description and becomes a change.
    The critic cites statement ids, so the delete side resolves by lookup rather
    than by matching text the model retyped from memory -- which is what made
    the previous contract lose most of its own removals. Screening then withhelds
    what the deployment does not allow a pass to destroy, and the rollback below
    undoes a pass that made things worse.
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

    unit_state.critic_fixes_applied += len(compiled.applied)
    unit_state.critic_fixes_residual = len(compiled.residual)
    unit_state.critic_fixes_noop += len(compiled.noop)
    # An applied fix must stop existing as a request: `suggestions` is what the
    # next pass sees as outstanding work, and a fix already carried out would
    # be asked for again against a graph that no longer matches it.
    unit_state.suggestions = Suggestions(
        actionable_fixes=list(compiled.residual),
        systemic_critique_summary=unit_state.suggestions.systemic_critique_summary,
    )

    if compiled.update is None:
        findings = phase.collect_findings(unit_state, atomic)
        unit_state.deterministic_findings = findings
        mandatory_after = sum(1 for finding in findings if finding.mandatory)
        _reevaluate_unit_status(unit_state, phase, atomic, findings)
        _record_attempt(
            unit_state,
            kind="critic_patch",
            render_attempt=render_attempt,
            critic_attempt=pass_index,
            n_findings=len(findings),
            n_mandatory=mandatory_after,
            n_fixes_noop=len(compiled.noop),
        )
        return PatchOutcome(
            residual=len(compiled.residual),
            noop=len(compiled.noop),
            mandatory_after=mandatory_after,
        )

    token = unit_state.snapshot_for_rollback()
    graph_before = graph.copy()
    product_before = unit_state.product_triple_count()
    deleted = sum(
        len(op.graph) for op in compiled.update.triple_operations if op.type == "delete"
    )
    inserted = sum(
        len(op.graph) for op in compiled.update.triple_operations if op.type == "insert"
    )

    applied_ok = unit_state.apply_patch(compiled.update)
    findings = phase.collect_findings(unit_state, atomic)
    mandatory_after = sum(1 for finding in findings if finding.mandatory)
    rolled_back = False
    if applied_ok and _patch_regressed(
        graph_before=graph_before,
        graph_after=unit_state.patch_target_graph(),
        product_before=product_before,
        product_after=unit_state.product_triple_count(),
        mandatory_before=mandatory_before,
        mandatory_after=mandatory_after,
    ):
        logger.warning(
            "Critic patch left the unit worse (-%d/+%d triples, mandatory %d -> %d)"
            " — rolling it back",
            deleted,
            inserted,
            mandatory_before,
            mandatory_after,
        )
        unit_state.restore(token)
        findings = phase.collect_findings(unit_state, atomic)
        mandatory_after = sum(1 for finding in findings if finding.mandatory)
        rolled_back = True
    elif not applied_ok:
        logger.warning("Critic patch refused by the phase; graph unchanged")
        rolled_back = True

    unit_state.deterministic_findings = findings
    _reevaluate_unit_status(unit_state, phase, atomic, findings)
    if not rolled_back and isinstance(unit_state, UnitFactsState):
        unit_state.applied_repairs.extend(
            GraphRepairRecord(
                kind=FactsUnitFindingKind.CRITIC_FIX,
                source=f"ids={fix.triple_ids}" if fix.triple_ids else "",
                target=fix.correct_value or "",
            )
            for fix in compiled.applied
        )
    _record_attempt(
        unit_state,
        kind="critic_patch",
        render_attempt=render_attempt,
        critic_attempt=pass_index,
        n_findings=len(findings),
        n_mandatory=mandatory_after,
        n_fixes_applied=0 if rolled_back else len(compiled.applied),
        n_fixes_noop=len(compiled.noop),
        n_triples_deleted=0 if rolled_back else deleted,
        n_triples_inserted=0 if rolled_back else inserted,
        patch_rolled_back=rolled_back,
        patch_delete_capped=compiled.delete_capped,
    )
    if rolled_back:
        unit_state.critic_fixes_applied -= len(compiled.applied)
    return PatchOutcome(
        applied=0 if rolled_back else len(compiled.applied),
        residual=len(compiled.residual),
        noop=len(compiled.noop),
        rolled_back=rolled_back,
        mandatory_after=mandatory_after,
    )


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

            unit_state.node_visits[phase.critic_node] += 1
            _reset_node_evidence_context(unit_state, phase.critic_node)
            await phase.criticise(unit_state, atomic)
            if unit_state.status != Status.SUCCESS:
                request = unit_state.get_external_evidence_request(phase.critic_node)
                if request.initiate_search:
                    await plan_external_evidence_for_node(
                        unit_state, atomic, phase.critic_node
                    )
                    await fetch_external_evidence_for_node(
                        unit_state, atomic, phase.critic_node
                    )
                    await phase.criticise(unit_state, atomic)

            outcome = _apply_critic_patch(
                unit_state,
                atomic,
                phase,
                render_attempt=render_attempt,
                pass_index=pass_index,
                mandatory_before=mandatory_before,
            )
            logger.info(
                "Unit %s critic pass %s/%s: %d applied, %d residual, %d no-op%s",
                phase.name,
                pass_index,
                phase.critic_passes(atomic),
                outcome.applied,
                outcome.residual,
                outcome.noop,
                " (rolled back)" if outcome.rolled_back else "",
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

        mandatory = sum(
            1 for finding in unit_state.deterministic_findings if finding.mandatory
        )
        if mandatory:
            logger.warning(
                "%d mandatory deterministic finding(s) remain unresolved", mandatory
            )
        return unit_state
    except OntologyContextConfigError:
        # Not a unit failure. An unresolvable ontology context is a property of
        # the deployment, identical for every other unit in the document, and
        # the whole point of ONTOLOGY_CONTEXT_REQUIRED is that the run stops
        # rather than emitting an ungrounded graph. Recording it per unit
        # instead let the fan-out finish, write a zero-triple manifest and exit
        # successfully -- the vacuous pass this error exists to prevent, now
        # with a traceback per unit to bury the cause.
        logger.error(
            "Ontology context is unusable for %s units; stopping the run",
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
