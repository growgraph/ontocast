"""Dedicated state models for parallel unit loops."""

import logging
from collections import defaultdict

from pydantic import Field

from ontocast.onto.constants import DEFAULT_DOMAIN
from ontocast.onto.content_unit import ContentUnit, SourceUnit
from ontocast.onto.enum import (
    FailureStage,
    LLMGraphFormat,
    OntologyAssemblyMode,
    Status,
    WorkflowNode,
)
from ontocast.onto.model import (
    BasePydanticModel,
    ExternalEvidenceCacheEntry,
    ExternalEvidenceHit,
    ExternalEvidencePlan,
    ExternalEvidenceRequest,
    FactsUnitFinding,
    GraphRepairRecord,
    LoopAttempt,
    OntologyUnitFinding,
    Suggestions,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_apply import OntologyDelta
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState, BudgetTracker

logger = logging.getLogger(__name__)


def _render_updated_graph(
    graph: RDFGraph, updates: list[GraphUpdate], max_triples: int | None = None
) -> tuple[RDFGraph, bool]:
    """Apply GraphUpdate objects to a graph. Delegates to AgentState implementation."""
    return AgentState.render_updated_graph(graph, updates, max_triples=max_triples)


class UnitState(BasePydanticModel):
    """Common per-unit workflow state.

    ``content_unit`` is typed to the :class:`SourceUnit` base here and narrowed
    to :class:`ContentUnit` by :class:`UnitFactsState`, which needs the mutable
    graph. Declaring it once keeps the progress string and the context-assembly
    fields below from being written twice.
    """

    content_unit: SourceUnit = Field(description="Unit under processing")
    assembly_anchor_iri: str = Field(
        default="",
        description="Primary writable IRI from context assembly (metrics / logging).",
    )
    assembly_mode_used: OntologyAssemblyMode = Field(
        default=OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
        description="How ontology_snapshot was assembled for this unit.",
    )
    ontology_snapshot: OntologySnapshot = Field(
        default_factory=OntologySnapshot.empty,
        description="Immutable ontology snapshot (prompt view, no catalog id).",
    )
    ontology_patch_sources: list[str] = Field(
        default_factory=list,
        description="Ontology IRIs that contributed to the snapshot context.",
    )
    writable_iris: list[str] = Field(
        default_factory=list,
        description="Catalog IRIs that apply() may update from this unit's deltas.",
    )
    suggestions: Suggestions = Field(default_factory=Suggestions)
    quarantined_literal_triples: list[RejectedLiteralTriple] = Field(
        default_factory=list,
        description=(
            "Triples the render withheld from the applied graph because their "
            "XSD typed literals were invalid. On the base state because both "
            "update agents share the hygiene that produces it."
        ),
    )
    budget_tracker: BudgetTracker = Field(default_factory=BudgetTracker)
    max_visits_per_node: int = Field(default=1, ge=1)
    #: Critic attempts allowed per render attempt. None couples it to
    #: ``max_visits_per_node``, which makes the worst case quadratic.
    max_critic_visits_per_node: int | None = Field(default=None, ge=1)
    llm_graph_format: LLMGraphFormat = Field(
        default=LLMGraphFormat.JSONLD,
        description=(
            "Format used by the LLM for emitting RDF graph payloads: "
            "'jsonld' (default) or 'turtle' (legacy)."
        ),
    )
    ontology_context_max_triples: int | None = Field(
        default=None,
        description=(
            "Triple budget for the ontology chapter in this unit's prompts. "
            "None disables condensing. Threaded from ServerConfig alongside "
            "llm_graph_format, because the agents that build chapters do not "
            "all hold a ToolBox."
        ),
    )

    attempt_log: list[LoopAttempt] = Field(
        default_factory=list,
        description="Per-attempt telemetry (render/critic/repair) for this unit.",
    )
    status: Status = Field(default=Status.NOT_VISITED)
    failure_stage: FailureStage | None = Field(default=None)
    failure_reason: str | None = Field(default=None)
    node_visits: dict[WorkflowNode, int] = Field(
        default_factory=lambda: defaultdict(int),
    )
    external_evidence_plan: ExternalEvidencePlan = Field(
        default_factory=ExternalEvidencePlan
    )
    external_evidence_hits: list[ExternalEvidenceHit] = Field(default_factory=list)
    external_evidence_text: str = Field(default="")
    external_evidence_requests: dict[WorkflowNode, ExternalEvidenceRequest] = Field(
        default_factory=dict
    )
    external_evidence_cache: dict[WorkflowNode, ExternalEvidenceCacheEntry] = Field(
        default_factory=dict
    )

    def get_content_unit_progress_string(self) -> str:
        """Progress string for logging with content unit index."""
        return f"content unit {self.content_unit.index + 1}"

    def set_node_status(self, node: WorkflowNode, status: Status) -> None:
        """Set workflow node status (for logging)."""
        self.status = status

    def set_failure(self, stage: FailureStage, reason: str) -> None:
        """Record failure stage and reason."""
        self.failure_stage = stage
        self.failure_reason = reason
        self.status = Status.FAILED

    def clear_failure(self) -> None:
        """Clear failure state."""
        self.failure_stage = None
        self.failure_reason = None

    def get_external_evidence_request(
        self, node: WorkflowNode
    ) -> ExternalEvidenceRequest:
        """Return node-scoped search request, defaulting to disabled."""
        return self.external_evidence_requests.get(node, ExternalEvidenceRequest())

    def set_external_evidence_request(
        self, node: WorkflowNode, request: ExternalEvidenceRequest
    ) -> None:
        """Store node-scoped search request."""
        self.external_evidence_requests[node] = request

    def set_external_evidence_cache_entry(
        self, node: WorkflowNode, entry: ExternalEvidenceCacheEntry
    ) -> None:
        """Persist node-scoped evidence plan/fetch result cache."""
        self.external_evidence_cache[node] = entry

    def get_external_evidence_cache_entry(
        self, node: WorkflowNode
    ) -> ExternalEvidenceCacheEntry:
        """Return node-scoped evidence cache entry."""
        return self.external_evidence_cache.get(node, ExternalEvidenceCacheEntry())

    def load_external_evidence_for_node(self, node: WorkflowNode) -> None:
        """Load node-scoped evidence cache into active prompt fields."""
        entry = self.get_external_evidence_cache_entry(node)
        self.external_evidence_plan = entry.plan
        self.external_evidence_hits = entry.hits
        self.external_evidence_text = entry.text


class UnitFactsState(UnitState):
    """Independent per-unit state for facts extraction and critique."""

    content_unit: ContentUnit = Field(description="Unit under processing (mutable)")
    facts_user_instruction: str = Field(default="")
    facts_updates: list[GraphUpdate] = Field(default_factory=list)
    deterministic_findings: list[FactsUnitFinding] = Field(
        default_factory=list,
        description=(
            "Machine-found violations/coverage gaps injected as MANDATORY "
            "fixes into the next repair render."
        ),
    )
    critic_fixes_applied: int = Field(
        default=0,
        description=(
            "Critic fixes compiled straight to a patch with no LLM call (tier 1)."
        ),
    )
    critic_fixes_residual: int = Field(
        default=0,
        description=(
            "Critic fixes that could not be compiled and were handed to the "
            "repair render (tier 2), or recorded unapplied on an accepted "
            "render. Never silently dropped."
        ),
    )
    applied_repairs: list[GraphRepairRecord] = Field(
        default_factory=list,
        description=(
            "Deterministic rewrites the machine applied to rendered graphs "
            "(alias repairs, rdf:type literal coercions) — the provenance "
            "trail distinguishing machine-altered triples from LLM output."
        ),
    )

    def update_facts(self) -> None:
        """Apply facts_updates to content_unit.graph and clear the list."""
        if not self.facts_updates:
            return
        updated_graph, _ = _render_updated_graph(
            self.content_unit.graph, self.facts_updates, max_triples=None
        )
        self.content_unit.graph = updated_graph
        self.facts_updates = []


class UnitOntologyState(UnitState):
    """Independent per-unit state for ontology improvement loop."""

    ontology_user_instruction: str = Field(default="")
    working_graph: RDFGraph = Field(
        default_factory=RDFGraph,
        description="Mutable scratchpad graph for in-loop GraphUpdate application.",
    )
    fresh_ontology: Ontology | None = Field(
        default=None,
        description="Full Ontology produced on the fresh-create path (empty seed).",
    )
    ontology_updates: list[GraphUpdate] = Field(default_factory=list)
    ontology_updates_applied: list[GraphUpdate] = Field(default_factory=list)
    current_domain: str = Field(default=DEFAULT_DOMAIN)
    ontology_max_triples: int | None = Field(default=None)
    deterministic_findings: list[OntologyUnitFinding] = Field(
        default_factory=list,
        description=(
            "Machine-found issues in this unit's ontology delta, injected "
            "into the critic prompt and summed into the document residual."
        ),
    )

    def model_post_init(self, __context) -> None:
        """Initialize mutable working graph from immutable snapshot."""
        if len(self.working_graph) == 0 and not self.ontology_snapshot.is_empty():
            self.working_graph = self.ontology_snapshot.graph.copy()

    @property
    def all_updates(self) -> list[GraphUpdate]:
        """All ontology updates produced by this unit (applied and pending)."""
        return [*self.ontology_updates_applied, *self.ontology_updates]

    def build_delta(self) -> OntologyDelta:
        """Net insert/delete delta of this unit against its prompt snapshot.

        All GraphUpdates (applied and pending) are replayed in order onto a
        copy of the snapshot, then diffed against it. This honors operation
        order -- a triple deleted and later re-inserted nets out -- and yields:

        - ``inserts``: true complements (``U \\ S``), never restated context
          triples;
        - ``deletes``: snapshot triples removed by delete operations, to be
          propagated onto catalog terminals during reduce.

        Fresh path (no GraphUpdates, empty seed): full working graph as
        inserts. Costs a snapshot copy per call, which is why the snapshot is
        otherwise shared by reference -- callers on the per-unit hot path
        budget-time it.
        """
        if self.all_updates:
            snapshot_graph = self.ontology_snapshot.graph
            final_graph, _ = AgentState.render_updated_graph(
                snapshot_graph, self.all_updates, max_triples=None
            )
            snapshot_set = set(snapshot_graph)
            final_set = set(final_graph)
            inserts = RDFGraph()
            deletes = RDFGraph()
            for prefix, namespace_uri in final_graph.namespaces():
                if prefix:
                    inserts.bind(prefix, namespace_uri)
                    deletes.bind(prefix, namespace_uri)
            for triple in final_set - snapshot_set:
                inserts.add(triple)
            for triple in snapshot_set - final_set:
                deletes.add(triple)
            if len(deletes) > 0:
                logger.info(
                    "build_delta: unit produced %d delete triple(s) for "
                    "catalog propagation.",
                    len(deletes),
                )
            return OntologyDelta(inserts=inserts, deletes=deletes)

        # Fresh generation with no structured updates: emit the working graph
        # only when the seed was empty (true create path).
        if self.ontology_snapshot.is_empty() and len(self.working_graph) > 0:
            return OntologyDelta(inserts=self.working_graph.copy())
        return OntologyDelta()

    def update_ontology(self) -> bool:
        """Apply ontology_updates to working_graph and clear the list.

        Returns:
            True when the updates were applied. False means the
            ``ontology_max_triples`` backstop rejected them and the working
            graph is unchanged -- the caller must not report that as a
            successful render without saying so, because a validator run
            afterwards would inspect the pre-update graph and find it clean.
        """
        if not self.ontology_updates:
            return True
        updated_graph, was_applied = _render_updated_graph(
            self.working_graph,
            self.ontology_updates,
            max_triples=self.ontology_max_triples,
        )
        if not was_applied:
            return False

        self.ontology_updates_applied += self.ontology_updates
        self.working_graph = updated_graph
        self.ontology_updates = []
        return True

    def working_graph_changed(self) -> bool:
        """True when the scratchpad graph differs from the seed snapshot.

        Plain set comparison is sound here: the working graph starts as an
        in-process copy of the snapshot graph (blank-node identity preserved,
        no serialization round-trip), so canonicalization-grade hashing adds
        cost without adding correctness.
        """
        snapshot_graph = self.ontology_snapshot.graph
        if len(self.working_graph) != len(snapshot_graph):
            return True
        if len(self.working_graph) == 0:
            return False
        return set(self.working_graph) != set(snapshot_graph)
