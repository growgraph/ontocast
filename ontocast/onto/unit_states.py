"""Dedicated state models for parallel unit loops."""

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
    FactsLoopAttempt,
    FactsUnitFinding,
    GraphRepairRecord,
    Suggestions,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState, BudgetTracker


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
    budget_tracker: BudgetTracker = Field(default_factory=BudgetTracker)
    max_visits_per_node: int = Field(default=1, ge=1)
    #: Critic attempts allowed per render attempt. None couples it to
    #: ``max_visits_per_node``, which makes the worst case quadratic.
    max_critic_visits_per_node: int | None = Field(default=None, ge=1)
    llm_graph_format: LLMGraphFormat = Field(
        default=LLMGraphFormat.TURTLE,
        description=(
            "Format used by the LLM for emitting RDF graph payloads: "
            "'turtle' or 'jsonld'."
        ),
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
    quarantined_literal_triples: list[RejectedLiteralTriple] = Field(
        default_factory=list,
        description="Triples excluded from the applied graph due to invalid XSD typed literals.",
    )
    deterministic_findings: list[FactsUnitFinding] = Field(
        default_factory=list,
        description=(
            "Machine-found violations/coverage gaps injected as MANDATORY "
            "fixes into the next repair render."
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
    attempt_log: list[FactsLoopAttempt] = Field(
        default_factory=list,
        description="Per-attempt telemetry (render/critic/repair) for this unit.",
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

    def model_post_init(self, __context) -> None:
        """Initialize mutable working graph from immutable snapshot."""
        if len(self.working_graph) == 0 and not self.ontology_snapshot.is_empty():
            self.working_graph = self.ontology_snapshot.graph.copy()

    @property
    def all_updates(self) -> list[GraphUpdate]:
        """All ontology updates produced by this unit (applied and pending)."""
        return [*self.ontology_updates_applied, *self.ontology_updates]

    def update_ontology(self) -> None:
        """Apply ontology_updates to working_graph and clear the list."""
        if not self.ontology_updates:
            return
        updated_graph, was_applied = _render_updated_graph(
            self.working_graph,
            self.ontology_updates,
            max_triples=self.ontology_max_triples,
        )
        if not was_applied:
            return

        self.ontology_updates_applied += self.ontology_updates
        self.working_graph = updated_graph
        self.ontology_updates = []

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
