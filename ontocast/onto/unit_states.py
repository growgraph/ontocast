"""Dedicated state models for parallel unit loops."""

from collections import defaultdict
from copy import deepcopy

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
    Suggestions,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.state import AgentState, BudgetTracker


def _render_updated_graph(
    graph: RDFGraph, updates: list[GraphUpdate], max_triples: int | None = None
) -> tuple[RDFGraph, bool]:
    """Apply GraphUpdate objects to a graph. Delegates to AgentState implementation."""
    return AgentState.render_updated_graph(graph, updates, max_triples=max_triples)


class UnitState(BasePydanticModel):
    """Common per-unit workflow state."""

    ontology_snapshot: Ontology = Field(description="Immutable ontology snapshot")
    ontology_patch_sources: list[str] = Field(
        default_factory=list,
        description="Ontology IRIs that contributed to the snapshot context.",
    )
    suggestions: Suggestions = Field(default_factory=Suggestions)
    budget_tracker: BudgetTracker = Field(default_factory=BudgetTracker)
    max_visits_per_node: int = Field(default=1, ge=1)
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
    external_evidence_source_count: int = Field(default=0, ge=0)
    external_evidence_domains: list[str] = Field(default_factory=list)
    external_evidence_planned_at_node: WorkflowNode | None = Field(default=None)
    external_evidence_used_by_nodes: list[WorkflowNode] = Field(default_factory=list)
    external_evidence_requests: dict[WorkflowNode, ExternalEvidenceRequest] = Field(
        default_factory=dict
    )
    external_evidence_cache: dict[WorkflowNode, ExternalEvidenceCacheEntry] = Field(
        default_factory=dict
    )

    def get_content_unit_progress_string(self) -> str:
        """Progress string for logging (single unit context)."""
        return "content unit"

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

    def clear_external_evidence(self) -> None:
        """Reset evidence plan, retrieved hits, and rendered evidence block."""
        self.external_evidence_plan = ExternalEvidencePlan()
        self.external_evidence_hits = []
        self.external_evidence_text = ""
        self.external_evidence_source_count = 0
        self.external_evidence_domains = []
        self.external_evidence_planned_at_node = None
        self.external_evidence_cache = {}

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

    def clear_external_evidence_request(self, node: WorkflowNode) -> None:
        """Clear node-scoped search request."""
        self.external_evidence_requests.pop(node, None)

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
        self.external_evidence_source_count = entry.source_count
        self.external_evidence_domains = entry.domains
        self.external_evidence_planned_at_node = node

    def mark_external_evidence_used(self, node: WorkflowNode) -> None:
        """Record that a workflow node consumed prepared external evidence."""
        if node not in self.external_evidence_used_by_nodes:
            self.external_evidence_used_by_nodes.append(node)


class UnitFactsState(UnitState):
    """Independent per-unit state for facts extraction and critique."""

    content_unit: ContentUnit = Field(description="Unit under processing (mutable)")
    facts_user_instruction: str = Field(default="")
    facts_updates: list[GraphUpdate] = Field(default_factory=list)
    quarantined_literal_triples: list[RejectedLiteralTriple] = Field(
        default_factory=list,
        description="Triples excluded from the applied graph due to invalid XSD typed literals.",
    )
    assembly_anchor_iri: str = Field(
        default="",
        description="Anchor IRI from context assembly (or merged document primary).",
    )
    assembly_mode_used: OntologyAssemblyMode = Field(
        default=OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
        description="How ontology_snapshot was assembled for this unit.",
    )

    def get_content_unit_progress_string(self) -> str:
        """Progress string for logging with content unit index."""
        return f"content unit {self.content_unit.index + 1}"

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

    content_unit: SourceUnit = Field(description="Unit under processing")
    assembly_anchor_iri: str = Field(
        default="",
        description="Anchor IRI from resolve_unit_ontology_context prelude.",
    )
    assembly_mode_used: OntologyAssemblyMode = Field(
        default=OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
        description="Ontology assembly mode from the context prelude.",
    )
    ontology_user_instruction: str = Field(default="")
    current_ontology: Ontology = Field(
        default_factory=Ontology, description="Current ontology under refinement"
    )
    ontology_updates: list[GraphUpdate] = Field(default_factory=list)
    ontology_updates_applied: list[GraphUpdate] = Field(default_factory=list)
    current_domain: str = Field(default=DEFAULT_DOMAIN)
    ontology_max_triples: int | None = Field(default=None)

    def get_content_unit_progress_string(self) -> str:
        """Progress string for logging with content unit index."""
        return f"content unit {self.content_unit.index + 1}"

    def model_post_init(self, __context) -> None:
        """Initialize mutable ontology state from immutable snapshot."""
        self.current_ontology = deepcopy(self.ontology_snapshot)

    @property
    def all_updates(self) -> list[GraphUpdate]:
        """All ontology updates produced by this unit (applied and pending)."""
        return [*self.ontology_updates_applied, *self.ontology_updates]

    def update_ontology(self) -> None:
        """Apply ontology_updates to current_ontology and clear the list."""
        if not self.ontology_updates:
            return
        updated_graph, was_applied = _render_updated_graph(
            self.current_ontology.graph,
            self.ontology_updates,
            max_triples=self.ontology_max_triples,
        )
        if not was_applied:
            return

        updated_ontology = self.current_ontology.derive_updated_version(updated_graph)
        self.ontology_updates_applied += self.ontology_updates
        self.current_ontology = updated_ontology
        self.ontology_updates = []
