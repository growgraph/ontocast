from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from pydantic import ConfigDict, Field
from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_DOMAIN, ONTOLOGY_NULL_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.context import AgentContext, AgentType, ContextManager
from ontocast.onto.enum import (
    FailureStage,
    LLMGraphFormat,
    OntologyAssemblyMode,
    OntologyContextMode,
    RenderMode,
    Status,
    WorkflowNode,
)
from ontocast.onto.iri_policy import normalize_namespace_iri
from ontocast.onto.model import BasePydanticModel, Suggestions
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.util import render_text_hash

# Top-level SPARQL update keywords at line start (used to split compound LLM output).
_TOP_LEVEL_UPDATE_START_RE = re.compile(r"(?m)^(?=(?:INSERT|DELETE|WITH)\b)")


class BudgetTracker(BasePydanticModel):
    """Lightweight tracker for LLM usage statistics and generated triples."""

    chars_sent: int = Field(default=0, description="Total characters sent to LLM")
    chars_received: int = Field(
        default=0, description="Total characters received from LLM"
    )
    calls_count: int = Field(default=0, description="Total number of LLM API calls")
    input_tokens: int = Field(
        default=0, description="Total input tokens (when reported by provider)"
    )
    output_tokens: int = Field(
        default=0, description="Total output tokens (when reported by provider)"
    )

    # Triple generation tracking
    ontology_triples_generated: int = Field(
        default=0, description="Total number of triples generated for ontology updates"
    )
    facts_triples_generated: int = Field(
        default=0, description="Total number of triples generated for facts"
    )
    ontology_operations_count: int = Field(
        default=0, description="Total number of ontology update operations"
    )
    facts_operations_count: int = Field(
        default=0, description="Total number of facts update operations"
    )

    def add_usage(
        self,
        chars_sent: int,
        chars_received: int,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Add usage statistics."""
        self.chars_sent += chars_sent
        self.chars_received += chars_received
        self.calls_count += 1
        if input_tokens is not None:
            self.input_tokens += input_tokens
        if output_tokens is not None:
            self.output_tokens += output_tokens

    def add_ontology_update(self, num_operations: int, num_triples: int) -> None:
        """Add ontology update statistics.

        Args:
            num_operations: Number of update operations generated
            num_triples: Number of triples in these operations
        """
        self.ontology_operations_count += num_operations
        self.ontology_triples_generated += num_triples

    def add_facts_update(self, num_operations: int, num_triples: int) -> None:
        """Add facts update statistics.

        Args:
            num_operations: Number of update operations generated
            num_triples: Number of triples in these operations
        """
        self.facts_operations_count += num_operations
        self.facts_triples_generated += num_triples

    def merge_from(self, other: BudgetTracker) -> None:
        """Accumulate counters from another tracker (e.g. parallel unit workers)."""
        self.chars_sent += other.chars_sent
        self.chars_received += other.chars_received
        self.calls_count += other.calls_count
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.ontology_triples_generated += other.ontology_triples_generated
        self.facts_triples_generated += other.facts_triples_generated
        self.ontology_operations_count += other.ontology_operations_count
        self.facts_operations_count += other.facts_operations_count

    def get_summary(self) -> str:
        """Get a summary of LLM usage and generated triples."""
        parts = [
            f"LLM: {self.calls_count} calls, "
            f"{self.chars_sent:,} sent, "
            f"{self.chars_received:,} received",
        ]

        if self.input_tokens > 0 or self.output_tokens > 0:
            parts.append(
                f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens"
            )

        if self.ontology_triples_generated > 0 or self.facts_triples_generated > 0:
            parts.append(
                f"Triples: {self.ontology_triples_generated} ontology, "
                f"{self.facts_triples_generated} facts"
            )

        return " | ".join(parts)


class AgentState(BasePydanticModel):
    """State for the ontology-based knowledge graph agent.

    This class maintains the state of the agent during document processing,
    including input text, content units, ontologies, and workflow status.

    Attributes:
        input_text: Input text to process.
        current_domain: IRI used for forming document namespace.
        doc_hid: An almost unique hash/id for the parent document.
        raw_input: Single raw input payload as {filename: bytes}.
        ontology_addendum: Additional ontology content.
        failure_stage: Stage where failure occurred.
        failure_reason: Reason for failure.
        success_score: Score indicating success level.
        status: Current workflow status.
        node_visits: Number of visits per node.
        max_visits: Maximum number of visits allowed per node.
        max_chunks: Maximum number of source content units to split and process.
    """

    input_text: str = Field(description="Input text", default="")
    current_domain: str = Field(
        description="IRI used for forming document namespace", default=DEFAULT_DOMAIN
    )
    doc_hid: str = Field(
        description="An almost unique hash / id for the parent document of the current unit",
        default="default_doc",
    )
    raw_input: dict[str, bytes] = Field(
        default_factory=dict,
        description="Single raw input payload: {filename: bytes}.",
    )
    content_units: list[ContentUnit] = Field(
        default_factory=list,
        description="Pending content units to process.",
    )
    ontology_patch_sources: list[str] = Field(
        default_factory=list,
        description="Ontology IRIs that contributed to a retrieved multi-source patch context.",
    )
    ontology_artifacts: list[Ontology] = Field(
        default_factory=list,
        description="Final per-anchor ontology artifacts produced for this document.",
    )
    reduced_ontology_artifacts: list[Ontology] = Field(
        default_factory=list,
        description="Reduced ontology artifacts after explicit ontology reduce step.",
    )
    reduced_ontology_by_anchor: dict[str, Ontology] = Field(
        default_factory=dict,
        description="Reduced ontology artifacts indexed by anchor IRI.",
    )
    ontology_reduce_metrics: dict[str, int | float | str] = Field(
        default_factory=dict,
        description="Metrics emitted by ontology reduce stage.",
    )
    ontology_reduce_provenance: RDFGraph = Field(
        default_factory=RDFGraph,
        description="Optional provenance graph emitted by ontology reduce stage.",
    )
    candidate_anchor_iris: list[str] = Field(
        default_factory=list,
        description="Candidate ontology IRIs discovered during multi-anchor preselection.",
    )
    unit_anchor_assignment: dict[int, str] = Field(
        default_factory=dict,
        description="Assigned anchor ontology IRI per content unit index.",
    )
    unit_patch_sources: dict[int, list[str]] = Field(
        default_factory=dict,
        description="Retrieved ontology source IRIs per content unit index.",
    )
    unit_context_mode_used: dict[int, OntologyAssemblyMode] = Field(
        default_factory=dict,
        description="Per-unit ontology assembly mode (ensemble / vote majority / primary).",
    )
    retrieval_metrics: dict[str, int | float | str] = Field(
        default_factory=dict,
        description="Runtime retrieval/evaluation metrics for observability.",
    )
    aggregated_facts: RDFGraph = Field(
        description="RDF triples representing aggregated facts "
        "from the current document",
        default_factory=RDFGraph,
    )
    ontology_user_instruction: str = Field(
        description="Specific user instructions for ontology extraction, e.g. `Focus on extracting places`",
        default="",
    )

    ontology_selection_user_instruction: str = Field(
        description=(
            "Specific user instructions for ontology selection, "
            "e.g. `Prefer ontologies focused on finance`"
        ),
        default="",
    )

    facts_user_instruction: str = Field(
        description="Specific user instructions for facts extraction, e.g. `Focus on extracting places`",
        default="",
    )

    ontology_context_fixed_ontology_id: str = Field(
        description=(
            "Catalog ontology id when ontology_context_mode is fixed_single_ontology "
            "(resolved via OntologyManager)."
        ),
        default="",
    )

    tenant: str | None = Field(
        default=None,
        description="Tenant id when request selected tenancy via query/CLI.",
    )
    project: str | None = Field(
        default=None,
        description="Project id when request selected tenancy via query/CLI.",
    )

    graph_uri_override: str | None = Field(default=None)

    source_url: str | None = Field(
        description="Source URL from JSON input file (for provenance tracking)",
        default=None,
    )

    ontology_updates: list[GraphUpdate] = Field(
        default_factory=list,
        description="A list of graph update that improve the current ontology",
    )

    ontology_updates_applied: list[GraphUpdate] = Field(
        default_factory=list,
        description="A list of graph update that improve the current ontology",
    )

    facts_updates: list[GraphUpdate] = Field(
        default_factory=list,
        description="A list of graph update that improve the current graph of facts (pending)",
    )

    facts_updates_applied: list[GraphUpdate] = Field(
        default_factory=list,
        description="A list of graph update that improve the current graph of facts (applied)",
    )

    facts_units: list[ContentUnit] = Field(
        default_factory=list,
        description="Successful per-unit facts outputs collected during parallel map phase",
    )

    ontology_units: list[ContentUnit] = Field(
        default_factory=list,
        description="Successful per-unit ontology outputs collected during parallel map phase",
    )
    ontology_provenance_artifact: RDFGraph = Field(
        default_factory=RDFGraph,
        description="Provenance/reification triples stripped from normalized ontology.",
    )

    ontology_addendum: Ontology = Field(
        default_factory=lambda: Ontology(
            ontology_id=None,
            title=None,
            description=None,
            graph=RDFGraph(),
            iri=ONTOLOGY_NULL_IRI,
        ),
        description="Ontology object that contain the semantic graph "
        "as well as the description, name, short name, version, "
        "and IRI of the ontology",
    )
    failure_stage: FailureStage | None = None
    failure_reason: str | None = None

    improvements_suggestions: list[str] = Field(
        description="Itemized concrete and actionable instructions for improvements of extraction of facts/ontology",
        default_factory=list,
    )

    success_score: float = 0.0
    status: Status = Status.SUCCESS
    statuses: dict[WorkflowNode, Status] = Field(
        default_factory=dict, description="Status of each node"
    )
    node_visits: defaultdict[WorkflowNode, int] = Field(
        default_factory=lambda: defaultdict(int),
        description="Number of visits per node",
    )
    max_visits: int = Field(
        default=3, description="Maximum number of visits allowed per node"
    )
    max_chunks: int | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
    render_mode: RenderMode = Field(
        default=RenderMode.ONTOLOGY_AND_FACTS,
        description=("Rendering mode: ontology, facts, or ontology_and_facts."),
    )
    llm_graph_format: LLMGraphFormat = Field(
        default=LLMGraphFormat.TURTLE,
        description=(
            "Format used by the LLM for emitting RDF graph payloads: "
            "'turtle' (legacy) or 'jsonld' (compact JSON-LD objects embedded "
            "directly in the structured response)."
        ),
    )
    ontology_context_mode: OntologyContextMode = Field(
        default=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        description=(
            "Per-unit ontology context: selected_single_ontology (LLM-picked catalog), "
            "selected_vector_search_ontology (Qdrant ensemble), or "
            "fixed_single_ontology (catalog ontology_id via ontology_context_fixed_ontology_id)."
        ),
    )
    ontology_max_triples: int | None = Field(
        default=50000,
        description="Maximum number of triples allowed in ontology graph. "
        "Updates that would exceed this limit are skipped with a warning. "
        "Set to None for unlimited.",
    )
    context_manager: ContextManager = Field(
        default_factory=ContextManager,
        description="Context manager for passing information between agents",
    )
    suggestions: Suggestions = Field(
        default_factory=Suggestions,
        description="Structured critique feedback for the next render/critic pass",
    )

    # Budget Tracking
    budget_tracker: BudgetTracker = Field(
        default_factory=BudgetTracker,
        description="Budget statistics tracker (LLM usage and generated triples)",
    )

    def model_post_init(self, __context):
        """Post-initialization hook for the model."""
        pass

    def __init__(self, **kwargs):
        """Initialize the agent state with given keyword arguments."""
        super().__init__(**kwargs)
        self.current_domain = os.getenv("CURRENT_DOMAIN", DEFAULT_DOMAIN)

    def get_node_status(self, node: WorkflowNode) -> Status:
        """Get the status of a workflow node, returning NOT_VISITED if not set."""
        return self.statuses.get(node, Status.NOT_VISITED)

    @property
    def render_ontology(self) -> bool:
        """Whether ontology rendering should run."""
        return self.render_mode in (
            RenderMode.ONTOLOGY,
            RenderMode.ONTOLOGY_AND_FACTS,
        )

    @property
    def render_facts(self) -> bool:
        """Whether facts rendering should run."""
        return self.render_mode in (
            RenderMode.FACTS,
            RenderMode.ONTOLOGY_AND_FACTS,
        )

    def set_node_status(self, node: WorkflowNode, status: Status) -> None:
        """Set the status of a workflow node."""
        self.statuses[node] = status

    def get_content_unit_progress_info(self) -> tuple[int, int]:
        """Get current content unit number and total content units."""
        total_content_units = len(self.content_units)
        current_content_unit_number = 1 if total_content_units > 0 else 0
        return current_content_unit_number, total_content_units

    def get_content_unit_progress_string(self) -> str:
        """Get a formatted string showing content unit progress."""
        current, total = self.get_content_unit_progress_info()
        if total == 0:
            return "no content units"
        return f"content unit {current}/{total}"

    def get_chunk_progress_info(self) -> tuple[int, int]:
        """Backward-compatible wrapper for content unit progress.

        Returns:
            tuple[int, int]: (current_chunk_number, total_chunks)
        """
        return self.get_content_unit_progress_info()

    def get_chunk_progress_string(self) -> str:
        """Backward-compatible wrapper for content unit progress.

        Returns:
            str: Formatted string like "chunk 3/10"
        """
        return self.get_content_unit_progress_string()

    @classmethod
    def render_updated_graph(
        cls, graph: RDFGraph, updates: list[GraphUpdate], max_triples: int | None = None
    ) -> tuple[RDFGraph, bool]:
        """Create a copy of the given graph with all GraphUpdate objects applied.

        This method:
        1. Creates a copy of the input graph
        2. Generates SPARQL queries from all GraphUpdate objects
        3. Executes the queries on the copied graph
        4. Checks if the updated graph exceeds max_triples limit
        5. Returns the updated graph copy, or original if limit exceeded

        Args:
            graph: The RDFGraph to update
            updates: List of GraphUpdate objects to apply
            max_triples: Maximum number of triples allowed. If None, no limit enforced.

        Returns:
            Tuple of (RDFGraph, bool): The updated graph (or original if limit exceeded),
            and a boolean indicating if the update was applied (True) or skipped (False)
        """
        if not updates:
            return graph, True

        # Create a copy of the input graph
        # Use RDFGraph's copy method to preserve type
        updated_graph = RDFGraph()
        for triple in graph:
            updated_graph.add(triple)
        # Copy namespace bindings
        for prefix, namespace in graph.namespaces():
            updated_graph.bind(prefix, namespace)

        all_prefixes = {}
        for graph_update in updates:
            for op in graph_update.triple_operations:
                # Extract prefixes from TripleOp operations
                if isinstance(op, TripleOp) and op.prefixes:
                    all_prefixes.update(op.prefixes)

        # Bind prefixes to the copied graph
        for prefix, uri in all_prefixes.items():
            updated_graph.bind(prefix, uri)

        # Apply each GraphUpdate to the copied graph
        for graph_update in updates:
            # Generate SPARQL queries from the GraphUpdate
            queries = graph_update.generate_sparql_queries()

            # Execute each query on the copied graph
            for query in queries:
                cls._apply_update_query(updated_graph, query)

        # Check if updated graph exceeds max_triples limit
        if max_triples is not None and len(updated_graph) > max_triples:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Ontology update skipped: would exceed limit "
                f"({len(updated_graph)} > {max_triples} triples). "
                f"Original size: {len(graph)} triples."
            )
            return graph, False  # Return original, unchanged

        return updated_graph, True

    @classmethod
    def _apply_update_query(cls, graph: RDFGraph, query: str) -> None:
        """Apply one SPARQL update query, splitting compound LLM output proactively."""
        parts = cls._split_compound_sparql_query(query)
        for part in parts:
            graph.update(part)

    @staticmethod
    def _split_compound_sparql_query(query: str) -> list[str]:
        """Split a query string containing concatenated top-level UPDATE statements.

        LLMs frequently emit several ``INSERT DATA`` / ``DELETE DATA`` blocks joined
        after a shared ``PREFIX`` block.  Splitting on top-level keyword boundaries
        before calling ``graph.update`` avoids parse errors entirely.

        A single-statement query is returned as a one-element list.
        """
        stripped = query.strip()
        if not stripped:
            return [stripped]

        starts = [m.start() for m in _TOP_LEVEL_UPDATE_START_RE.finditer(stripped)]
        if len(starts) <= 1:
            return [stripped]

        prefix_block = stripped[: starts[0]].strip()
        parts: list[str] = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(stripped)
            body = stripped[start:end].strip()
            if body:
                parts.append(f"{prefix_block}\n{body}" if prefix_block else body)
        return parts or [stripped]

    def generate_ontology_updates_markdown(self) -> str:
        """Generate a markdown string representing the chain of ontology updates.

        Returns:
            Markdown-formatted string showing all pending ontology updates.
            Returns empty string if no updates are pending.
        """
        if not self.ontology_updates:
            return ""

        markdown_parts = []
        for i, graph_update in enumerate(self.ontology_updates, 1):
            diff_summary = graph_update.generate_diff_summary()
            if diff_summary:
                markdown_parts.append(f"## Update {i}")
                markdown_parts.append(diff_summary)

            markdown_parts.append("")

            # Add separator between updates (except for the last one)
            if i < len(self.ontology_updates):
                markdown_parts.append("---")
                markdown_parts.append("")

        return "\n".join(markdown_parts)

    def set_text(self, text):
        """Set the input text and generate document hash.

        Args:
            text: The input text to set.
        """
        self.input_text = text
        self.doc_hid = render_text_hash(self.input_text)

    def set_failure(self, stage: FailureStage, reason: str, success_score: float = 0.0):
        """Set failure state with stage and reason.

        Args:
            stage: The stage where the failure occurred.
            reason: The reason for the failure.
            success_score: The success score at failure (default: 0.0).
        """
        self.failure_stage = stage
        self.failure_reason = reason
        self.success_score = success_score
        self.status = Status.FAILED

    def clear_failure(self):
        """Clear failure state and set status to success."""
        self.failure_stage = None
        self.failure_reason = None
        self.success_score = 0.0
        self.status = Status.SUCCESS

    @property
    def doc_iri(self) -> URIRef:
        """Get the document IRI.

        Returns:
            str: The document IRI.
        """
        return URIRef(f"{self.current_domain}/doc/{self.doc_hid}")

    @property
    def doc_namespace(self):
        """Get the document namespace.

        Returns:
            str: The document namespace.
        """
        return normalize_namespace_iri(self.doc_iri, context="facts")

    @property
    def graph_uri(self):
        if self.graph_uri_override is not None:
            return self.graph_uri_override
        return self.doc_namespace

    @property
    def ontology_ids(self) -> list[str]:
        """Ontology ids for all current ontology artifacts."""
        artifacts = (
            self.reduced_ontology_artifacts
            if self.reduced_ontology_artifacts
            else self.ontology_artifacts
        )
        return [ontology.ontology_id for ontology in artifacts if ontology.ontology_id]

    def get_context_for_agent(self, agent_type: AgentType) -> AgentContext:
        """Get or create context for a specific agent.

        Args:
            agent_type: Type of agent (renderer, critic, etc.).

        Returns:
            AgentContext: The context for the agent.
        """
        existing_context = self.context_manager.get_latest_context_by_agent(agent_type)

        if existing_context:
            return existing_context

        # Create new context if none exists
        return self.context_manager.create_context(agent_type=agent_type)

    def update_context_for_agent(
        self,
        agent_type: AgentType,
        ontology_version: Any | None = None,
        facts_version: Any | None = None,
        ontology_operations: list[Any] | None = None,
        facts_operations: list[Any] | None = None,
        ontology_critique: dict[str, Any] | None = None,
        facts_critique: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentContext:
        """Update context for a specific agent.

        Args:
            agent_type: Name of the agent updating context.
            ontology_version: New ontology version if available.
            facts_version: New facts version if available.
            ontology_operations: New ontology operations if available.
            facts_operations: New facts operations if available.
            ontology_critique: New ontology critique if available.
            facts_critique: New facts critique if available.
            metadata: Additional metadata for the context.

        Returns:
            AgentContext: The updated context.
        """
        return self.context_manager.update_context(
            agent_type=agent_type,
            ontology_version=ontology_version,
            facts_version=facts_version,
            ontology_operations=ontology_operations,
            facts_operations=facts_operations,
            ontology_critique=ontology_critique,
            facts_critique=facts_critique,
            metadata=metadata,
        )

    def get_context_summary_for_agent(self, agent_type: AgentType) -> str:
        """Get a context summary for a specific agent.

        Args:
            agent_type: Name of the agent requesting context summary.

        Returns:
            str: A formatted context summary.
        """
        context = self.context_manager.get_latest_context_by_agent(agent_type)
        if not context:
            return "No context available for this agent."

        return context.get_full_context_summary()
