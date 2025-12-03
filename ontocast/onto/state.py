import os
from collections import defaultdict
from typing import Any

from pydantic import ConfigDict, Field

from ontocast.onto.chunk import Chunk
from ontocast.onto.constants import (
    CHUNK_NULL_IRI,
    DEFAULT_DOMAIN,
    ONTOLOGY_NULL_IRI,
)
from ontocast.onto.context import AgentContext, AgentType, ContextManager
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import BasePydanticModel, Suggestions
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.util import iri2namespace, render_text_hash


class BudgetTracker(BasePydanticModel):
    """Lightweight tracker for LLM usage statistics and generated triples."""

    chars_sent: int = Field(default=0, description="Total characters sent to LLM")
    chars_received: int = Field(
        default=0, description="Total characters received from LLM"
    )
    calls_count: int = Field(default=0, description="Total number of LLM API calls")

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

    def add_usage(self, chars_sent: int, chars_received: int) -> None:
        """Add usage statistics."""
        self.chars_sent += chars_sent
        self.chars_received += chars_received
        self.calls_count += 1

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

    def get_summary(self) -> str:
        """Get a summary of LLM usage and generated triples."""
        parts = [
            f"LLM: {self.calls_count} calls, "
            f"{self.chars_sent:,} sent, "
            f"{self.chars_received:,} received",
        ]

        if self.ontology_triples_generated > 0 or self.facts_triples_generated > 0:
            parts.append(
                f"Triples: {self.ontology_triples_generated} ontology, "
                f"{self.facts_triples_generated} facts"
            )

        return " | ".join(parts)


class AgentState(BasePydanticModel):
    """State for the ontology-based knowledge graph agent.

    This class maintains the state of the agent during document processing,
    including input text, chunks, ontologies, and workflow status.

    Attributes:
        input_text: Input text to process.
        current_domain: IRI used for forming document namespace.
        doc_hid: An almost unique hash/id for the parent document.
        files: Files to process.
        current_chunk: Current document chunk for processing (property, accessed via index).
        chunks: List of chunks of the input text.
        chunks_processed: List of processed chunks.
        current_ontology: Current ontology object.
        ontology_addendum: Additional ontology content.
        failure_stage: Stage where failure occurred.
        failure_reason: Reason for failure.
        success_score: Score indicating success level.
        status: Current workflow status.
        node_visits: Number of visits per node.
        max_visits: Maximum number of visits allowed per node.
        max_chunks: Maximum number of chunks to process.
    """

    input_text: str = Field(description="Input text", default="")
    current_domain: str = Field(
        description="IRI used for forming document namespace", default=DEFAULT_DOMAIN
    )
    doc_hid: str = Field(
        description="An almost unique hash / id for the parent document of the chunk",
        default="default_doc",
    )
    files: dict[str, bytes] = Field(
        default_factory=lambda: dict(), description="Files to process"
    )
    chunks: list[Chunk] = Field(
        default_factory=lambda: list(), description="Chunks of the input text"
    )
    current_chunk: Chunk = Field(
        default_factory=lambda: Chunk(
            text="",
            hid="default",
            doc_iri=CHUNK_NULL_IRI,
        ),
        description="Chunks of the input text",
    )
    chunks_processed: list[Chunk] = Field(
        default_factory=lambda: list(), description="Chunks of the input text"
    )
    current_ontology: Ontology = Field(
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
    aggregated_facts: RDFGraph = Field(
        description="RDF triples representing aggregated facts "
        "from the current document",
        default_factory=RDFGraph,
    )
    ontology_user_instruction: str = Field(
        description="Specific user instructions for ontology extraction, e.g. `Focus on extracting places`",
        default="",
    )

    facts_user_instruction: str = Field(
        description="Specific user instructions for facts extraction, e.g. `Focus on extracting places`",
        default="",
    )

    dataset: str | None = Field(
        description="Fuseki dataset name for this request (optional)",
        default=None,
    )

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
    model_config = ConfigDict(arbitrary_types_allowed=True)
    skip_ontology_development: bool = Field(
        default=False, description="Skip ontology create/improve steps if True"
    )
    skip_facts_rendering: bool = Field(
        default=False,
        description="Skip facts rendering and go straight to aggregation if True",
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
        description="Context manager for passing information between agents",
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

    def set_node_status(self, node: WorkflowNode, status: Status) -> None:
        """Set the status of a workflow node."""
        self.statuses[node] = status

    def get_chunk_progress_info(self) -> tuple[int, int]:
        """Get current chunk number and total chunks.

        Returns:
            tuple[int, int]: (current_chunk_number, total_chunks)
        """
        from ontocast.onto.constants import CHUNK_NULL_IRI

        # Check if there's a chunk currently being processed
        has_current_chunk = CHUNK_NULL_IRI not in self.current_chunk.iri

        # Current chunk number = chunks done + (1 if currently processing, else 0)
        current_chunk_number = len(self.chunks_processed) + (
            1 if has_current_chunk else 0
        )

        # Total chunks = remaining + done + (1 if currently processing, else 0)
        total_chunks = (
            len(self.chunks)
            + len(self.chunks_processed)
            + (1 if has_current_chunk else 0)
        )

        return current_chunk_number, total_chunks

    def get_chunk_progress_string(self) -> str:
        """Get a formatted string showing chunk progress.

        Returns:
            str: Formatted string like "chunk 3/10"
        """
        current, total = self.get_chunk_progress_info()
        if total == 0:
            return "no chunks"
        return f"chunk {current}/{total}"

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
                updated_graph.update(query)

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

    def render_uptodate_ontology(self) -> Ontology:
        """Create a copy of the current ontology with all GraphUpdate objects applied.

        This method:
        1. Creates a copy of the current ontology
        2. Generates SPARQL queries from all GraphUpdate objects
        3. Executes the queries on the copied ontology graph
        4. Checks if the updated graph exceeds max_triples limit
        5. Sets the current hash as parent_hash in the updated ontology
        6. Computes a new hash for the updated ontology
        7. Syncs properties to ensure object fields are updated
        8. Returns the updated ontology copy, or original if limit exceeded

        Returns:
            Ontology: A copy of the current ontology with all updates applied and
            a new hash generated, with the previous hash set as parent.
            Returns original ontology if update would exceed max_triples limit.
        """
        if not self.ontology_updates:
            return self.current_ontology

        # Create a copy of the current ontology
        from copy import deepcopy

        updated_ontology = deepcopy(self.current_ontology)

        # Use the generalized function to update the graph
        updated_graph, was_applied = self.render_updated_graph(
            self.current_ontology.graph,
            self.ontology_updates,
            max_triples=self.ontology_max_triples,
        )
        updated_ontology.graph = updated_graph

        # If graph wasn't updated (limit exceeded), return original ontology
        if not was_applied:
            return self.current_ontology

        # Set current hash as parent and generate new hash
        if self.current_ontology.hash:
            # Set current hash as parent
            updated_ontology.parent_hashes = [self.current_ontology.hash]
        else:
            # If no current hash, this is a root ontology with no parents
            updated_ontology.parent_hashes = []

        # Set created_at for new version if not already set
        if not updated_ontology.created_at:
            from datetime import datetime, timezone

            updated_ontology.created_at = datetime.now(timezone.utc)

        # Compute new hash for the updated ontology
        # Clear existing hash first so it gets recomputed
        updated_ontology.hash = None
        updated_ontology._compute_and_set_hash()

        # If hash computation failed and we have a parent, use parent hash as fallback
        if not updated_ontology.hash and updated_ontology.parent_hashes:
            updated_ontology.hash = updated_ontology.parent_hashes[0]

        # Sync properties to update object fields after graph changes
        updated_ontology.sync_properties_to_graph()
        return updated_ontology

    def update_ontology(self) -> None:
        """Update the current ontology with all GraphUpdate objects and clear the updates list.

        This method:
        1. Uses render_uptodate_ontology() to get an updated copy
        2. Replaces the current ontology with the updated copy
        3. Clears the ontology_updates list

        Note: Version update is deferred to aggregate_serialize() to update only once at the end.
        """
        if not self.ontology_updates:
            return

        # Get the updated ontology copy
        updated_ontology = self.render_uptodate_ontology()

        # Replace the current ontology with the updated copy
        self.current_ontology = updated_ontology

        # Clear the updates list
        self.ontology_updates_applied += self.ontology_updates
        self.ontology_updates = []

    def render_uptodate_facts(self) -> RDFGraph:
        """Create a copy of the current chunk's graph with all facts GraphUpdate objects applied.

        This method:
        1. Creates a copy of the current chunk's graph
        2. Generates SPARQL queries from all facts GraphUpdate objects
        3. Executes the queries on the copied graph
        4. Returns the updated graph copy

        Returns:
            RDFGraph: A copy of the current chunk's graph with all facts updates applied
        """
        if not self.facts_updates:
            return self.current_chunk.graph

        # Use the generalized function to update the graph
        updated_graph, _ = self.render_updated_graph(
            self.current_chunk.graph, self.facts_updates, max_triples=None
        )
        return updated_graph

    def update_facts(self) -> None:
        """Update the current chunk's graph with all facts GraphUpdate objects and clear the updates list.

        This method:
        1. Uses render_uptodate_facts() to get an updated copy
        2. Replaces the current chunk's graph with the updated copy
        3. Clears the facts_updates list
        """
        if not self.facts_updates:
            return

        # Get the updated graph copy
        updated_graph = self.render_uptodate_facts()

        # Replace the current chunk's graph with the updated copy
        self.current_chunk.graph = updated_graph

        # Clear the updates list
        self.facts_updates_applied += self.facts_updates
        self.facts_updates = []

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
    def doc_iri(self):
        """Get the document IRI.

        Returns:
            str: The document IRI.
        """
        return f"{self.current_domain}/doc/{self.doc_hid}"

    @property
    def doc_namespace(self):
        """Get the document namespace.

        Returns:
            str: The document namespace.
        """
        return iri2namespace(self.doc_iri, ontology=False)

    @property
    def ontology_id(self):
        """Get the document namespace.

        Returns:
            str: The document namespace.
        """
        return self.current_ontology.ontology_id

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
