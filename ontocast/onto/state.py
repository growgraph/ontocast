import os
from collections import defaultdict
from typing import Any

from pydantic import ConfigDict, Field

from ontocast.onto.chunk import Chunk
from ontocast.onto.constants import (
    CHUNK_NULL_IRI,
    DEFAULT_DOMAIN,
    ONTOLOGY_NULL_ID,
    ONTOLOGY_NULL_IRI,
)
from ontocast.onto.context import AgentContext, AgentType, ContextManager
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import BasePydanticModel
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import AddPrefixOp, GraphUpdate
from ontocast.util import iri2namespace, render_text_hash


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
            ontology_id=ONTOLOGY_NULL_ID,
            title="null title",
            description="null description",
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
    ontology_updates: list[GraphUpdate] = Field(
        default_factory=list,
        description="A list of graph update that improve the current ontology",
    )

    ontology_addendum: Ontology = Field(
        default_factory=lambda: Ontology(
            ontology_id=ONTOLOGY_NULL_ID,
            title="null title",
            description="null description",
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
    context_manager: ContextManager = Field(
        default_factory=ContextManager,
        description="Context manager for passing information between agents",
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

    def render_uptodate_ontology(self) -> Ontology:
        """Create a copy of the current ontology with all GraphUpdate objects applied.

        This method:
        1. Creates a copy of the current ontology
        2. Generates SPARQL queries from all GraphUpdate objects
        3. Executes the queries on the copied ontology graph
        4. Syncs properties to ensure object fields are updated
        5. Returns the updated ontology copy

        Returns:
            Ontology: A copy of the current ontology with all updates applied
        """
        if not self.ontology_updates:
            return self.current_ontology

        # Create a copy of the current ontology
        from copy import deepcopy

        updated_ontology = deepcopy(self.current_ontology)

        all_prefixes = {}
        for graph_update in self.ontology_updates:
            for op in graph_update.operations:
                if isinstance(op, AddPrefixOp):
                    all_prefixes[op.prefix] = op.namespace_uri

        # Bind prefixes to the copied ontology graph
        for prefix, uri in all_prefixes.items():
            updated_ontology.graph.bind(prefix, uri)

        # Apply each GraphUpdate to the copied ontology
        for graph_update in self.ontology_updates:
            # Generate SPARQL queries from the GraphUpdate
            queries = graph_update.generate_sparql_queries()

            # Execute each query on the copied ontology graph
            for query in queries:
                updated_ontology.graph.update(query)

        # Sync properties to update object fields after graph changes
        updated_ontology.sync_properties_to_graph()

        return updated_ontology

    def update_ontology(self) -> None:
        """Update the current ontology with all GraphUpdate objects and clear the updates list.

        This method:
        1. Uses render_uptodate_ontology() to get an updated copy
        2. Replaces the current ontology with the updated copy
        3. Clears the ontology_updates list
        """
        if not self.ontology_updates:
            return

        # Get the updated ontology copy
        updated_ontology = self.render_uptodate_ontology()

        # Replace the current ontology with the updated copy
        self.current_ontology = updated_ontology

        # Clear the updates list
        self.ontology_updates = []

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

    def set_failure(self, stage: str, reason: str, success_score: float = 0.0):
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
