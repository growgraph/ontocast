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
from ontocast.onto.enum import Status, WorkflowNode
from ontocast.onto.model import BasePydanticModel
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
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
    failure_stage: str | None = None
    failure_reason: str | None = None
    success_score: float = 0.0
    status: Status = Status.SUCCESS
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
