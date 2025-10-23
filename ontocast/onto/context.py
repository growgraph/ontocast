"""Context passing system for agent-based workflow.

This module provides functionality for passing context between agents,
enabling memory and incremental processing.
"""

import logging
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ontocast.onto.sparql_models import SPARQLOperationModel
from ontocast.tool.graph_version_manager import GraphVersion

logger = logging.getLogger(__name__)

summary_template = """
ONTOLOGY CONTEXT:
{ontology_context}

FACTS CONTEXT:
{facts_context}

CONTEXT METADATA:
- Agent type: `{agent_type}`
- Timestamp: {context_timestamp}
- Additional metadata: {context_metadata}
"""


class AgentType(StrEnum):
    """Enumeration of agent types for type safety."""

    RENDERER_FACTS = "renderer_facts"
    RENDERER_ONTOLOGY = "renderer_ontology"
    CRITIC_FACTS = "critic_facts"
    CRITIC_ONTOLOGY = "critic_ontology"
    AGGREGATOR = "aggregator"
    CONVERTER = "converter"
    CHUNKER = "chunker"


class Role(StrEnum):
    """Enumeration of conversation roles for type safety."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class AgentContext(BaseModel):
    """Context information passed between agents.

    This class encapsulates all the context information that agents
    need to build upon previous work rather than starting fresh.
    """

    # Agent identification
    agent_type: AgentType = Field(description="Type of agent for type safety")

    # Previous work context
    previous_ontology_version: GraphVersion | None = Field(
        default=None, description="Previous ontology version if available"
    )
    previous_facts_version: GraphVersion | None = Field(
        default=None, description="Previous facts version if available"
    )

    # Previous operations (append-only for performance)
    previous_ontology_operations: list[SPARQLOperationModel] = Field(
        default_factory=list, description="Previous ontology SPARQL operations"
    )
    previous_facts_operations: list[SPARQLOperationModel] = Field(
        default_factory=list, description="Previous facts SPARQL operations"
    )

    # Previous critiques (append-only for consistency)
    previous_ontology_critique: dict[str, Any] | None = Field(
        default=None, description="Previous ontology critique if available"
    )
    previous_facts_critique: dict[str, Any] | None = Field(
        default=None, description="Previous facts critique if available"
    )

    # Context metadata (append-only strategy)
    context_timestamp: datetime = Field(
        default_factory=datetime.now, description="When this context was created"
    )
    context_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context metadata"
    )

    # Conversation memory for LLM calls
    conversation_memory: list[dict[str, Any]] = Field(
        default_factory=list, description="Conversation history for LLM context"
    )

    # Dynamic context construction
    dynamic_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamically constructed context for current interaction",
    )

    def get_ontology_context_summary(self) -> str:
        """Get a summary of ontology context for prompts."""
        if not self.previous_ontology_version:
            return "No previous ontology context available."

        summary = f"Previous ontology version: {self.previous_ontology_version.id}\n"
        summary += f"Previous ontology size: {self.previous_ontology_version.get_size()} triples\n"
        summary += f"Previous ontology operations: {len(self.previous_ontology_operations)} SPARQL operations\n"

        if self.previous_ontology_critique:
            summary += f"Previous ontology critique score: {self.previous_ontology_critique.get('score', 'N/A')}\n"
            summary += f"Previous ontology critique issues: {self.previous_ontology_critique.get('issues', 'None')}\n"

        return summary

    def get_facts_context_summary(self) -> str:
        """Get a summary of facts context for prompts."""
        if not self.previous_facts_version:
            return "No previous facts context available."

        summary = f"Previous facts version: {self.previous_facts_version.id}\n"
        summary += (
            f"Previous facts size: {self.previous_facts_version.get_size()} triples\n"
        )
        summary += f"Previous facts operations: {len(self.previous_facts_operations)} SPARQL operations\n"

        if self.previous_facts_critique:
            summary += f"Previous facts critique score: {self.previous_facts_critique.get('score', 'N/A')}\n"
            summary += f"Previous facts critique issues: {self.previous_facts_critique.get('issues', 'None')}\n"

        return summary

    def get_full_context_summary(self) -> str:
        """Get a complete context summary for prompts."""
        ontology_context = self.get_ontology_context_summary()
        facts_context = self.get_facts_context_summary()

        summary = summary_template.format(
            facts_context=facts_context,
            ontology_context=ontology_context,
            agent_type=self.agent_type.value,
            context_timestamp=self.context_timestamp.isoformat(),
            context_metadata=self.context_metadata,
        )

        return summary

    def add_conversation_memory(
        self, role: Role, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a conversation entry to memory (append-only strategy).

        Args:
            role: Role of the speaker (Role.SYSTEM, Role.USER, Role.ASSISTANT)
            content: Content of the message
            metadata: Optional metadata for the conversation entry
        """
        entry = {
            "role": role.value,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.conversation_memory.append(entry)
        logger.debug(f"Added conversation memory for {self.agent_type}: {role.value}")

    def get_conversation_context(self, max_entries: int = 10) -> str:
        """Get conversation context for LLM calls.

        Args:
            max_entries: Maximum number of conversation entries to include

        Returns:
            str: Formatted conversation context
        """
        if not self.conversation_memory:
            return "No conversation history available."

        # Get the most recent entries (append-only strategy preserves order)
        recent_entries = self.conversation_memory[-max_entries:]

        context = "CONVERSATION HISTORY:\n"
        for entry in recent_entries:
            context += f"{entry['role'].upper()}: {entry['content']}\n"
            if entry.get("metadata"):
                context += f"  Metadata: {entry['metadata']}\n"
            context += "\n"

        return context

    def build_dynamic_context(self, interaction_type: str, **kwargs) -> dict[str, Any]:
        """Build dynamic context for current interaction.

        Args:
            interaction_type: Type of interaction (render, critique, etc.)
            **kwargs: Additional context parameters

        Returns:
            dict[str, Any]: Dynamic context for the interaction
        """
        dynamic_context = {
            "interaction_type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "agent_type": self.agent_type,
            "context_summary": self.get_full_context_summary(),
            "conversation_context": self.get_conversation_context(),
            **kwargs,
        }

        # Update the dynamic context
        self.dynamic_context.update(dynamic_context)

        return dynamic_context

    def get_llm_context(self) -> str:
        """Get complete context for LLM calls including conversation memory.

        Returns:
            str: Complete context for LLM calls
        """

        return (
            f"{self.get_full_context_summary()}\n\n"
            f"{self.get_conversation_context()}\n\n"
            f"DYNAMIC CONTEXT:\n{self.dynamic_context}"
        )


class ContextManager(BaseModel):
    """Manages context passing between agents.

    This class handles the creation, storage, and retrieval of context
    information for agent-based workflows.
    """

    context_history: list[AgentContext] = Field(
        default_factory=list, description="History of agent contexts"
    )
    current_context: AgentContext | None = Field(
        default=None, description="Current active context"
    )

    def __init__(self, **kwargs):
        """Initialize the context manager."""
        super().__init__(**kwargs)

    def create_context(
        self,
        agent_type: AgentType,
        previous_ontology_version: GraphVersion | None = None,
        previous_facts_version: GraphVersion | None = None,
        previous_ontology_operations: list[SPARQLOperationModel] | None = None,
        previous_facts_operations: list[SPARQLOperationModel] | None = None,
        previous_ontology_critique: dict[str, Any] | None = None,
        previous_facts_critique: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentContext:
        """Create a new context for an agent.

        Args:
            agent_type: Name of the agent creating the context.
            agent_type: Type of agent (renderer, critic, etc.).
            previous_ontology_version: Previous ontology version if available.
            previous_facts_version: Previous facts version if available.
            previous_ontology_operations: Previous ontology operations if available.
            previous_facts_operations: Previous facts operations if available.
            previous_ontology_critique: Previous ontology critique if available.
            previous_facts_critique: Previous facts critique if available.
            metadata: Additional metadata for the context.

        Returns:
            AgentContext: The created context.
        """
        context = AgentContext(
            agent_type=agent_type,
            previous_ontology_version=previous_ontology_version,
            previous_facts_version=previous_facts_version,
            previous_ontology_operations=previous_ontology_operations or [],
            previous_facts_operations=previous_facts_operations or [],
            previous_ontology_critique=previous_ontology_critique,
            previous_facts_critique=previous_facts_critique,
            context_metadata=metadata or {},
        )

        self.context_history.append(context)
        self.current_context = context

        logger.info(f"Created context for {agent_type} ({agent_type})")
        return context

    def get_current_context(self) -> AgentContext | None:
        """Get the current context.

        Returns:
            AgentContext | None: The current context, or None if not set.
        """
        return self.current_context

    def get_context_history(self) -> list[AgentContext]:
        """Get the full context history.

        Returns:
            list[AgentContext]: The complete context history.
        """
        return self.context_history

    def get_context_by_agent(self, agent_type: AgentType) -> list[AgentContext]:
        """Get context history for a specific agent.

        Args:
            agent_type: Name of the agent to get context for.

        Returns:
            list[AgentContext]: Context history for the specified agent.
        """
        return [ctx for ctx in self.context_history if ctx.agent_type == agent_type]

    def get_latest_context_by_agent(self, agent_type: AgentType) -> AgentContext | None:
        """Get the latest context for a specific agent.

        Args:
            agent_type: Name of the agent to get latest context for.

        Returns:
            AgentContext | None: The latest context for the specified agent, or None.
        """
        agent_contexts = self.get_context_by_agent(agent_type)
        return agent_contexts[-1] if agent_contexts else None

    def update_context(
        self,
        agent_type: AgentType,
        ontology_version: GraphVersion | None = None,
        facts_version: GraphVersion | None = None,
        ontology_operations: list[SPARQLOperationModel] | None = None,
        facts_operations: list[SPARQLOperationModel] | None = None,
        ontology_critique: dict[str, Any] | None = None,
        facts_critique: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentContext:
        """Update the current context with new information.

        Args:
            agent_type: Name of the agent updating the context.
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
        if not self.current_context:
            # Create new context if none exists
            return self.create_context(
                agent_type=agent_type,
                previous_ontology_version=ontology_version,
                previous_facts_version=facts_version,
                previous_ontology_operations=ontology_operations,
                previous_facts_operations=facts_operations,
                previous_ontology_critique=ontology_critique,
                previous_facts_critique=facts_critique,
                metadata=metadata,
            )

        # Update existing context
        if ontology_version:
            self.current_context.previous_ontology_version = ontology_version
        if facts_version:
            self.current_context.previous_facts_version = facts_version
        if ontology_operations:
            self.current_context.previous_ontology_operations = ontology_operations
        if facts_operations:
            self.current_context.previous_facts_operations = facts_operations
        if ontology_critique:
            self.current_context.previous_ontology_critique = ontology_critique
        if facts_critique:
            self.current_context.previous_facts_critique = facts_critique
        if metadata:
            self.current_context.context_metadata.update(metadata)

        self.current_context.context_timestamp = datetime.now()

        logger.info(f"Updated context for {agent_type}")
        return self.current_context

    def clear_context(self):
        """Clear the current context."""
        self.current_context = None
        logger.info("Cleared current context")

    def clear_history(self):
        """Clear the entire context history."""
        self.context_history = []
        self.current_context = None
        logger.info("Cleared context history")
