"""Context passing system for agent-based workflow.

This module provides functionality for passing context between agents,
enabling memory and incremental processing.
"""

import logging
from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ontocast.onto.sparql_models import SPARQLOperationModel
from ontocast.tool.graph_version_manager import GraphVersion

logger = logging.getLogger(__name__)


class AgentType(StrEnum):
    """Enumeration of agent types for type safety."""

    RENDERER = "renderer"
    CRITIC = "critic"
    AGGREGATOR = "aggregator"
    CONVERTER = "converter"
    CHUNKER = "chunker"


class AgentContext(BaseModel):
    """Context information passed between agents.

    This class encapsulates all the context information that agents
    need to build upon previous work rather than starting fresh.
    """

    # Agent identification
    agent_name: str = Field(description="Name of the agent")
    agent_type: AgentType = Field(description="Type of agent for type safety")

    # Previous work context
    previous_ontology_version: Optional[GraphVersion] = Field(
        default=None, description="Previous ontology version if available"
    )
    previous_facts_version: Optional[GraphVersion] = Field(
        default=None, description="Previous facts version if available"
    )

    # Previous operations (append-only for performance)
    previous_ontology_operations: List[SPARQLOperationModel] = Field(
        default_factory=list, description="Previous ontology SPARQL operations"
    )
    previous_facts_operations: List[SPARQLOperationModel] = Field(
        default_factory=list, description="Previous facts SPARQL operations"
    )

    # Previous critiques (append-only for consistency)
    previous_ontology_critique: Optional[Dict[str, Any]] = Field(
        default=None, description="Previous ontology critique if available"
    )
    previous_facts_critique: Optional[Dict[str, Any]] = Field(
        default=None, description="Previous facts critique if available"
    )

    # Context metadata (append-only strategy)
    context_timestamp: datetime = Field(
        default_factory=datetime.now, description="When this context was created"
    )
    context_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context metadata"
    )

    # Conversation memory for LLM calls
    conversation_memory: List[Dict[str, Any]] = Field(
        default_factory=list, description="Conversation history for LLM context"
    )

    # Dynamic context construction
    dynamic_context: Dict[str, Any] = Field(
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

        return f"""
ONTOLOGY CONTEXT:
{ontology_context}

FACTS CONTEXT:
{facts_context}

CONTEXT METADATA:
- Agent: {self.agent_name} ({self.agent_type})
- Timestamp: {self.context_timestamp.isoformat()}
- Additional metadata: {self.context_metadata}
"""

    def add_conversation_memory(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a conversation entry to memory (append-only strategy).

        Args:
            role: Role of the speaker (user, assistant, system)
            content: Content of the message
            metadata: Optional metadata for the conversation entry
        """
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.conversation_memory.append(entry)
        logger.debug(f"Added conversation memory for {self.agent_name}: {role}")

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

    def build_dynamic_context(self, interaction_type: str, **kwargs) -> Dict[str, Any]:
        """Build dynamic context for current interaction.

        Args:
            interaction_type: Type of interaction (render, critique, etc.)
            **kwargs: Additional context parameters

        Returns:
            Dict[str, Any]: Dynamic context for the interaction
        """
        dynamic_context = {
            "interaction_type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "context_summary": self.get_full_context_summary(),
            "conversation_context": self.get_conversation_context(),
            **kwargs,
        }

        # Update the dynamic context
        self.dynamic_context.update(dynamic_context)

        logger.debug(f"Built dynamic context for {self.agent_name}: {interaction_type}")
        return dynamic_context

    def get_llm_context(self) -> str:
        """Get complete context for LLM calls including conversation memory.

        Returns:
            str: Complete context for LLM calls
        """
        return f"""
{self.get_full_context_summary()}

{self.get_conversation_context()}

DYNAMIC CONTEXT:
{self.dynamic_context}
"""


class ContextManager:
    """Manages context passing between agents.

    This class handles the creation, storage, and retrieval of context
    information for agent-based workflows.
    """

    def __init__(self):
        """Initialize the context manager."""
        self.context_history: List[AgentContext] = []
        self.current_context: Optional[AgentContext] = None

    def create_context(
        self,
        agent_name: str,
        agent_type: AgentType,
        previous_ontology_version: Optional[GraphVersion] = None,
        previous_facts_version: Optional[GraphVersion] = None,
        previous_ontology_operations: Optional[List[SPARQLOperationModel]] = None,
        previous_facts_operations: Optional[List[SPARQLOperationModel]] = None,
        previous_ontology_critique: Optional[Dict[str, Any]] = None,
        previous_facts_critique: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentContext:
        """Create a new context for an agent.

        Args:
            agent_name: Name of the agent creating the context.
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
            agent_name=agent_name,
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

        logger.info(f"Created context for {agent_name} ({agent_type})")
        return context

    def get_current_context(self) -> Optional[AgentContext]:
        """Get the current context.

        Returns:
            Optional[AgentContext]: The current context, or None if not set.
        """
        return self.current_context

    def get_context_history(self) -> List[AgentContext]:
        """Get the full context history.

        Returns:
            List[AgentContext]: The complete context history.
        """
        return self.context_history

    def get_context_by_agent(self, agent_name: str) -> List[AgentContext]:
        """Get context history for a specific agent.

        Args:
            agent_name: Name of the agent to get context for.

        Returns:
            List[AgentContext]: Context history for the specified agent.
        """
        return [ctx for ctx in self.context_history if ctx.agent_name == agent_name]

    def get_latest_context_by_agent(self, agent_name: str) -> Optional[AgentContext]:
        """Get the latest context for a specific agent.

        Args:
            agent_name: Name of the agent to get latest context for.

        Returns:
            Optional[AgentContext]: The latest context for the specified agent, or None.
        """
        agent_contexts = self.get_context_by_agent(agent_name)
        return agent_contexts[-1] if agent_contexts else None

    def update_context(
        self,
        agent_name: str,
        ontology_version: Optional[GraphVersion] = None,
        facts_version: Optional[GraphVersion] = None,
        ontology_operations: Optional[List[SPARQLOperationModel]] = None,
        facts_operations: Optional[List[SPARQLOperationModel]] = None,
        ontology_critique: Optional[Dict[str, Any]] = None,
        facts_critique: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentContext:
        """Update the current context with new information.

        Args:
            agent_name: Name of the agent updating the context.
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
                agent_name=agent_name,
                agent_type=AgentType.RENDERER,
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

        logger.info(f"Updated context for {agent_name}")
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
