"""Demonstration of the diff generation and application system.

This module demonstrates how the diff system works for efficient
graph updates and context passing between agents.
"""

import logging
from datetime import datetime

from ontocast.config import Config
from ontocast.onto.context import AgentType, Role
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool.graph_diff import DiffOperation, DiffTool, GraphDiff, TripleDiff
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def demonstrate_diff_generation():
    """Demonstrate diff generation between graph versions."""

    print("=== Diff Generation System Demo ===\n")

    # Create sample graphs
    source_graph = RDFGraph()
    source_graph.add_triple(
        "http://example.org/ns#Person",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )
    source_graph.add_triple(
        "http://example.org/ns#Employee",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )
    source_graph.add_triple(
        "http://example.org/ns#Employee",
        "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "http://example.org/ns#Person",
    )

    target_graph = RDFGraph()
    target_graph.add_triple(
        "http://example.org/ns#Person",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )
    target_graph.add_triple(
        "http://example.org/ns#Employee",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )
    target_graph.add_triple(
        "http://example.org/ns#Employee",
        "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "http://example.org/ns#Person",
    )
    # Add new triple
    target_graph.add_triple(
        "http://example.org/ns#Manager",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )
    target_graph.add_triple(
        "http://example.org/ns#Manager",
        "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "http://example.org/ns#Employee",
    )

    print("1. Created sample graphs:")
    print(f"   Source graph: {len(source_graph)} triples")
    print(f"   Target graph: {len(target_graph)} triples")

    # Generate diff
    diff_tool = DiffTool()
    diff = diff_tool.generate_diff(
        source_graph=source_graph,
        target_graph=target_graph,
        source_version_id="v1.0",
        target_version_id="v1.1",
        context_metadata={
            "agent": "ontology_renderer",
            "interaction_type": "ontology_update",
            "timestamp": datetime.now().isoformat(),
        },
    )

    print("\n2. Generated diff:")
    print(f"   Diff ID: {diff.diff_id}")
    print(f"   Added triples: {diff.added_triples}")
    print(f"   Removed triples: {diff.removed_triples}")
    print(f"   Modified triples: {diff.modified_triples}")
    print(f"   Unchanged triples: {diff.unchanged_triples}")

    print("\n3. Diff summary:")
    print(diff.get_summary())

    print("\n4. SPARQL operations:")
    sparql_ops = diff.get_sparql_operations()
    for i, op in enumerate(sparql_ops, 1):
        print(f"   {i}. {op}")

    print(f"\n5. Changed subjects: {diff.get_changed_subjects()}")
    print(f"   Changed predicates: {diff.get_changed_predicates()}")

    return diff


def demonstrate_diff_application():
    """Demonstrate applying diffs to graphs."""

    print("\n=== Diff Application Demo ===\n")

    # Create a base graph
    base_graph = RDFGraph()
    base_graph.add_triple(
        "http://example.org/ns#Person",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )

    print("1. Base graph:")
    print(f"   Triples: {len(base_graph)}")
    for triple in base_graph:
        print(f"   {triple[0]} {triple[1]} {triple[2]}")

    # Create a diff
    diff_tool = DiffTool()

    # Create target graph
    target_graph = RDFGraph()
    target_graph.add_triple(
        "http://example.org/ns#Person",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )
    target_graph.add_triple(
        "http://example.org/ns#Employee",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )
    target_graph.add_triple(
        "http://example.org/ns#Employee",
        "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "http://example.org/ns#Person",
    )

    # Generate diff
    diff = diff_tool.generate_diff(
        source_graph=base_graph,
        target_graph=target_graph,
        source_version_id="base",
        target_version_id="updated",
    )

    print(f"\n2. Generated diff with {diff.added_triples} additions")

    # Apply diff
    updated_graph = diff_tool.apply_diff(base_graph, diff)

    print("\n3. Applied diff:")
    print(f"   Updated graph triples: {len(updated_graph)}")
    for triple in updated_graph:
        print(f"   {triple[0]} {triple[1]} {triple[2]}")

    return updated_graph


def demonstrate_diff_with_context():
    """Demonstrate diff system with agent context."""

    print("\n=== Diff System with Agent Context Demo ===\n")

    # Initialize system
    config = Config()
    toolbox = ToolBox(config)

    # Create agent state
    state = AgentState()
    state.set_text(
        "John Smith is a software engineer at TechCorp. He manages a team of developers."
    )

    print("1. Created agent state with sample text")

    # Get context for ontology renderer
    agent_context = state.get_context_for_agent(AgentType.RENDERER_FACTS)

    # Add conversation memory about diff processing
    agent_context.add_conversation_memory(
        role=Role.SYSTEM,
        content="Starting ontology rendering with diff support",
        metadata={"diff_enabled": True, "interaction_type": "ontology_rendering"},
    )

    agent_context.add_conversation_memory(
        role=Role.USER,
        content="Please create an ontology for software engineering domain with management hierarchy",
        metadata={"domain": "software_engineering", "hierarchy": "management"},
    )

    agent_context.add_conversation_memory(
        role=Role.ASSISTANT,
        content="I'll create an ontology with Person, Employee, and Manager classes with appropriate relationships",
        metadata={
            "response_type": "acknowledgment",
            "classes": ["Person", "Employee", "Manager"],
        },
    )

    print("2. Added conversation memory with diff context")
    print(f"   Conversation entries: {len(agent_context.conversation_memory)}")

    # Simulate diff generation
    source_graph = RDFGraph()
    source_graph.add_triple(
        "http://example.org/ns#Person",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )

    target_graph = RDFGraph()
    target_graph.add_triple(
        "http://example.org/ns#Person",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )
    target_graph.add_triple(
        "http://example.org/ns#Employee",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )
    target_graph.add_triple(
        "http://example.org/ns#Manager",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    )

    # Generate diff
    diff = toolbox.diff_tool.generate_diff(
        source_graph=source_graph,
        target_graph=target_graph,
        source_version_id="v1.0",
        target_version_id="v1.1",
        context_metadata={
            "agent": "ontology_renderer",
            "interaction_type": "ontology_update",
            "conversation_context": agent_context.get_conversation_context(),
        },
    )

    print("\n3. Generated diff with context:")
    print(f"   Diff ID: {diff.diff_id}")
    print(f"   Added triples: {diff.added_triples}")
    print(f"   Context metadata: {diff.context_metadata}")

    # Add diff information to conversation memory
    agent_context.add_conversation_memory(
        role=Role.SYSTEM,
        content=f"Generated diff with {diff.added_triples} additions",
        metadata={
            "diff_id": diff.diff_id,
            "added_triples": diff.added_triples,
            "context_metadata": diff.context_metadata,
        },
    )

    print("\n4. Added diff information to conversation memory")
    print(f"   Total conversation entries: {len(agent_context.conversation_memory)}")

    # Show complete LLM context
    llm_context = agent_context.get_llm_context()
    print(f"\n5. Complete LLM context length: {len(llm_context)} characters")
    print("   Context includes:")
    print("   - Full context summary")
    print("   - Conversation history")
    print("   - Dynamic context")
    print("   - Diff information")

    return diff


def demonstrate_diff_merging():
    """Demonstrate merging multiple diffs."""

    print("\n=== Diff Merging Demo ===\n")

    diff_tool = DiffTool()

    # Create multiple diffs
    diff1 = GraphDiff(
        diff_id="diff1",
        source_version_id="v1.0",
        target_version_id="v1.1",
        triple_diffs=[
            TripleDiff(
                subject="http://example.org/ns#Employee",
                predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                object="http://www.w3.org/2000/01/rdf-schema#Class",
                operation=DiffOperation.ADD,
            ),
        ],
        added_triples=1,
        removed_triples=0,
        modified_triples=0,
        unchanged_triples=0,
    )

    diff2 = GraphDiff(
        diff_id="diff2",
        source_version_id="v1.1",
        target_version_id="v1.2",
        triple_diffs=[
            TripleDiff(
                subject="http://example.org/ns#Manager",
                predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                object="http://www.w3.org/2000/01/rdf-schema#Class",
                operation=DiffOperation.ADD,
            ),
        ],
        added_triples=1,
        removed_triples=0,
        modified_triples=0,
        unchanged_triples=0,
    )

    print("1. Created two diffs:")
    print(f"   Diff 1: {diff1.added_triples} additions")
    print(f"   Diff 2: {diff2.added_triples} additions")

    # Merge diffs
    merged_diff = diff_tool.merge_diffs([diff1, diff2])

    print("\n2. Merged diff:")
    print(f"   Total additions: {merged_diff.added_triples}")
    print(f"   Total triple diffs: {len(merged_diff.triple_diffs)}")

    return merged_diff


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run demonstrations
    diff = demonstrate_diff_generation()
    updated_graph = demonstrate_diff_application()
    context_diff = demonstrate_diff_with_context()
    merged_diff = demonstrate_diff_merging()

    print("\n=== Diff System Benefits ===")
    print("✓ Efficient graph updates with minimal data transfer")
    print("✓ Incremental processing instead of full document processing")
    print("✓ Context-aware diff generation")
    print("✓ SPARQL operation generation for database updates")
    print("✓ Diff merging for complex workflows")
    print("✓ Conversation memory integration")
    print("✓ Agent context preservation")
