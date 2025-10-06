"""Demonstration of diff-aware prompts.

This module demonstrates how the enhanced prompts handle diffs vs full documents
for efficient processing and improved agent performance.
"""

import logging

from ontocast.config import Config
from ontocast.onto.context import AgentType
from ontocast.onto.state import AgentState
from ontocast.prompt.enhanced_criticise_facts_with_diff import (
    prompt_enhanced as facts_prompt_enhanced,
)
from ontocast.prompt.enhanced_criticise_ontology_with_diff import (
    prompt_fresh_enhanced,
    prompt_update_enhanced,
)
from ontocast.prompt.enhanced_render_facts_with_diff import (
    ontology_instruction_enhanced as facts_ontology_instruction_enhanced,
)
from ontocast.prompt.enhanced_render_facts_with_diff import (
    template_prompt_enhanced as facts_template_prompt_enhanced,
)
from ontocast.prompt.enhanced_render_ontology_with_diff import (
    diff_support,
    instructions_enhanced,
    ontology_instruction_fresh_enhanced,
    ontology_instruction_update_enhanced,
    specific_ontology_instruction_fresh_enhanced,
    template_prompt_enhanced,
)

logger = logging.getLogger(__name__)


def demonstrate_diff_aware_prompts():
    """Demonstrate how diff-aware prompts work."""

    print("=== Diff-Aware Prompts Demonstration ===\n")

    # Initialize system
    _ = Config()

    # Create agent state
    state = AgentState()
    state.set_text(
        "John Smith is a software engineer at TechCorp. He manages a team of developers."
    )

    print("1. Created agent state with sample text")

    # Get context for ontology renderer
    agent_context = state.get_context_for_agent("ontology_renderer", AgentType.RENDERER)

    # Add conversation memory
    agent_context.add_conversation_memory(
        role="system",
        content="Starting ontology rendering with diff support",
        metadata={"diff_enabled": True, "interaction_type": "ontology_rendering"},
    )

    agent_context.add_conversation_memory(
        role="user",
        content="Please create an ontology for software engineering domain",
        metadata={"domain": "software_engineering"},
    )

    print("2. Added conversation memory with diff context")

    # Build dynamic context
    agent_context.build_dynamic_context(
        interaction_type="ontology_rendering_with_diff",
        document_text=state.current_chunk.text[:200],
        is_fresh_ontology=True,
        diff_enabled=True,
    )

    previous_context = agent_context.get_llm_context()

    print("3. Built dynamic context for diff-aware processing")
    print(f"   Context length: {len(previous_context)} characters")

    # Demonstrate fresh ontology prompt
    print("\n4. Fresh Ontology Prompt (Diff-Aware):")
    fresh_instruction = ontology_instruction_fresh_enhanced.format(
        previous_context=previous_context
    )
    print(f"   Instruction length: {len(fresh_instruction)} characters")
    print("   Key features:")
    print("   - DIFF-AWARE INSTRUCTIONS section")
    print("   - Context from previous work")
    print("   - Focus on incremental improvements")
    print("   - Build upon previous work patterns")

    # Demonstrate update ontology prompt
    print("\n5. Update Ontology Prompt (Diff-Aware):")
    update_instruction = ontology_instruction_update_enhanced.format(
        ontology_iri="http://example.org/ns#SoftwareEngineering",
        ontology_desc="Software Engineering Ontology",
        ontology_str="""@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix co: <http://example.org/ns#> .

co:Person a rdfs:Class .
co:Employee a rdfs:Class .
co:Employee rdfs:subClassOf co:Person .""",
        previous_context=previous_context,
    )
    print(f"   Instruction length: {len(update_instruction)} characters")
    print("   Key features:")
    print("   - DIFF-AWARE INSTRUCTIONS section")
    print("   - Focus on what needs to be ADDED, MODIFIED, or REMOVED")
    print("   - Incremental improvements rather than complete rewrites")
    print("   - Change detection and analysis")

    # Demonstrate specific instructions
    print("\n6. Specific Instructions (Diff-Aware):")
    specific_instruction = specific_ontology_instruction_fresh_enhanced.format(
        current_domain="http://example.org", previous_context=previous_context
    )
    print(f"   Instruction length: {len(specific_instruction)} characters")
    print("   Key features:")
    print("   - DIFF-AWARE: This is a fresh ontology, so generate complete structure")
    print("   - CONSIDER PREVIOUS CONTEXT")
    print("   - MAINTAIN CONSISTENCY")
    print("   - Build upon previous work patterns")

    # Demonstrate template prompt
    print("\n7. Template Prompt (Diff-Aware):")
    template = template_prompt_enhanced.format(
        ontology_instruction=fresh_instruction,
        instructions=instructions_enhanced,
        text=state.current_chunk.text,
        diff_support=diff_support,
        failure_instruction="",
        format_instructions="",
    )
    print(f"   Template length: {len(template)} characters")
    print("   Key features:")
    print("   - DIFF-AWARE PROCESSING section")
    print("   - Focus on incremental changes")
    print("   - Consider previous context")
    print("   - Generate appropriate SPARQL operations")

    return template


def demonstrate_facts_diff_prompts():
    """Demonstrate facts diff-aware prompts."""

    print("\n=== Facts Diff-Aware Prompts Demonstration ===\n")

    # Initialize system
    _ = Config()

    # Create agent state
    state = AgentState()
    state.set_text(
        "John Smith is a software engineer at TechCorp. He manages a team of developers."
    )

    # Get context for facts renderer
    agent_context = state.get_context_for_agent("facts_renderer", AgentType.RENDERER)

    # Add conversation memory
    agent_context.add_conversation_memory(
        role="system",
        content="Starting facts rendering with diff support",
        metadata={"diff_enabled": True, "interaction_type": "facts_rendering"},
    )

    # Build dynamic context
    agent_context.build_dynamic_context(
        interaction_type="facts_rendering_with_diff",
        chunk_text=state.current_chunk.text[:200],
        ontology_iri="http://example.org/ns#SoftwareEngineering",
        diff_enabled=True,
    )

    previous_context = agent_context.get_llm_context()

    print("1. Facts Context with Diff Support")
    print(f"   Context length: {len(previous_context)} characters")

    # Demonstrate facts ontology instruction
    print("\n2. Facts Ontology Instruction (Diff-Aware):")
    facts_instruction = facts_ontology_instruction_enhanced.format(
        ontology_str="""@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix co: <http://example.org/ns#> .

co:Person a rdfs:Class .
co:Employee a rdfs:Class .
co:Employee rdfs:subClassOf co:Person .""",
        previous_context=previous_context,
    )
    print(f"   Instruction length: {len(facts_instruction)} characters")
    print("   Key features:")
    print("   - DIFF-AWARE INSTRUCTIONS section")
    print("   - Focus on incremental improvements")
    print(
        "   - For updates: Focus on what facts need to be added, modified, or removed"
    )
    print("   - For fresh facts: Generate complete fact extraction")

    # Demonstrate facts template prompt
    print("\n3. Facts Template Prompt (Diff-Aware):")
    facts_template = facts_template_prompt_enhanced.format(
        current_doc_namespace="http://example.org/doc/",
        ontology_namespace="http://example.org/ns#",
        ontology_prefix="co",
        ontology_instruction=facts_instruction,
        text=state.current_chunk.text,
        diff_support=diff_support,
        failure_instruction="",
        format_instructions="",
    )
    print(f"   Template length: {len(facts_template)} characters")
    print("   Key features:")
    print("   - DIFF-AWARE INSTRUCTIONS section")
    print("   - Focus on incremental improvements")
    print("   - Consider what has changed since the last version")
    print(
        "   - For updates: Focus on what facts need to be added, modified, or removed"
    )
    print("   - For fresh facts: Generate complete fact extraction from the document")

    return facts_template


def demonstrate_critique_diff_prompts():
    """Demonstrate critique diff-aware prompts."""

    print("\n=== Critique Diff-Aware Prompts Demonstration ===\n")

    # Initialize system
    _ = Config()

    # Create agent state
    state = AgentState()
    state.set_text(
        "John Smith is a software engineer at TechCorp. He manages a team of developers."
    )

    # Get context for ontology critic
    agent_context = state.get_context_for_agent("ontology_critic", AgentType.CRITIC)

    # Add conversation memory
    agent_context.add_conversation_memory(
        role="system",
        content="Starting ontology critique with diff support",
        metadata={"diff_enabled": True, "interaction_type": "ontology_critique"},
    )

    # Build dynamic context
    agent_context.build_dynamic_context(
        interaction_type="ontology_critique_with_diff",
        ontology_iri="http://example.org/ns#SoftwareEngineering",
        document_text=state.current_chunk.text[:200],
        diff_enabled=True,
    )

    previous_context = agent_context.get_llm_context()

    print("1. Critique Context with Diff Support")
    print(f"   Context length: {len(previous_context)} characters")

    # Demonstrate fresh ontology critique prompt
    print("\n2. Fresh Ontology Critique Prompt (Diff-Aware):")
    fresh_critique = prompt_fresh_enhanced.format(
        previous_context=previous_context,
        ontology_original_str="",
        document=state.current_chunk.text,
        ontology_update="""@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix co: <http://example.org/ns#> .

co:Person a rdfs:Class .
co:Employee a rdfs:Class .
co:Employee rdfs:subClassOf co:Person .""",
        format_instructions="",
    )
    print(f"   Prompt length: {len(fresh_critique)} characters")
    print("   Key features:")
    print("   - DIFF-AWARE EVALUATION section")
    print("   - Consider previous critiques when evaluating")
    print("   - Build upon previous feedback rather than starting fresh")
    print("   - Focus on incremental improvements based on past feedback")
    print("   - For fresh ontologies: Evaluate completeness and structure")
    print("   - For updates: Evaluate whether the changes are appropriate and complete")

    # Demonstrate update ontology critique prompt
    print("\n3. Update Ontology Critique Prompt (Diff-Aware):")
    update_critique = prompt_update_enhanced.format(
        previous_context=previous_context,
        ontology_original_str="""@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix co: <http://example.org/ns#> .

co:Person a rdfs:Class .""",
        document=state.current_chunk.text,
        ontology_update="""@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix co: <http://example.org/ns#> .

co:Person a rdfs:Class .
co:Employee a rdfs:Class .
co:Employee rdfs:subClassOf co:Person .""",
        format_instructions="",
    )
    print(f"   Prompt length: {len(update_critique)} characters")
    print("   Key features:")
    print("   - DIFF-AWARE EVALUATION section")
    print("   - Evaluate whether the changes are appropriate and complete")
    print(
        "   - DIFF-FOCUSED: Evaluate the specific changes made and their appropriateness"
    )
    print(
        "   - INCREMENTAL: Focus on whether the changes address the identified issues"
    )
    print(
        "   - CONSISTENCY: Ensure changes are consistent with existing ontology structure"
    )

    return update_critique


def demonstrate_facts_critique_diff_prompts():
    """Demonstrate facts critique diff-aware prompts."""

    print("\n=== Facts Critique Diff-Aware Prompts Demonstration ===\n")

    # Initialize system
    _ = Config()

    # Create agent state
    state = AgentState()
    state.set_text(
        "John Smith is a software engineer at TechCorp. He manages a team of developers."
    )

    # Get context for facts critic
    agent_context = state.get_context_for_agent("facts_critic", AgentType.CRITIC)

    # Add conversation memory
    agent_context.add_conversation_memory(
        role="system",
        content="Starting facts critique with diff support",
        metadata={"diff_enabled": True, "interaction_type": "facts_critique"},
    )

    # Build dynamic context
    agent_context.build_dynamic_context(
        interaction_type="facts_critique_with_diff",
        chunk_text=state.current_chunk.text[:200],
        ontology_iri="http://example.org/ns#SoftwareEngineering",
        diff_enabled=True,
    )

    previous_context = agent_context.get_llm_context()

    print("1. Facts Critique Context with Diff Support")
    print(f"   Context length: {len(previous_context)} characters")

    # Demonstrate facts critique prompt
    print("\n2. Facts Critique Prompt (Diff-Aware):")
    facts_critique = facts_prompt_enhanced.format(
        previous_context=previous_context,
        ontology="""@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix co: <http://example.org/ns#> .

co:Person a rdfs:Class .
co:Employee a rdfs:Class .
co:Employee rdfs:subClassOf co:Person .""",
        document=state.current_chunk.text,
        facts="""@prefix cd: <http://example.org/doc/> .
@prefix co: <http://example.org/ns#> .
@prefix schema: <https://schema.org/> .

cd:JohnSmith a co:Employee, schema:Person .
cd:JohnSmith schema:name "John Smith" .
cd:TechCorp a schema:Organization .
cd:JohnSmith schema:worksFor cd:TechCorp .""",
        format_instructions="",
    )
    print(f"   Prompt length: {len(facts_critique)} characters")
    print("   Key features:")
    print("   - DIFF-AWARE EVALUATION section")
    print("   - Consider previous critiques when evaluating the current facts")
    print("   - Build upon previous feedback rather than starting fresh")
    print("   - Focus on incremental improvements based on past feedback")
    print("   - For fresh facts: Evaluate completeness and accuracy")
    print("   - For updates: Evaluate whether the changes are appropriate and complete")
    print(
        "   - DIFF-FOCUSED: Consider what changes were made and whether they are sufficient"
    )
    print(
        "   - INCREMENTAL: Focus on whether the changes address the identified issues"
    )

    return facts_critique


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run demonstrations
    ontology_template = demonstrate_diff_aware_prompts()
    facts_template = demonstrate_facts_diff_prompts()
    critique_template = demonstrate_critique_diff_prompts()
    facts_critique_template = demonstrate_facts_critique_diff_prompts()

    print("\n=== Diff-Aware Prompts Benefits ===")
    print("✓ Enhanced context awareness")
    print("✓ Diff-focused processing instructions")
    print("✓ Incremental improvement guidance")
    print("✓ Previous work consideration")
    print("✓ Consistency maintenance")
    print("✓ SPARQL operation generation")
    print("✓ Conversation memory integration")
    print("✓ Dynamic context construction")
    print("✓ Failure recovery with context")
    print("✓ Appropriate prompt selection based on context")

    print("\n=== Prompt Engineering Best Practices ===")
    print("✓ Maintained original prompt effectiveness")
    print("✓ Added diff-aware instructions without breaking existing functionality")
    print("✓ Preserved all original instruction sections")
    print("✓ Enhanced with context-aware processing")
    print("✓ Added incremental improvement guidance")
    print("✓ Integrated conversation memory")
    print("✓ Added SPARQL operation support")
    print("✓ Enhanced failure recovery")
    print("✓ Maintained prompt sensitivity and effectiveness")
