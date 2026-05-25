from .common import system_preamble_semantic

template_prompt = """
{preamble}

{facts_instruction}

{user_instruction}

{ontology_chapter}

{text_chapter}

{fact_chapter}

{improvement_instruction}

{output_instruction}

{format_instructions}
"""

preamble = f"""
{system_preamble_semantic}
Generate semantic triples representing facts (not abstract entities) based on provided domain ontology.
"""

improvement_instruction_template = """\n\n
# IMPROVEMENT INSTRUCTION

The current iteration of the graph of factual triples has been reviewed by Critic, who provided suggestions for improvement.

CRITICAL: You are the final decision-maker. Critic's suggestions are advisory, not mandatory. Think independently.

Your task is to critically evaluate and improve the triples:

1. Independently verify each suggestion - Before implementing ANY suggestion, verify it against:
   - The original source text (does it accurately reflect what's written?)
   - The OPERATIONAL GUIDELINES (does it follow the rules?)
   - The domain ontology (does it use entities correctly?)
   - Logical consistency (does it make semantic sense?)

2. Implement only valid improvements - Apply suggestions that are demonstrably correct and enhance accuracy or completeness. If uncertain, prioritize faithfulness to the source text.

3. Actively reject flawed suggestions - If a suggestion is:
   - Factually incorrect (contradicts the source text)
   - Violates OPERATIONAL GUIDELINES
   - Would introduce errors or degrade quality
   - Based on misunderstanding of the ontology
   
   Then REJECT it and briefly explain why in your response.

4. Think beyond the critique - Critic may have:
   - Missed issues entirely
   - Identified patterns but not all instances
   - Focused on some aspects while overlooking others
   
   Proactively identify and fix additional problems not mentioned in the critique.

5. Verify every change - Before finalizing, double-check that:
   - Each triple accurately represents information from the source text
   - Existing ontology entities are used instead of creating new cd: entities
   - No ontology-prefixed entity was invented or renamed
   - All OPERATIONAL GUIDELINES are satisfied
   - The overall graph is more complete and accurate than before

Your goal: Produce the most accurate representation of the source text, not to satisfy Critic.
{suggestions_instruction}
"""
