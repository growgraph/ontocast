"""Enhanced facts criticism prompts with context passing and memory support.

This module provides enhanced prompt templates for facts criticism that support
context passing, memory, and improved critique quality.
"""

from .common import system_preamble_semantic

template_prompt = """
{preamble}

{evaluation_instruction}

{user_instruction}

{ontology_chapter}

{facts_chapter}

{text_chapter}

{output_instruction}

{format_instructions}
"""

preamble = f"""
{system_preamble_semantic}
You are given an ontology, a text and a semantic graph of facts, generated from the text (guided by ontology).
Following evaluation guidelines provide concrete suggestions for improvement of the extracted facts graph with respect to provided text and ontology.
"""


evaluation_instruction = """\n\n
# CRITICAL EVALUATION GUIDELINES

1.  Appropriateness: Is the ontology appropriate for the document?
2.  Completeness and Consistency: Is the ontology complete and consistent? **Identify missing classes/properties needed to represent facts in the text.**
3.  Abstraction: Only abstract classes and properties should be present in the ontology. **Ensure no specific individuals (instances) are mistakenly defined.**
4.  Domain Coverage: Ontology should include implicit domain-specific abstractions and relationships **needed for knowledge extraction**, even if not explicitly mentioned in the text.
5.  Structure: Check hierarchies, property definitions, redundancy, and appropriate granularity. **Identify specific structural patterns (e.g., circular hierarchies, redundant properties) that violate best practices.**
"""

output_extra = """\n\n
# OUTPUT INSTRUCTION
Provide a constructive critique in the required JSON format.

The **`actionable_ontology_fixes`** field must contain an itemized list of specific, actionable suggestions for ontology modification (e.g., ADD CLASS, REMOVE PROPERTY).
* **Each suggestion must reference the specific text fragment** that justifies the addition or modification (for completeness/appropriateness issues).
* **Each suggestion must use concrete ontology syntax** where applicable (e.g., 'ADD the class `ex:NewClass` to cover the concept of X mentioned in the text: "...text snippet..."').

The **`systemic_critique_summary`** field must contain a general, non-itemized summary addressing high-level deficiencies identified by Criteria 4 and 5 (Domain Coverage, Structure).
"""
