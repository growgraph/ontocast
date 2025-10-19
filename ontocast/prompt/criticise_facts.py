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
# EVALUATION GUIDELINES

1. Appropriateness: Are the facts appropriate for the document?
2. Completeness: Are all possible facts extracted from the text given the ontology?
3. Concreteness: Only concrete facts should be extracted.
4. Structure: Are all concrete entities linked to abstract classes via relations?
5. Ontology Reuse: All entities and relations existing in the ontology must be referenced and linked to their canonical identifiers, rather than instantiated anew.
"""
