"""Enhanced facts criticism prompts with context passing and memory support.

This module provides enhanced prompt templates for facts criticism that support
context passing, memory, and improved critique quality.
"""

from .common import system_preamble_semantic

template_prompt = """
{preamble}

{intro_instruction}

{facts_criteria}

{user_instruction}

{ontology_chapter}

{facts_chapter}

{document_chapter}
"""

system_preamble = system_preamble_semantic

intro_instruction = """
You are given an ontology, a text and facts in the form of semantic triples, extracted from the text (guided by ontology).
You task is to provide a constructive critique of the extracted facts with respect to provided text and ontology.
"""

facts_criteria = """
EVALUATION CRITERIA:
1. Appropriateness: Are the facts appropriate for the document?
2. Completeness: Are all possible facts extracted from the text given the ontology?
3. Concreteness: Only concrete should be extracted.
4. Structure: Are all concrete entities linked to abstract classes via relations?

OUTPUT:
Provide itemized, actionable critique specifying how to improve the ontology.
"""

facts_template = """
### Facts
```ttl
{facts_ttl}
```
"""
