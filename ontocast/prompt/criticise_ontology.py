template_prompt = """
{preamble}

{intro_instruction}

{ontology_criteria}

{user_instruction}

{ontology_chapter}

{document_chapter}
"""

system_preamble = """
# INSTRUCTION

You are an expert in semantic technologies and ontology engineering.
"""

intro_instruction = """
You are given a text and an ontology.
You task is to provide a constructive critique of the ontology with respect to provided text.
"""


ontology_criteria = """
EVALUATION CRITERIA:
1. Appropriateness: Is the ontology appropriate for the document?
2. Completeness and Consistency: Is the ontology complete and consistent?
3. Abstraction: Only abstract classes and properties should be present in the ontology.
4. Domain Coverage: Ontology should include implicit domain-specific abstractions and relationships not explicitly mentioned
5. Structure: Check hierarchies, property definitions, redundancy, and appropriate granularity

OUTPUT:
Provide itemized, actionable critique specifying how to improve the ontology.
"""


ontology_template = """
### Ontology
```ttl
{ontology_ttl}
```
"""


document_template = """
### The Document of Interest
```ttl
{document}
```
"""
