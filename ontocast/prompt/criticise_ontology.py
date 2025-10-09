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

You are an expert in SPARQL and ontology engineering. You task is to provide critique of an ontology with respect to provided text.
"""

ontology_criteria = """
EVALUATION CRITERIA:
1. Appropriateness: Are changes appropriate for the document and ontology?
2. Completeness: Do changes capture all necessary updates?
3. Consistency: Are changes consistent with existing ontology structure?
4. Abstraction: Only abstract classes and properties belong in the ontology, not concrete instances
5. Domain Coverage: Include implicit domain-specific abstractions and relationships not explicitly mentioned
6. Structure: Check hierarchies, property definitions, redundancy, and appropriate granularity

CONTEXT-AWARE EVALUATION:
- Review previous critiques before evaluating
- Build upon previous feedback incrementally
- Acknowledge improvements already made
- Focus on remaining gaps or new issues

OUTPUT:
Provide itemized, actionable critique specifying what should be done to improve the ontology.
"""

intro_first_no_seed_instruction = """
You are provided a text and an ontology (below).
"""

intro_first_with_seed_instruction = """
You are provided a text (below), an ontology and its update (suggested edits to the ontology).
Provide the critique of the ontology taking into account the update.
"""

intro_subsequent_instruction = """
You were provided the text previously in the conversation as well as the ontology with its updates.
Below you are provided a current update addressing your previous critiques.
Refine your critique of the ontology taking into account all the preceding updates.
"""

ontology_template = """
### Ontology
```ttl
{ontology_ttl}
```
{ontology_update}
"""

ontology_update_template = """
### Ontology Update
{ontology_update}
"""

document_template = """
### The Document of Interest
```ttl
{document}
```

"""
