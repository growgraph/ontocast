"""Common prompt templates and components shared across the application.

This module contains reusable prompt templates and components to avoid
duplication across different prompt modules.
"""

system_preamble_semantic = """
# SYSTEM INSTRUCTION

You are an expert in semantic technologies, SPARQL and triple extraction.
"""

system_preamble_ontology = """
# SYSTEM INSTRUCTION

You are an expert in semantic technologies and ontology engineering.
"""

ontology_template = """\n\n
# ONTOLOGY

```ttl
{ontology_ttl}
```
"""

text_template = """\n\n
# TEXT

```
{text}
```
"""

facts_template = """\n\n
# SEMANTIC GRAPH OF FACTS

```ttl
{facts_ttl}
```
"""


improvement_instruction_template = """\n\n
# IMPROVEMENT INSTRUCTION

Previous triples representing facts raised the following suggestions for improvement:

{suggestions_instruction}

Address all the suggestions and generate ADD, REMOVE
"""

output_instruction_empty = """\n\n
# OUTPUT INSTRUCTION

"""

output_instruction_ttl = """\n\n
# OUTPUT INSTRUCTION

1. ontology must be provided in turtle format as a single string
2. define all prefixes for all namespaces used in the ontology, etc rdf, rdfs, owl, schema, etc.
"""

output_instruction_sparql = """\n\n
# OUTPUT INSTRUCTION

Generate SPARQL operations that modify the existing ontology, not replace it entirely
"""

output_instruction_crit_facts = """\n\n
# OUTPUT INSTRUCTION

Provide itemized, actionable suggestions specifying how to improve the graph of extracted facts (markdown list). Each suggestion must be highly specific, following these rules:

1.  Completeness/Appropriateness Suggestions: For every suggested addition or removal of a fact, cite the exact sentence or phrase from the TEXT that justifies the change.
2.  Structural/Reuse Suggestions: For every structural or ontology reuse error, provide the incorrect triple(s) from the SEMANTIC GRAPH and the corresponding corrected triple(s), using the appropriate ontology prefix.
3.  Actionable Format: Start each suggestion with a clear verb (e.g., ADD, REMOVE, MODIFY).

Example of a concrete suggestion:
* ADD the triple `<:report1 a ex:Report .>` to address the missing type for the main entity. This fact is derived from the text: "The report details the project."
* MODIFY the incorrect property usage. INCORRECT: `ex:Person1 :has_age "45" .` CORRECT: `ex:Person1 ex:age "45"^^xsd:integer .` (The canonical property is `ex:age`).
"""

user_template = """\n\n
# USER INSTRUCTION

{user_instruction}
"""

suggestion_general_template = """\n\n
## GENERAL

{general_suggestion}
"""

suggestion_concrete_template = """\n\n
## CONCRETE

{suggestion_str}
"""
