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
The following facts were extracted

```ttl
{facts_ttl}
```
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

Generate SPARQL operations that modify the existing ontology, not replace it entirely.
Follow the Pydantic schema definitions exactly - they fully specify the output structure.
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
