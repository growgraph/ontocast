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

facts_template = """
# FACTS

```ttl
{facts_ttl}
```
"""


critique_instruction_template = """\n\n
# CRITIQUE INSTRUCTION

Previous triples representing facts raised the following suggestions for improvement:

{suggestions_instruction}

Address all the suggestions and generate fact triples again.
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

Provide itemized, actionable critique specifying how to improve the graph of extracted facts.
"""

user_template = """\n\n
# USER INSTRUCTION

{user_instruction}
"""
