"""Common prompt templates and components shared across the application.

This module contains reusable prompt templates and components to avoid
duplication across different prompt modules.
"""

# Common system preambles
system_preamble_semantic = """
# INSTRUCTION
You are an expert in semantic technologies, SPARQL and triple extraction.
"""

system_preamble_ontology = """
# INSTRUCTION
You are an expert in semantic technologies and ontology engineering.
"""

# Common instruction templates
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

# Common critique instruction template
critique_instruction_template = """
### CRITIQUE INSTRUCTION
Previous triples representing facts raised the following suggestions for improvement:

{suggestions_instruction}

Address all the suggestions and generate fact triples again.
"""

# Common output instructions
output_instruction_ttl = """
### OUTPUT INSTRUCTION

1. ontology must be provided in turtle format as a single string
2. define all prefixes for all namespaces used in the ontology, etc rdf, rdfs, owl, schema, etc.
"""

output_instruction_sparql = """
### OUTPUT instructions

1. generate SPARQL operations that modify the existing ontology, not replace it entirely
"""
