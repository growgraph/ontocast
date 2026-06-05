"""Common prompt templates and components shared across the application.

This module contains reusable prompt templates and components to avoid
duplication across different prompt modules.
"""

system_preamble_semantic = """
# SYSTEM INSTRUCTION

You are an expert in semantic technologies, RDF, and triple extraction.
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
