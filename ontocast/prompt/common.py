"""Common prompt templates and components shared across the application.

This module contains reusable prompt templates and components to avoid
duplication across different prompt modules.
"""

from ontocast.onto.enum import LLMGraphFormat

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

`sparql_operations` is for complex custom queries only. All standard add/remove operations
MUST use `triple_operations` with type `insert` or `delete`.
"""

output_instruction_jsonld = """\n\n
# OUTPUT INSTRUCTION

Provide each RDF graph field as a compact JSON-LD **object** (not a string) with:

1. "@context": a map of every prefix alias used to its full namespace IRI. Always declare
   rdf, rdfs, owl, xsd, schema, the facts prefix (e.g. cd), and any domain ontology prefixes.
2. "@graph": an array of subject nodes. Each node MUST have "@id" (compact IRI) and SHOULD
   include "@type" plus all predicate-value pairs for that subject grouped in one object.
3. Use compact IRIs (`prefix:local`) throughout - never expand to full URIs in the body.
4. Typed literals MUST use the value/type form: {"@value": "2024-01-15", "@type": "xsd:date"}.
   Language-tagged literals use {"@value": "...", "@language": "en"}.
5. Multi-valued predicates use a JSON array of objects/values.
6. Object references use {"@id": "prefix:local"} (or a plain compact IRI string when unambiguous).
7. No comments, no trailing prose - output strictly valid JSON.
"""

output_instruction_sparql_jsonld = """\n\n
# OUTPUT INSTRUCTION

Generate SPARQL operations that modify the existing graph, not replace it entirely.
Follow the Pydantic schema definitions exactly - they fully specify the output structure.

For each `TripleOp.graph` field, provide a compact JSON-LD **object** (not a string) with:

1. "@context": a map of every prefix alias used to its full namespace IRI.
   Always declare rdf, rdfs, owl, xsd, schema, the facts prefix (e.g. cd), and any
   domain ontology prefixes referenced by the operation.
2. "@graph": an array of subject nodes. Each node MUST have "@id" (compact IRI) and SHOULD
   include "@type" plus all predicate-value pairs for that subject grouped in one object.
3. Use compact IRIs (`prefix:local`) throughout - never expand to full URIs in the body.
4. Typed literals MUST use the value/type form: {"@value": "...", "@type": "xsd:date"}.
   Language-tagged literals use {"@value": "...", "@language": "en"}.
5. No comments, no trailing prose - output strictly valid JSON.

`sparql_operations` is for complex custom queries only. All standard add/remove operations
MUST use `triple_operations` with type `insert` or `delete`.
"""

output_instruction_critique_turtle = """\n\n
# GRAPH FORMAT INSTRUCTION (LLM_GRAPH_FORMAT=turtle)

The deployment emits RDF graph fixes in Turtle syntax.
For each `incorrect_value` and `correct_value` in actionable fixes, provide a **string**
containing valid Turtle: `@prefix` declarations when needed, then one or more triples.
Example: "@prefix ex: <http://example.org/> . ex:alice ex:worksFor ex:acme ."
"""

output_instruction_critique_jsonld = """\n\n
# GRAPH FORMAT INSTRUCTION (LLM_GRAPH_FORMAT=jsonld)

The deployment emits RDF graph fixes as compact JSON-LD.
For each `incorrect_value` and `correct_value` in actionable fixes, provide a **string**
containing valid JSON for one subject node (inline `@context` or compact IRIs only):
Example: "{\\"@context\\": {\\"ex\\": \\"http://example.org/\\"}, \\"@id\\": \\"ex:alice\\", \\"ex:worksFor\\": {\\"@id\\": \\"ex:acme\\"}}"
Use `{"@value": "...", "@type": "xsd:date"}` for typed literals and `{"@value": "...", "@language": "en"}`
for language-tagged literals.
"""


def critique_graph_format_instruction(llm_graph_format: LLMGraphFormat) -> str:
    """Return critic-specific graph syntax instructions for TripleFix values."""
    if llm_graph_format == LLMGraphFormat.JSONLD:
        return output_instruction_critique_jsonld
    return output_instruction_critique_turtle


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
