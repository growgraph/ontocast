"""Enhanced facts rendering prompts with context passing and SPARQL support.

This module provides enhanced prompt templates for facts rendering that support
context passing, memory, and SPARQL operations.
"""

# Enhanced ontology instruction with context
ontology_instruction_enhanced = """
```ttl
{ontology_str}
```

PREVIOUS CONTEXT:
{previous_context}

MEMORY INSTRUCTIONS:
- Consider the previous context when generating new facts
- Build upon previous work rather than starting from scratch
- Maintain consistency with previous versions
- Focus on incremental improvements rather than complete rewrites
"""

# Enhanced template prompt with context support
template_prompt_enhanced = """
Generate semantic triples representing facts (not abstract entities) based on provided domain ontology.

# Instructions

1. The facts (entities that are more concrete than the ones defined in ontologies) should be defined in custom namespace <{current_doc_namespace}> using the prefix `cd:` ( e.g. `@prefix cd: {current_doc_namespace} .` )
2. Use the provided domain ontology <{ontology_namespace}> (below) and standard ontologies (RDFS, OWL, schema.org, etc.) to identify/infer entities, classes, types, and relationships
3. Thoroughly Extract and Link: Extract all possible text mentions that correspond to entities, classes, types, or relationships defined in the domain ontology <{ontology_namespace}>. When referring to the domain ontology, use the prefix `{ontology_prefix}:`
4. Enforce typing: all `cd:` entities (facts) must be linked (e.g. using rdf:type) to entities from either the DOMAIN ONTOLOGY <{ontology_namespace}> or basic ontologies (RDFS, OWL, etc), e.g. rdfs:Class, rdf:Property, schema:Person, schema:Organization, etc.
5. Define all prefixes for all namespaces used rdf, rdfs, owl, schema, etc
6. Prefer Ontology IRIs: If a term (class/property/individual) exists in the domain or any standard ontology, use its IRI, **do not** create a `cd:` IRI with the same local name.
7. Maximize atomicity: decompose complex facts and complex literals into simple subject-predicate-object statements
8. Literals Handling:
    - Use appropriate XSD datatypes: xsd:integer, xsd:decimal, xsd:float, xsd:date, xsd:dateTime
    - Dates: Use ISO 8601 format (e.g., "2024-01-15"^^xsd:date)
    - Numbers: Always use typed literals (e.g., "42"^^xsd:integer, "99.95"^^xsd:decimal)
    - Currencies: Include currency codes (e.g., "1000"^^xsd:decimal with schema:priceCurrency "USD")
9. To extract data from tables, use CSV on the Web (CSVW) to describe tables
10. No comments in Turtle: Output must contain only @prefix declarations and triples. Do not include comments (lines starting with #)

CONTEXT-AWARE INSTRUCTIONS:
- Use the previous context to inform your decisions
- Build upon previous work rather than starting fresh
- Maintain consistency with previous versions
- Focus on incremental improvements
- Consider what has changed since the last version

# Domain Ontology

{ontology_instruction}

# Text for processing:

```
{text}
```

{failure_instruction}

{format_instructions}
"""

# Enhanced failure instruction with context
failure_instruction_enhanced = """
# FAILURE INSTRUCTION
The previous attempt to generate triples failed.

It failed at the stage: {failure_stage}

{failure_reason}

PREVIOUS CONTEXT:
{previous_context}

CONTEXT-AWARE FAILURE RECOVERY:
- Consider the previous context when addressing the failure
- Build upon previous work rather than starting fresh
- Maintain consistency with previous versions
- Focus on fixing the specific issues while preserving good work

Please fix the errors and do your best to generate fact triples again.
"""

# SPARQL-focused prompts for incremental updates
sparql_facts_instruction = """
Generate SPARQL operations to update the facts based on the document and previous context.

PREVIOUS FACTS:
{previous_facts}

PREVIOUS CONTEXT:
{previous_context}

ONTOLOGY CONTEXT:
{ontology_context}

INSTRUCTIONS:
1. Generate SPARQL INSERT/UPDATE/DELETE operations to update the facts
2. Focus on what needs to be added, modified, or removed
3. Use the ontology context to properly type entities
4. Use appropriate namespaces and prefixes
5. Return ONLY SPARQL operations
6. Each operation should be a complete INSERT/UPDATE/DELETE block
7. Consider the previous context when determining what changes are needed

EXAMPLE FORMAT:
INSERT DATA {{
    <http://example.org/doc/{chunk_id}/NewEntity> a schema:Person .
    <http://example.org/doc/{chunk_id}/NewEntity> schema:name "John Doe" .
}}
DELETE DATA {{
    <http://example.org/doc/{chunk_id}/OldEntity> a schema:Person .
}}
"""

sparql_template_prompt = """
{sparql_facts_instruction}

Here is the document:

```
{text}
```

{failure_instruction}

{format_instructions}
"""
