"""Enhanced ontology rendering prompts with context passing and SPARQL support.

This module provides enhanced prompt templates for ontology rendering that support
context passing, memory, and SPARQL operations.
"""

# Enhanced ontology instruction for fresh ontologies with context
ontology_instruction_fresh_enhanced = """
Propose/develop a new domain ontology based on the provided document. When deciding on the name and scope, remember that the document you are given is just an example, so the ontology name, ontology identifier and scope should be at least one level of abstraction above the scope of the document.

CONTEXT FROM PREVIOUS WORK:
{previous_context}

MEMORY INSTRUCTIONS:
- Consider the previous context when generating new ontology triples
- Build upon previous work rather than starting from scratch
- Maintain consistency with previous versions
- Focus on incremental improvements rather than complete rewrites
"""

# Enhanced specific ontology instruction for fresh ontologies
specific_ontology_instruction_fresh_enhanced = """
1. all new abstract entities/classes/types or properties added to the new ontology must be linked to entities from basic ontologies (RDFS, OWL, schema etc), e.g. rdfs:Class, rdfs:subClassOf, rdf:Property, rdfs:domain, owl:Restriction, schema:Person, schema:Organization, etc
2. propose a domain specific and succinct specifier if for the new ontology, which should be an abbreviation, consistent with the Ontology property `ontology_id`, for example it could be `abc` for a hypothetical A... B... of C... Ontology.
3. derive from a proposed `ontology_id` an IRI (URI) using domain {current_domain}, for example `{current_domain}/abc`
4. explicitly use namespace `co:` for entities/properties placed in the proposed ontology.
5. CONSIDER PREVIOUS CONTEXT: Build upon any previous ontology work mentioned in the context
6. MAINTAIN CONSISTENCY: Ensure new entities are consistent with previous versions
"""

# Enhanced ontology instruction for updates with context
ontology_instruction_update_enhanced = """
Update/complement the domain ontology {ontology_iri} provided below with abstract entities and relations that can be inferred from the document.

{ontology_desc}

Feel free to modify the description of the ontology to make it more accurate and complete, but to change neither the ontology IRI nor name.

```ttl
{ontology_str}
```

CONTEXT FROM PREVIOUS WORK:
{previous_context}

MEMORY INSTRUCTIONS:
- Consider the previous context when updating the ontology
- Build upon previous work rather than starting from scratch
- Maintain consistency with previous versions
- Focus on incremental improvements rather than complete rewrites
"""

# Enhanced specific ontology instruction for updates
specific_ontology_instruction_update_enhanced = """
- all new abstract entities/classes/types or properties added to <{ontology_namespace}> ontology must be linked to entities from either domain ontology <{ontology_namespace}> or basic ontologies (RDFS, OWL, schema etc), e.g. rdfs:Class, rdfs:subClassOf, rdf:Property, rdfs:domain, owl:Restriction, schema:Person, schema:Organization, etc
- add new constraints and axioms if needed
- CONSIDER PREVIOUS CONTEXT: Build upon any previous ontology work mentioned in the context
- MAINTAIN CONSISTENCY: Ensure new entities are consistent with previous versions
- INCREMENTAL UPDATES: Focus on what needs to be added or modified, not complete rewrites
"""

# Enhanced instructions with context support
instructions_enhanced = """
Follow the instructions:

{specific_ontology_instruction}

1. ontology must be provided in turtle (ttl) format as a single string.
2. (IMPORTANT) define all prefixes for all namespaces used in the ontology, etc rdf, rdfs, owl, schema, etc.
3. in case you are familiar with domain specific ontologies, feel free to use them. For example (Financial Industry Business Ontology (FIBO) in finance, or XBRL-to-RDF transformations.
4. do not add facts, or concrete entities from the document.
5. make sure newly introduced entities are well linked / described by their properties.
6. assign where possible correct units to numeric literals.
7. make sure that the semantic representation is faithful to the document, feel to use your knowledge and common sense to make the ontology more complete and accurate.
8. feel free to update/assign the version of the ontology using semantic versioning convention.

CONTEXT-AWARE INSTRUCTIONS:
- Use the previous context to inform your decisions
- Build upon previous work rather than starting fresh
- Maintain consistency with previous versions
- Focus on incremental improvements
- Consider what has changed since the last version
"""

# Enhanced failure instruction with context
failure_instruction_enhanced = """
IMPORTANT: The previous attempt to generate ontology triples failed/was unsatisfactory.

It failed at the stage: {failure_stage}

{failure_reason}

PREVIOUS CONTEXT:
{previous_context}

CONTEXT-AWARE FAILURE RECOVERY:
- Consider the previous context when addressing the failure
- Build upon previous work rather than starting fresh
- Maintain consistency with previous versions
- Focus on fixing the specific issues while preserving good work

Please address ALL the issues outlined in the critique. We will be penalized :( for each unaddressed issue.
"""

# Enhanced template prompt with context support
template_prompt_enhanced = """
{ontology_instruction}

{instructions}

Here is the document:

```
{text}
```

{failure_instruction}

{format_instructions}
"""

# SPARQL-focused prompts for incremental updates
sparql_ontology_instruction = """
Generate SPARQL operations to update the ontology based on the document and previous context.

PREVIOUS ONTOLOGY:
{previous_ontology}

PREVIOUS CONTEXT:
{previous_context}

INSTRUCTIONS:
1. Generate SPARQL INSERT/UPDATE/DELETE operations to update the ontology
2. Focus on what needs to be added, modified, or removed
3. Use appropriate namespaces and prefixes
4. Return ONLY SPARQL operations
5. Each operation should be a complete INSERT/UPDATE/DELETE block
6. Consider the previous context when determining what changes are needed

EXAMPLE FORMAT:
INSERT DATA {{
    <http://example.org/ns#NewClass> a rdfs:Class .
    <http://example.org/ns#NewProperty> a rdf:Property .
}}
DELETE DATA {{
    <http://example.org/ns#OldProperty> a rdf:Property .
}}
"""

sparql_template_prompt = """
{sparql_ontology_instruction}

Here is the document:

```
{text}
```

{failure_instruction}

{format_instructions}
"""
