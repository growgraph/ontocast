from .common import system_preamble_semantic

template_prompt = """
{preamble}

{facts_instruction}

{user_instruction}

{ontology_chapter}

{text_chapter}

{fact_chapter}

{improvement_instruction}

{output_instruction}

{format_instructions}
"""

preamble = f"""
{system_preamble_semantic}
Generate semantic triples representing facts (not abstract entities) based on provided domain ontology.
"""

facts_instruction_template = """\n\n
# OPERATIONAL GUIDELINES

1. The facts (entities that are more concrete than the ones defined in ontologies) should be defined in custom namespace <{current_doc_namespace}> using the prefix `cd:` ( e.g. `@prefix cd: {current_doc_namespace} .` )
2. Use the provided domain ontology <{ontology_namespace}> (below) and standard ontologies (RDFS, OWL, schema.org, etc.) to identify/infer entities, classes, types, and relationships
3. Thoroughly Extract and Link: extract all possible text mentions that correspond to entities, classes, types, or relationships defined in the domain ontology <{ontology_namespace}>. When referring to the domain ontology, use the prefix `{ontology_prefix}:`
4. Enforce typing: all `cd:` entities (facts) must be linked (e.g. using rdf:type) to entities from either the DOMAIN ONTOLOGY <{ontology_namespace}> or basic ontologies (RDFS, OWL, etc), e.g. rdfs:Class, rdf:Property, schema:Person, schema:Organization, etc.
5. Define all prefixes for all namespaces used rdf, rdfs, owl, schema, etc
6. CRITICAL - Entity Matching Protocol:
   - BEFORE creating any `cd:` entity, you MUST search the domain ontology for existing entities that match the concept semantically
   - Match by meaning, not just exact label matching
   - Check all language variants of `rdfs:label` and alternative names
   - If a matching entity exists in the domain ontology, use its IRI directly - DO NOT create a duplicate in the `cd:` namespace
   - Only create `cd:` entities for NEW facts not already defined in the ontology
7. Maximize atomicity: decompose complex facts and complex literals into simple subject-predicate-object statements
8. Literals Handling:
    - Use appropriate XSD datatypes: xsd:integer, xsd:decimal, xsd:float, xsd:date, xsd:dateTime
    - Dates: Use ISO 8601 format (e.g., "2024-01-15"^^xsd:date)
    - Numbers: Always use typed literals (e.g., "42"^^xsd:integer, "99.95"^^xsd:decimal)
    - Currencies: Include currency codes (e.g., "1000"^^xsd:decimal with schema:priceCurrency "USD")
9. To extract data from tables, use CSV on the Web (CSVW) to describe tables
10. No comments in Turtle: Output must contain only @prefix declarations and triples. Do not include comments (lines starting with #)
"""

improvement_instruction_template = """\n\n
# IMPROVEMENT INSTRUCTION

The current iteration of the graph of factual triples has been reviewed by Critic, who provided suggestions for improvement.

CRITICAL: You are the final decision-maker. Critic's suggestions are advisory, not mandatory. Think independently.

Your task is to critically evaluate and improve the triples:

1. Independently verify each suggestion - Before implementing ANY suggestion, verify it against:
   - The original source text (does it accurately reflect what's written?)
   - The OPERATIONAL GUIDELINES (does it follow the rules?)
   - The domain ontology (does it use entities correctly?)
   - Logical consistency (does it make semantic sense?)

2. Implement only valid improvements - Apply suggestions that are demonstrably correct and enhance accuracy or completeness. If uncertain, prioritize faithfulness to the source text.

3. Actively reject flawed suggestions - If a suggestion is:
   - Factually incorrect (contradicts the source text)
   - Violates OPERATIONAL GUIDELINES
   - Would introduce errors or degrade quality
   - Based on misunderstanding of the ontology
   
   Then REJECT it and briefly explain why in your response.

4. Think beyond the critique - Critic may have:
   - Missed issues entirely
   - Identified patterns but not all instances
   - Focused on some aspects while overlooking others
   
   Proactively identify and fix additional problems not mentioned in the critique.

5. Verify every change - Before finalizing, double-check that:
   - Each triple accurately represents information from the source text
   - Existing ontology entities are used instead of creating new cd: entities
   - All OPERATIONAL GUIDELINES are satisfied
   - The overall graph is more complete and accurate than before

Your goal: Produce the most accurate representation of the source text, not to satisfy Critic.
{suggestions_instruction}
"""
