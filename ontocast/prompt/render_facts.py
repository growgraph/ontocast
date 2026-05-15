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

1. Facts MUST use the fixed namespace `{facts_namespace}` with the prefix `cd:` (declare exactly: `@prefix cd: <{facts_namespace}> .`). Local names for facts should not be capitalized.

1a. TWO-NAMESPACE CONTRACT (most important rule):
    - {domain_ontologies_clause}: schema elements only — classes (as `rdf:type` objects), predicates, and named individuals that exist verbatim in the ontology
    - `cd:`: ALL new instances extracted from the text, even if typed by an ontology class

    CORRECT: `cd:trial_1 a onto:Trial ; onto:hasJudgment cd:judgment_1 .`
    WRONG:   `onto:Trial_1 a onto:Trial .`  — new instance under ontology prefix, FORBIDDEN

2. Use the provided {domain_ontologies_clause} (below) and standard ontologies (RDFS, OWL, schema.org, etc.) to identify/infer entities, classes, types, and relationships
3. Thoroughly Extract and Link: extract all possible text mentions that correspond to entities, classes, types, or relationships defined in {domain_ontologies_clause}
4. Enforce typing: all `cd:` entities (facts) must be linked (e.g. using rdf:type) to entities from either {domain_ontologies_clause} or basic ontologies (RDFS, OWL, etc), e.g. rdfs:Class, rdf:Property, schema:Person, schema:Organization, etc.
5. Define all prefixes for all namespaces used rdf, rdfs, owl, schema, etc
6. CRITICAL - Entity Matching Protocol:
   - BEFORE creating any `cd:` entity, search the domain ontology for existing entities that match the concept semantically
   - A "matching entity" means a resource that EXISTS VERBATIM in the provided ontology as a named individual
     (declared with owl:NamedIndividual or explicitly typed) — NOT simply a class whose name resembles the entity.
     A class existing in the ontology does NOT mean an instance of that class also exists: create a new `cd:` instance typed by that class.
   - Match by meaning, not just exact label; check all `rdfs:label` language variants
   - If a matching named individual exists in the domain ontology, use its IRI directly — do NOT duplicate it in `cd:`
   - Only create `cd:` entities for NEW facts not already defined in the ontology as named individuals
   - NEVER mint new IRIs in the domain ontology namespace(s) unless that exact IRI already exists in the provided ontology as a named individual
   - Preserve canonical ontology IRIs exactly as given (character-for-character): no translation, no transliteration, no casing changes
   - Cross-lingual mentions MUST be linked to the existing canonical ontology IRI when semantically equivalent
   - If no ontology entity can be verified, create a `cd:` entity instead of inventing a new ontology-prefixed IRI
6a. Opaque Identifier Ontologies (Wikidata-style Q/P codes, hashes, UUIDs):
   - When ontology IRIs contain opaque local names (Q-numbers, P-numbers, hash strings, numeric IDs),
     entity identity is determined EXCLUSIVELY by `rdfs:label`, `rdfs:comment`, skos:altLabel — not the IRI fragment
   - Use the TERM INDEX (if provided below the ontology) to map text mentions to their canonical IRI
   - NEVER construct an IRI by appending a label string to the ontology namespace
     (e.g. `onto:culture` is ALWAYS wrong — the correct IRI is whatever appears in the ontology with `rdfs:label "culture"`)
   - NEVER invent or guess a Q/P code — only use codes that appear explicitly in the provided ontology
   - For property domain/range chains: resolve referenced opaque IRIs to their labels before deciding
     which subject/object types are valid for a given property
7. Maximize atomicity: decompose complex facts and complex literals into simple subject-predicate-object statements (e.g. decompose person's  first name and last name).
8. Literals and Quantity Values:
   - Use appropriate XSD datatypes: xsd:integer, xsd:decimal, xsd:float,
     xsd:date, xsd:dateTime. Dates use ISO 8601.
    - Dates: Use ISO 8601 format (e.g., "2024-01-15"^^xsd:date)
    - Numbers: Always use typed literals (e.g., "42"^^xsd:integer, "99.95"^^xsd:decimal)
    - Currencies: Include currency codes (e.g., "1000"^^xsd:decimal with schema:priceCurrency "USD")
   - NEVER encode a numeric measurement as xsd:string, even if the source
     text is approximate or bounded.
   - When a measurement appears with an epistemic qualifier — approximation
     (∼, ~, ≈, ca., about), bound (<, >, ≤, ≥, up to, at least, more than,
     exceeding), range (X–Y, X to Y), or uncertainty (X ± Y) — decompose
     it into a structured node:
       * Search the provided ontology for a class representing approximate or
         bounded quantity values (e.g. a QuantityValue subclass, a
         MeasuredValue class, or equivalent).
       * If found, instantiate it and use its typed decimal properties for
         the numeric components (nominal value, lower/upper bound, uncertainty)
         and its qualifier properties for the epistemic marker.
       * If no such class is found in the domain ontology, use qudt:QuantityValue
         as the type and attach the numeric parts with qudt:numericValue /
         qudt:unit, adding a plain qualifier annotation (e.g. rdfs:comment
         or a well-known approximation property).
   - Prose restatements of a measurement in dcterms:description are redundant
     once typed numeric properties exist — omit them.
9. To extract data from tables, use CSV on the Web (CSVW) to describe tables
10. No comments in Turtle: Output must contain only @prefix declarations and triples. Do not include comments (lines starting with #)
11. Decide whether external evidence is needed for a retry and set `external_evidence_request`:
    - Set `initiate_search=true` only when ambiguity/term disambiguation/standards lookup materially blocks quality.
    - Otherwise keep `initiate_search=false`.
    - Provide concise `rationale` and optional focused `query_hints` when search is requested.
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
   - No ontology-prefixed entity was invented or renamed
   - All OPERATIONAL GUIDELINES are satisfied
   - The overall graph is more complete and accurate than before

Your goal: Produce the most accurate representation of the source text, not to satisfy Critic.
{suggestions_instruction}
"""
