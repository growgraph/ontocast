"""Enhanced facts criticism prompts.

This module provides enhanced prompt templates for facts criticism that support
with improved critique quality.
"""

from ontocast.onto.constants import DEFAULT_IRI

from .common import system_preamble_semantic

template_prompt = """
{preamble}

{evaluation_instruction}

{user_instruction}

{ontology_chapter}

{facts_chapter}

{text_chapter}

{graph_format_instruction}

{format_instructions}
"""

preamble = f"""
{system_preamble_semantic}
You are given an ontology, a text and a semantic graph of facts, generated from the text (guided by ontology).
Following evaluation guidelines provide concrete suggestions for improvement of the extracted facts graph with respect to provided text and ontology.
"""


evaluation_instruction = f"""\n\n
# EVALUATION GUIDELINES

1. Appropriateness: Are the facts appropriate for the document?

2. Completeness: Are all possible facts extracted from the text given the ontology?

3. Concreteness: Only concrete facts should be extracted.

4. Structure: Are all concrete entities linked to abstract classes via relations?

5. Namespace Consistency (TWO-NAMESPACE CONTRACT):
   - Facts MUST use `cd:` with fixed namespace `<{DEFAULT_IRI}>`
   - Flag any fact entity that uses a different "facts-like" namespace as error
   - `cd:` is reserved for concrete instances/facts only (not ontology classes/properties)
   - CRITICAL — Invented instances under ontology prefix: the most common error is placing newly-created
     instances under the domain ontology prefix just because their class exists in the ontology.
     Example of the error: `onto:Trial_1 a onto:Trial` — `Trial_1` is a new instance, so it MUST be `cd:trial_1 a onto:Trial`.
     Flag every case where a subject IRI uses a domain ontology prefix but represents a concrete instance
     that does not literally exist in the provided ontology as a named individual.

6. Ontology Validity: Verify that every non-cd: entity exists in either the provided domain ontology or standard ontologies (RDFS, OWL, schema.org, etc.).
   - Every class, property, and individual using ontology prefixes (fca:, onto:, schema:, etc.) must be defined in its respective ontology
   - Invented entities using ontology prefixes are errors
   - Fix: REMOVE invented entities or REPLACE with semantically similar existing entities from available ontologies
   - Treat morphology/casing variants of ontology terms as likely mistakes (e.g. `AppealCourt_Rouen` vs `AppealCourtRouen`) unless the variant exists explicitly in ontology
   - For multilingual variants, prefer the exact canonical ontology IRI that exists in ontology; do not allow translated/reformatted ontology-prefixed IRIs as new entities
6a. Opaque Identifier Ontologies (Wikidata-style Q/P codes, hashes, UUIDs):
   - When the domain ontology uses opaque local IRI names (Q-numbers, P-numbers, hash IDs, numeric codes),
     the ONLY valid way to identify an entity is via its `rdfs:label` — NEVER by appending a human-readable label to the namespace
   - Flag as error any entity IRI of the form `<namespace>:<human-readable-label>` when the ontology uses opaque identifiers
     (e.g. `ont_10_culture_concepts:culture` is an invented IRI if the ontology only defines `ont_10_culture_concepts:Q11042` with `rdfs:label "culture"`)
   - Do NOT flag correct Q/P code IRIs as invented — verify against the provided ontology, not against intuition about what the IRI "should" look like
   - When checking property domain/range triples: resolve the referenced opaque IRI to its label before deciding whether the subject/object type is appropriate

# VERIFICATION CHECKLIST

Before finalizing your critique:

1. For every triple using a non-cd: prefix as the **subject**, confirm the entity exists verbatim as a
   named individual in the corresponding ontology (not merely that its class exists there).
   - If it does not exist verbatim as a named individual, flag as "invented instance under ontology prefix"
     and propose replacement: move it to `cd:` and keep the ontology prefix only for the `rdf:type` and predicates.

2. For every fact entity, confirm it uses `cd:` with `<{DEFAULT_IRI}>`

3. When you find invented entities:
   - Flag as error
   - Search available ontologies for semantically similar entities
   - Suggest REPLACE if found, or REMOVE if no suitable replacement exists

7. Quarantined typed literals: When a "Quarantined triples" section is present,
   each entry had an XSD datatype whose lexical form is invalid for that type
   (e.g. ranges, approximations, or unit suffixes encoded as a single decimal).
   - Propose critical REPLACE fixes that use ontology-defined structured representations
     from the ontology chapter (not ad-hoc xsd:string unless no pattern exists).
   - `correct_value` / `incorrect_value` MUST follow the GRAPH FORMAT INSTRUCTION below.

# OUTPUT FORMAT CONSTRAINTS

- Respond with valid JSON only. Do NOT include `//` comments or any other non-JSON syntax inside the JSON block.
- Use the `explanation` field to convey any reasoning that you would otherwise put in a comment.
"""
