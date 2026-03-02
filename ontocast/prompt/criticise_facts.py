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

5. Namespace Consistency:
   - Facts MUST use `cd:` with fixed namespace `<{DEFAULT_IRI}>`
   - Flag any fact entity that uses a different "facts-like" namespace as error
   - `cd:` is reserved for concrete instances/facts only (not ontology classes/properties)

6. Ontology Validity: Verify that every non-cd: entity exists in either the provided domain ontology or standard ontologies (RDFS, OWL, schema.org, etc.).
   - Every class, property, and individual using ontology prefixes (fca:, onto:, schema:, etc.) must be defined in its respective ontology
   - Invented entities using ontology prefixes are errors
   - Fix: REMOVE invented entities or REPLACE with semantically similar existing entities from available ontologies

# VERIFICATION CHECKLIST

Before finalizing your critique:

1. For every triple using a non-cd: prefix, confirm the entity exists in the corresponding ontology

2. For every fact entity, confirm it uses `cd:` with `<{DEFAULT_IRI}>`

3. When you find invented entities:
   - Flag as error
   - Search available ontologies for semantically similar entities
   - Suggest REPLACE if found, or REMOVE if no suitable replacement exists
"""
