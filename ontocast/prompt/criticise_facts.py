"""Enhanced facts criticism prompts.

This module provides enhanced prompt templates for facts criticism that support
with improved critique quality.

The evaluation guidelines are built per unit (:func:`evaluation_instruction_for`):
the quantitative-completeness guideline only for a unit whose text states a
number, the cited-work clause only when the citation vocabulary names an
``attributed_to`` term.
"""

from collections.abc import Mapping

from ontocast.onto.constants import DEFAULT_IRI

from .common import system_preamble_semantic

# Opens exactly like the render template (preamble, conformance contract,
# ontology chapter) so the two prompts on a unit share their prefix through
# the end of the ontology chapter -- the part a provider's prefix cache can
# reuse. See prompt/render_facts.py; a test pins the two heads to each other.
template_prompt = """
{preamble}

{conformance_chapter}

{ontology_chapter}

# TASK

You are given an ontology, a text and a semantic graph of facts, generated from the text (guided by ontology).
Following evaluation guidelines provide concrete suggestions for improvement of the extracted facts graph with respect to provided text and ontology.

{evaluation_instruction}

{user_instruction}

{text_chapter}

{facts_chapter}

{graph_format_instruction}

{format_instructions}
"""

# Shared verbatim with the renderer; the task statement sits in the template
# after the ontology chapter for the same reason.
preamble = system_preamble_semantic


_QUANTITATIVE_COMPLETENESS = """   - Quantitative completeness is checked mechanically: the NUMERIC COVERAGE
     findings list numbers stated in the text but absent from the graph —
     measurements (a number with its unit) first, bare numbers after. Do NOT
     propose an ADD fix per listed number. A number only becomes a fact
     together with the quantity it measures, and a value whose quantity you
     can name is an omission the extractor must see in full context: list the
     measurements you judge missed — verbatim value, unit, and the phrase
     they occur in — in `systemic_critique_summary`{completion}.
"""

_COMPLETION_CLAUSE = "; a completion pass may extract them"

_CITED_WORK = """     The work a citation marker refers to, linked with `{attributed_to}` from
     the statement the text attributes to it, is an entity; keep it.
"""

_EVALUATION_INSTRUCTION = f"""\n\n
# EVALUATION GUIDELINES

1. Appropriateness: Are the facts appropriate for the document?

2. Completeness: Are all possible facts extracted from the text given the ontology?
{{quantitative_completeness}}   - NEVER mint an entity or placeholder node for a bare number, an ignored
     token, a citation marker, a page/figure/equation token, or a conversion
     artifact. Such tokens are typography, not facts: name them in
     `explanation` if at all, never as a subject in `correct_value`. A node
     whose only content is a label or comment is not a fact either.
{{cited_work}}
3. Concreteness: Only concrete facts should be extracted.

4. Structure: Are all concrete entities linked to abstract classes via relations?

5. Namespace Consistency (TWO-NAMESPACE CONTRACT):
   - Facts MUST use `cd:` with fixed namespace `<{DEFAULT_IRI}>`
   - Flag any fact entity that uses a different "facts-like" namespace as error
   - `cd:` is reserved for concrete instances/facts only (not ontology classes/properties)
   - CRITICAL — Invented instances under ontology prefix: the most common error is placing newly-created
     instances under the domain ontology prefix just because their class exists in the ontology.
     Example of the error: `onto:Project_1 a onto:Project` — `Project_1` is a new instance, so it MUST be `cd:project_1 a onto:Project`.
     Flag every case where a subject IRI uses a domain ontology prefix but represents a concrete instance
     that does not literally exist in the provided ontology as a named individual.

6. Ontology Validity: Verify that every non-cd: entity exists in either the provided domain ontology or standard ontologies (RDFS, OWL, schema.org, etc.).
   - Every class, property, and individual using ontology prefixes (onto:, schema:, etc.) must be defined in its respective ontology
   - Invented entities using ontology prefixes are errors
   - Fix: REMOVE invented entities or REPLACE with semantically similar existing entities from available ontologies
   - Treat morphology/casing variants of ontology terms as likely mistakes (e.g. `onto:Work_Unit` vs `onto:WorkUnit`) unless the variant exists explicitly in ontology
   - For multilingual variants, prefer the exact canonical ontology IRI that exists in ontology; do not allow translated/reformatted ontology-prefixed IRIs as new entities
6a. Opaque Identifier Ontologies (Wikidata-style Q/P codes, hashes, UUIDs):
   - When the domain ontology uses opaque local IRI names (Q-numbers, P-numbers, hash IDs, numeric codes),
     the ONLY valid way to identify an entity is via its `rdfs:label` — NEVER by appending a human-readable label to the namespace
   - Flag as error any entity IRI of the form `<namespace>:<human-readable-label>` when the ontology uses opaque identifiers
     (e.g. `ont_10_culture_concepts:culture` is an invented IRI if the ontology only defines `ont_10_culture_concepts:Q11042` with `rdfs:label "culture"`)
   - Do NOT flag correct Q/P code IRIs as invented — verify against the provided ontology, not against intuition about what the IRI "should" look like
   - When checking property domain/range triples: resolve the referenced opaque IRI to its label before deciding whether the subject/object type is appropriate

6b. Specificity (ancestor vs. descendant terms):
   - The ontology context may include both an ancestor term and narrower `rdfs:subClassOf` /
     `rdfs:subPropertyOf` descendants shown together for domain/range context.
   - Flag any fact that uses a broader ancestor class or property when a narrower descendant is
     present in the supplied ontology context and its stated domain/range/restrictions also fit
     the subject and object.
   - Suggest REPLACE with the narrower descendant term; keep the ancestor only when no descendant
     fits or the source text is genuinely stated at the abstract level.

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

7. Quarantined literals: When a "Quarantined triples" section is present,
   each entry is a literal that does not fit its property's declared range,
   or whose form is invalid for its datatype: text where the property takes a
   number or a date (e.g. a range, an approximation, or a number with a unit
   written as one value).
   - Propose critical REPLACE fixes that use ontology-defined structured representations
     from the ontology chapter (not ad-hoc xsd:string unless no pattern exists).
   - `correct_value` / `incorrect_value` MUST follow the GRAPH FORMAT INSTRUCTION below.

# OUTPUT FORMAT CONSTRAINTS

- Respond with valid JSON only. Do NOT include `//` comments or any other non-JSON syntax inside the JSON block.
- Use the `explanation` field to convey any reasoning that you would otherwise put in a comment.
"""


def evaluation_instruction_for(
    *,
    states_quantity: bool,
    citation_vocabulary: Mapping[str, str] | None = None,
    recovers_measurements: bool = False,
) -> str:
    """The EVALUATION GUIDELINES chapter for one unit.

    Args:
        states_quantity: Whether the unit text states a number; without one
            the quantitative-completeness guideline is left out.
        citation_vocabulary: ``CHUNK_CITATION_VOCABULARY``; its
            ``attributed_to`` term keeps the cited work out of the typography
            the critic removes.
        recovers_measurements: Whether a gap pass recovering missing
            measurements runs after the critic; only then does the guideline
            say one may extract them.
    """
    attributed_to = (citation_vocabulary or {}).get("attributed_to")
    completeness = _QUANTITATIVE_COMPLETENESS.format(
        completion=_COMPLETION_CLAUSE if recovers_measurements else ""
    )
    return _EVALUATION_INSTRUCTION.format(
        quantitative_completeness=completeness if states_quantity else "",
        cited_work=_CITED_WORK.format(attributed_to=attributed_to)
        if attributed_to
        else "",
    )
