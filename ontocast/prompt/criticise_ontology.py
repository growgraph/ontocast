from ontocast.onto.constants import DEFAULT_IRI

template_prompt = """
{preamble}

{intro_instruction}

{ontology_criteria}

{user_instruction}

{ontology_chapter}

{text_chapter}

{external_evidence}

{format_instructions}
"""

intro_instruction = """
You are given a text and an ontology.
You task is to evaluate the quality of the ontology with respect to the provided doc and provide a constructive critique of the ontology with respect to provided text.
"""


ontology_criteria = f"""
# TASK
Provide a constructive, actionable critique following these priorities:

## PRIMARY EVALUATION CRITERIA (in order of importance):
1. **Consistency**: No logical contradictions, proper use of OWL semantics
2. **Completeness**: All key domain concepts from text are represented
3. **Correctness**: Accurate relationships, proper datatypes, valid syntax
4. **Structure**: Appropriate class hierarchies and property definitions
5. **Abstraction**: Uses abstract classes/properties (no instances)
6. **Domain Coverage**: Includes implicit domain knowledge beyond literal text

## SCORING:
- 90-100: Excellent - minor refinements only
- 70-89: Good - some improvements needed
- 50-69: Adequate - significant gaps or errors
- 30-49: Poor - major structural issues
- 0-29: Inadequate - fundamental problems

## OUTPUT REQUIREMENTS:
1. Start with what works well (2-3 strengths)
2. Group fixes by severity: critical → important → minor
   - Use severity: "critical" (breaks semantic graph), "important" (significant gap), or "minor" (polish)
3. For each fix, provide:
   - Exact text evidence (quote from source)
   - Clear before/after using Turtle syntax
   - Actionable explanation
4. Systemic summary should identify patterns, not repeat individual fixes

## SPECIAL INSTRUCTIONS:
- For missing concepts: specify WHERE in the hierarchy they belong
- For relationship errors: explain the correct domain/range constraints
- For redundancies: suggest consolidation strategy
- Prioritize fixes that have cascading impact
- Enforce namespace hygiene: ontology classes/properties MUST NOT be modeled in `cd:` (`{DEFAULT_IRI}`), since `cd:` is reserved for facts/instances
- Treat external web evidence as optional support only. If evidence conflicts with source text or ontology context, prioritize source text and ontology context.
- Include `external_evidence_request` in your structured response:
  - Set `initiate_search=true` only when external web evidence is needed to resolve ambiguity.
  - Keep `initiate_search=false` when source text + ontology are sufficient.
  - Provide concise `rationale` and optional focused `query_hints` when search is requested.
"""
