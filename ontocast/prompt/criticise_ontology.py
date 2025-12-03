template_prompt = """
{preamble}

{intro_instruction}

{ontology_criteria}

{user_instruction}

{ontology_chapter}

{text_chapter}

{format_instructions}
"""

intro_instruction = """
You are given a text and an ontology.
You task is to evaluate the quality of the ontology with respect to the provided doc and provide a constructive critique of the ontology with respect to provided text.
"""


ontology_criteria = """
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
"""
