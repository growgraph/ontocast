"""Prompt for document consolidation: one decision per minted term, before write."""

template_prompt = """
{preamble}

{instruction}

{terms}

{other_terms}

{user_instruction}

{format_instructions}
"""

consolidation_instruction = """
# TASK

The terms below were minted while reading one document of the domain described by the ontology `{ontology_id}` ({title}), and each was flagged for review. Before they are written to the shared catalog, decide each one. Each term is listed with why it is reviewed, its kind, label, definition, parents, whether the document plan named it (`planned`) or a renderer added it on its own (`candidate`), nearby catalog terms, checks that flagged it, and a short excerpt where the document uses it. The document's other minted terms are listed after them as possible merge targets; do not return actions for those.

Return one action per term:

- `keep`: a reusable domain term, correctly placed.
- `merge_into` with `target`: the same concept as another listed term, a catalog term or a well-known term. Prefer the existing catalog term.
- `generalise_to` with `target`: a reusable term with the wrong or too specific parent; `target` becomes its only parent.
- `demote_to_property_value`: a class that is really a value of some property (a material, a method, a category, one composition); it becomes a SKOS concept.
- `drop`: one document's case, a duplicate with nothing to merge into, or a term no fact needs.

Keep a term only if a second document of the domain would use it. A `candidate` with no action is dropped; a `planned` term with no action is kept. Name terms and targets exactly as listed.
"""

terms_intro = """
# TERMS

{term_blocks}
"""

other_terms_intro = """
# OTHER TERMS OF THIS DOCUMENT

Kept as they are; you may name one as a target, as `CURIE | kind | label`:

{term_lines}
"""
