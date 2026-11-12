"""Prompt for catalog curation: one decision per flagged minted term, across documents."""

template_prompt = """
{preamble}

{instruction}

{terms}

{other_terms}

{format_instructions}
"""

curation_instruction = """
# TASK

The ontology `{ontology_id}` ({title}) is shared by every document processed so far. The terms below were added to it by individual documents, and each was flagged for review: used by the facts of one document only, used by none, named like another term, one of a family of sibling classes that differ by one word, or naming a term the ontology's authors already defined. Each term is listed with why it is reviewed, its kind, label, definition, parents and the document that added it.

Return one action per term:

- `keep`: a reusable domain term, correctly placed, that more documents of the domain will use.
- `merge_into` with `target`: the same concept as another listed term, an authored term or a well-known term. Prefer the authored term.
- `generalise_to` with `target`: a reusable term with the wrong or too specific parent; `target` becomes its only parent.
- `demote_to_property_value`: a class that is really a value of some property (a kind, a method, a category); it becomes a SKOS concept.
- `drop`: one document's case or a term no fact needs.

A term with no action is kept. The ontology's other added terms are listed after them as possible merge targets; do not return actions for those. Name terms and targets exactly as listed.
"""
