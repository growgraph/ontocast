"""Prompt for the document ontology plan: one call per document, before units render."""

template_prompt = """
{preamble}

{instruction}

{catalog}

{user_instruction}

{text}

{format_instructions}
"""

plan_instruction = """
# TASK

Plan what this document adds to the vocabulary of its domain. The ontology `{ontology_id}` ({title}) receives the new terms; their IRIs are its namespace plus the local name you give.

{description}

You see the whole document below. Several renderers will then each read one part of it and may mint ONLY the terms you plan here, so plan every class and property the document's facts will need, and nothing a single document would use alone.

# RULES

1. **Reuse first.** When a listed catalog term or a well-known vocabulary term (schema.org, SKOS, SOSA/SSN, QUDT, PROV-O, Dublin Core, FOAF) names the concept, propose it with `reuse` set to that term. Never propose a new term for a concept the catalog already has under another wording.
2. **Domain level.** Propose a term only if a second document of the same domain would use it. The particular cases this document reports (a named case, a person, a date, one measured value) are individuals or values in the facts, not terms.
3. **Values are not subclasses.** A distinction that varies from instance to instance (a kind, a method, a status, a category, a quantity being measured) is a property, with individuals or SKOS concepts as values. Never plan a family of sibling classes that differ by one word.
4. **One property per relation, not per quantity.** A measured quantity is the value of a general relation (a result, a value, a measured property) unless the catalog already models that quantity as its own property.
5. **Shallow.** Give every class a parent from the catalog, a well-known vocabulary or another planned term, at most two levels below an existing class. No chains of single-child classes. A property's parent is a more general property, or none; a class such as `schema:Property` is never one.
6. **Few and general.** A short plan of general terms beats a long plan of specific ones. Do not exceed {max_terms} terms.

For each term give: `kind` (class, object_property or datatype_property), `label`, `local_name` (PascalCase for a class, camelCase for a property), `parent`, a one-sentence domain-wide `definition`, `reuse` when an existing term already names it, and a `justification` that says why other documents of the domain need it.
"""

catalog_intro = """
# CATALOG TERMS

Existing terms you may reuse or attach to, as `CURIE | kind | label`:

{term_sheet}
"""

catalog_empty = """
# CATALOG TERMS

The catalog holds no terms yet: attach new classes to well-known vocabulary terms.
"""
