"""Facts rendering operational guidelines (format-specific)."""

# Fallback vocabulary for bounded/approximate quantities when retrieval supplied
# no suitable class. QUDT by default because it is the most widely deployed
# quantity vocabulary -- but a default, not a compiled-in assumption: override
# via FACTS_QUANTITY_FALLBACK_VOCABULARY, or set it empty to forbid reaching
# outside the provided context at all.
DEFAULT_QUANTITY_FALLBACK_VOCABULARY: dict[str, str] = {
    "value_class": "qudt:QuantityValue",
    "numeric_value": "qudt:numericValue",
    "unit": "qudt:unit",
}

facts_instruction_shared = """\n\n
# OPERATIONAL GUIDELINES

1. Facts MUST use the fixed namespace `{facts_namespace}` with the prefix `cd:`. Local names for facts should not be capitalized (use lowercase_snake_case).

1a. TWO-NAMESPACE CONTRACT (most important rule):
    - {domain_ontologies_clause}: schema elements only — classes (as `rdf:type` objects), predicates, and named individuals that exist verbatim in the ontology
    - `cd:`: ALL new instances extracted from the text, even if typed by an ontology class

    CORRECT: `cd:trial_1 a onto:Trial ; onto:hasJudgment cd:judgment_1 .`
    WRONG:   `onto:Trial_1 a onto:Trial .`  — new instance under ontology prefix, FORBIDDEN

1b. Every new `cd:` instance MUST carry `rdfs:label` with its canonical name from the source text (same language). URI local names and predicate-object literals do not substitute for a label.

1c. CLASS VS. INSTANCE POSITION RULE (Anti-Mixing Constraint):
    - Ontology Classes (typically in PascalCase, e.g., `onto:ClinicalTrial`) represent abstract concepts, NOT concrete occurrences.
    - NEVER use an ontology Class IRI as the subject or object of a standard domain property. You must mint a unique `cd:` instance for that specific occurrence and type it using the class.
    - WRONG: `cd:patient_1 onto:underwentTrial onto:ClinicalTrial .` (Using a Class as a factual instance slot)
    - CORRECT: `cd:patient_1 onto:underwentTrial cd:trial_1 . cd:trial_1 a onto:ClinicalTrial .`

1d. SPECIFICITY RULE (ancestor vs. descendant terms):
    - The provided ontology context may include both an ancestor (abstract) term and one or more
      `rdfs:subClassOf` / `rdfs:subPropertyOf` descendants of it. They appear together
      deliberately so you can read domain/range context — not as interchangeable synonyms.
    - Always use the most specific descendant class or property whose stated domain, range, and
      restrictions fit the concrete entities in the fact. Fall back to the ancestor only when no
      descendant fits or the source text is genuinely stated at the abstract level.
    - WRONG: `cd:assembly_1 onto:hasPart cd:widget_1 .` when `onto:hasComponent` is a
      `rdfs:subPropertyOf` of `onto:hasPart` and its domain/range fit the subject and object.
    - CORRECT: `cd:assembly_1 onto:hasComponent cd:widget_1 .`

2. Use the provided domain ontology namespace(s) above and standard ontologies (RDFS, OWL, schema.org, etc.) to identify/infer entities, classes, types, and relationships
3. Thoroughly Extract and Link: extract all possible text mentions that correspond to entities, classes, types, or relationships defined in the domain ontology namespace(s) above
4. Enforce typing: all `cd:` entities (facts) are data instances and must be linked via `rdf:type` to a valid operational Class from either the domain ontology namespace(s) above or standard core vocabularies (e.g., `schema:Person`, `schema:Organization`, `onto:Trial`).
   - CRITICAL: NEVER type a `cd:` instance as `rdfs:Class` or `rdf:Property`. You are extracting data occurrences, not rewriting or defining the schema.
5. Declare every namespace prefix you use (rdf, rdfs, owl, schema, domain ontologies, cd, etc.).
5a. PREFIX HYGIENE: Use **only** prefix aliases declared in the ontology context above. Do not invent alternative aliases.
6. CRITICAL - Entity Matching & Namespace Isolation Protocol:
   - Understand the Ontology Contents: The provided ontology contains the schema (Classes and Properties). It may also contain a small set of static Reference Individuals (e.g., fixed status constants, countries, or controlled vocabularies). It does NOT contain the dynamic data instances described in your source text.
   - The Target Lookup Rule: BEFORE creating a `cd:` entity, check if the text mention refers to one of those static Reference Individuals existing verbatim in the provided ontology context (declared as `owl:NamedIndividual` or an explicit individual token).
   - Class vs. Individual Boundary: A Class in the ontology (e.g., `onto:Trial`) is an abstract concept, NOT an instance. Finding a Class that matches the type of your text mention does NOT mean you found a matching individual. For any text occurrence, you must create a NEW `cd:` instance and type it with that Class.
   - Namespace Isolation Guardrail: NEVER mint or invent new IRIs inside the domain ontology namespace. The domain ontology namespace is strictly READ-ONLY. If a real-world entity mentioned in the text is not explicitly present in the provided ontology as a verbatim reference individual, it is a NEW fact and MUST receive a `cd:` namespace IRI.
   - Exact Matching: If (and only if) an exact matching reference individual is already defined in the ontology, use its canonical IRI directly instead of duplicating it in `cd:`. Match by semantic meaning and language variants (`rdfs:label`), preserving character-for-character casing.
   - Safe Fallback: If you cannot find an explicit, pre-declared reference individual in the provided ontology for a text mention, treat it as a new data instance and place it under the `cd:` namespace.
   
6a. Opaque Identifier Ontologies (Wikidata-style Q/P codes, hashes, UUIDs):
   - When ontology IRIs contain opaque local names (Q-numbers, P-numbers, hash strings, numeric IDs),
     entity identity is determined EXCLUSIVELY by `rdfs:label`, `rdfs:comment`, skos:altLabel — not the IRI fragment.
   - Use the TERM INDEX (if provided below the ontology) to map text mentions to their canonical IRI.
   - NEVER construct an IRI by appending a label string to the ontology namespace
     (e.g. `onto:culture` is ALWAYS wrong — the correct IRI is whatever appears in the ontology with `rdfs:label "culture"`)
   - NEVER invent or guess a Q/P code — only use codes that appear explicitly in the provided ontology.
   - For property domain/range chains: resolve referenced opaque IRIs to their labels before deciding
     which subject/object types are valid for a given property.

7. Maximize atomicity: decompose complex facts and complex literals into simple subject-predicate-object statements (e.g. decompose person's first name and last name).

8. Literals and Quantity Values:
   - Use appropriate XSD datatypes: xsd:integer, xsd:decimal, xsd:float,
     xsd:date, xsd:dateTime. Dates use ISO 8601.
{literal_encoding_rules}
   - NEVER encode a numeric measurement as xsd:string, even if the source
     text is approximate or bounded.
   - When a measurement appears with an epistemic qualifier — approximation
     (∼, ~, ≈, ca., about), bound (<, >, ≤, ≥, up to, at least, more than,
     exceeding), range (X–Y, X to Y), or uncertainty (X ± Y) — decompose
     it into a structured node:
       * Search the provided ontology for a class representing approximate or
         bounded quantity values (e.g. a QuantityValue subclass, a
         MeasuredValue class, or equivalent).
       * If found, instantiate it and populate exactly the numeric properties
         the ontology documents for the stated form — its property
         definitions, comments, and scope notes say which property carries
         which numeric role — plus its qualifier properties for the epistemic
         marker. Record each number once, on the property whose documented
         role matches the source statement; never write the same number into
         two different numeric properties of one node, and never use
         bound properties for a value the source states exactly.
{quantity_fallback_clause}
   - Prose restatements of a measurement in dcterms:description are redundant
     once typed numeric properties exist — omit them.
   - VERBATIM VALUES AND UNITS: transcribe every measurement with its exact
     source value and source unit. NEVER convert units — not between scale
     prefixes, not between related quantity kinds, not into a "preferred"
     unit — and never round or re-derive values —
     canonicalization happens downstream in code. If the exact source unit
     has no individual in the provided context, keep the value verbatim and
     preserve the unit token per rule 8a's fallback.

8a. OBJECT PROPERTIES TAKE IRIs, NOT STRINGS:
   - A property whose range is a class (declared `owl:ObjectProperty`, or with an
     `rdfs:range` pointing to a class — e.g. `qudt:unit` with range `qudt:Unit`)
     MUST have an IRI as its object, never a string literal.
   - Resolve the surface token to an individual in the provided ontology context by
     matching its `rdfs:label`, `skos:notation`, symbol, or code annotations,
     preserving case exactly (a lowercase symbol and its uppercase variant denote
     DIFFERENT individuals — match character-for-character).
   - WRONG:   `cd:value_1 qudt:unit "nm" .` — string on an object property
   - CORRECT: `cd:value_1 qudt:unit unit:NanoM .` — when that individual's
     label/notation matches the text token; use only unit IRIs that appear
     verbatim in the provided context, never invent one
   - Only if NO individual in the provided context matches the token: keep the raw
     token in a literal-code annotation property (e.g. `qudt:ucumCode` or
     `rdfs:comment`) on the `cd:` node instead of putting a string on the
     object property, so the surface form is preserved without breaking typing.

9. To extract data from tables, use CSV on the Web (CSVW) to describe tables.

10. {output_hygiene_rule}

{search_guidelines}

# FINAL STRUCTURAL VALIDATION CHECKLIST
Before finalizing output, verify:
- Are there any PascalCase IRIs acting as graph subjects or objects (excluding `rdf:type` targets)? If yes, fix them into `cd:` instances.
- Did you use `rdfs:Class` or `rdf:Property` anywhere as an instance type? If yes, replace it with its meaningful domain-specific concept class.
"""

facts_literal_rules_turtle = """\
   - Dates: ISO 8601 with Turtle typing (e.g., "2024-01-15"^^xsd:date)
   - Numbers: typed literals (e.g., "42"^^xsd:integer, "99.95"^^xsd:decimal)
   - Currencies: include currency codes (e.g., "1000"^^xsd:decimal with schema:priceCurrency "USD")"""

facts_literal_rules_jsonld = """\
   - Dates: {"@value": "2024-01-15", "@type": "xsd:date"} — never use Turtle ^^ syntax
   - Numbers: {"@value": "42", "@type": "xsd:integer"} etc.
   - Language tags: {"@value": "...", "@language": "en"}"""

facts_output_hygiene_turtle = (
    "No comments in Turtle: output must contain only @prefix declarations and triples "
    "(no lines starting with #)."
)

facts_output_hygiene_jsonld = (
    "Output strictly valid JSON for graph fields: no comments, no trailing prose, "
    "no Turtle ^^ or @prefix syntax inside JSON."
)


_QUANTITY_FALLBACK_TEMPLATE = """       * If no such class is found in the domain ontology, use {value_class}
         as the type and attach the numeric parts with {numeric_value} /
         {unit}, adding a plain qualifier annotation (e.g. rdfs:comment
         or a well-known approximation property)."""

_QUANTITY_FALLBACK_NONE = """       * If no such class is found in the domain ontology, keep the numeric
         parts as typed literals on the entity itself and record the
         qualifier with rdfs:comment. Do NOT borrow a class from a
         vocabulary outside the provided context."""


def format_quantity_fallback_clause(vocabulary: dict[str, str]) -> str:
    """Render the bounded-quantity fallback for a configured vocabulary.

    The fallback is what the renderer reaches for when retrieval supplied no
    bounded-quantity class. Which vocabulary that is belongs to the deployment,
    not to the prompt: a catalog modelling quantities with anything other than
    QUDT would otherwise be told to emit QUDT terms it never declared.

    Args:
        vocabulary: Role -> term mapping (``FACTS_QUANTITY_FALLBACK_VOCABULARY``).
            Empty disables the fallback and keeps the renderer inside the
            provided context.

    Returns:
        str: The guideline bullet for the configured fallback.
    """
    if not vocabulary:
        return _QUANTITY_FALLBACK_NONE
    return _QUANTITY_FALLBACK_TEMPLATE.format(
        value_class=vocabulary.get("value_class", "a quantity-value class"),
        numeric_value=vocabulary.get("numeric_value", "its numeric-value property"),
        unit=vocabulary.get("unit", "its unit property"),
    )


def format_facts_operational_guidelines(
    *,
    facts_namespace: str,
    domain_ontologies_clause: str,
    jsonld: bool,
    quantity_fallback_vocabulary: dict[str, str] | None = None,
    search_guidelines: str = "",
) -> str:
    """Build operational guidelines for the active graph format."""
    literal_rules = facts_literal_rules_jsonld if jsonld else facts_literal_rules_turtle
    hygiene = facts_output_hygiene_jsonld if jsonld else facts_output_hygiene_turtle
    guidelines = facts_instruction_shared.format(
        domain_ontologies_clause=domain_ontologies_clause,
        facts_namespace=facts_namespace,
        literal_encoding_rules=literal_rules,
        output_hygiene_rule=hygiene,
        quantity_fallback_clause=format_quantity_fallback_clause(
            quantity_fallback_vocabulary
            if quantity_fallback_vocabulary is not None
            else DEFAULT_QUANTITY_FALLBACK_VOCABULARY
        ),
        search_guidelines=search_guidelines,
    )
    if jsonld:
        # 10a., not 11. or 12.: rule 11 is the conditional search guideline
        # injected above and is absent whenever web grounding is off, so a fixed
        # number either collides with it or leaves a gap in the default prompt.
        # The template already uses this suffix convention for 1a. and 6a.
        guidelines += (
            "\n10a. In structured output, express facts as a JSON-LD object "
            "(`@context` + `@graph`), not as a Turtle string. "
            "Map the examples above to compact IRIs and JSON-LD literal objects.\n"
        )
    return guidelines
