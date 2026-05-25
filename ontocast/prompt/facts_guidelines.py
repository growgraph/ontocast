"""Facts rendering operational guidelines (format-specific)."""

facts_instruction_shared = """\n\n
# OPERATIONAL GUIDELINES

1. Facts MUST use the fixed namespace `{facts_namespace}` with the prefix `cd:`. Local names for facts should not be capitalized.

1a. TWO-NAMESPACE CONTRACT (most important rule):
    - {domain_ontologies_clause}: schema elements only — classes (as `rdf:type` objects), predicates, and named individuals that exist verbatim in the ontology
    - `cd:`: ALL new instances extracted from the text, even if typed by an ontology class

    CORRECT: `cd:trial_1 a onto:Trial ; onto:hasJudgment cd:judgment_1 .`
    WRONG:   `onto:Trial_1 a onto:Trial .`  — new instance under ontology prefix, FORBIDDEN

1b. Every new `cd:` instance MUST carry `rdfs:label` with its canonical name from the source text (same language). URI local names and predicate-object literals do not substitute for a label.

2. Use the provided {domain_ontologies_clause} (below) and standard ontologies (RDFS, OWL, schema.org, etc.) to identify/infer entities, classes, types, and relationships
3. Thoroughly Extract and Link: extract all possible text mentions that correspond to entities, classes, types, or relationships defined in {domain_ontologies_clause}
4. Enforce typing: all `cd:` entities (facts) must be linked (e.g. using rdf:type) to entities from either {domain_ontologies_clause} or basic ontologies (RDFS, OWL, etc), e.g. rdfs:Class, rdf:Property, schema:Person, schema:Organization, etc.
5. Declare every namespace prefix you use (rdf, rdfs, owl, schema, domain ontologies, cd, etc.).
5a. PREFIX HYGIENE: Use **only** prefix aliases declared in the ontology context above. Do not invent alternative aliases.
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
10. {output_hygiene_rule}
11. Decide whether external evidence is needed for a retry and set `external_evidence_request`:
    - Set `initiate_search=true` only when ambiguity/term disambiguation/standards lookup materially blocks quality.
    - Otherwise keep `initiate_search=false`.
    - Provide concise `rationale` and optional focused `query_hints` when search is requested.
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


def format_facts_operational_guidelines(
    *,
    facts_namespace: str,
    domain_ontologies_clause: str,
    jsonld: bool,
) -> str:
    """Build operational guidelines for the active graph format."""
    literal_rules = facts_literal_rules_jsonld if jsonld else facts_literal_rules_turtle
    hygiene = facts_output_hygiene_jsonld if jsonld else facts_output_hygiene_turtle
    guidelines = facts_instruction_shared.format(
        domain_ontologies_clause=domain_ontologies_clause,
        facts_namespace=facts_namespace,
        literal_encoding_rules=literal_rules,
        output_hygiene_rule=hygiene,
    )
    if jsonld:
        guidelines += (
            "\n12. In structured output, express facts as a JSON-LD object "
            "(`@context` + `@graph`), not as a Turtle string. "
            "Map the examples above to compact IRIs and JSON-LD literal objects.\n"
        )
    return guidelines


facts_instruction_template = facts_instruction_shared
