"""Facts rendering operational guidelines (format-specific).

The quantity rules form one chapter (:func:`format_quantity_chapter`), rendered
only for a content unit whose text states a number: a text that states no
quantity is not told how to model one.
"""

from collections.abc import Mapping

from rdflib import Graph, URIRef

from ontocast.onto.well_known import (
    well_known_prompt_list,
    well_known_statement_classes,
    well_known_statement_roles,
    well_known_vocabulary,
)
from ontocast.tool.facts_validation.statements import StatementContract, own_terms

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
    - {domain_ontologies_clause} (see the ONTOLOGY section below): schema elements only — classes (as `rdf:type` objects), predicates, and named individuals that exist verbatim in the ontology
    - `cd:`: ALL new instances extracted from the text, even if typed by an ontology class

    CORRECT: `cd:project_1 a onto:Project ; onto:hasOutcome cd:outcome_1 .`
    WRONG:   `onto:Project_1 a onto:Project .`  — new instance under ontology prefix, FORBIDDEN

1b. Every new `cd:` instance MUST carry `rdfs:label` with its canonical name from the source text (same language). URI local names and predicate-object literals do not substitute for a label.

1c. CLASS VS. INSTANCE POSITION RULE (Anti-Mixing Constraint):
    - Ontology Classes (typically in PascalCase, e.g., `onto:Project`) represent abstract concepts, NOT concrete occurrences.
    - NEVER use an ontology Class IRI as the subject or object of a standard domain property. You must mint a unique `cd:` instance for that specific occurrence and type it using the class.
    - WRONG: `cd:person_1 onto:worksOn onto:Project .` (Using a Class as a factual instance slot)
    - CORRECT: `cd:person_1 onto:worksOn cd:project_1 . cd:project_1 a onto:Project .`

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

1e. DOMAIN FIT: a property applies only to a subject of its declared domain. Never use a
    {link_kinds} as a generic "relates to" link: `prov:used` takes an
    activity as its subject{sensor_clause}. A statement whose subject
    is not of the property's domain is removed.

{vocabulary_precedence_rule}
3. EXTRACT STATEMENTS: extract every statement the text makes in the terms of the provided ontology — the relations between things and the values it gives them. Mint a `cd:` instance only for a thing a statement is about: it is the subject or the object of a relation, or it holds a value. A mention the text says nothing about is not a fact; do not mint it.
{quantity_chapter}{connect_rule}
{statement_rule}{attribution_rule}{typing_rule}
   - CRITICAL: NEVER type a `cd:` instance as `rdfs:Class` or `rdf:Property` (you are extracting data occurrences, not rewriting or defining the schema), nor as `owl:Thing` alone — a vacuous type is no type.
   - Do NOT mint an entity for a bare number, a citation marker, or a reference token (`[12]`, "19", equation numbers). These are typography, not facts.{cited_work_clause}
5. Declare every namespace prefix you use (rdf, rdfs, owl, schema, domain ontologies, cd, etc.).
5a. PREFIX HYGIENE: Write each namespace with the prefix this prompt shows for it: the domain ontologies of rule 1a, the well-known vocabularies of rule 2, and the names in the ONTOLOGY section. Do not invent alternative aliases.
6. CRITICAL - Entity Matching & Namespace Isolation Protocol:
   - Understand the Ontology Contents: The provided ontology contains the schema (Classes and Properties). It may also contain a small set of static Reference Individuals (e.g., fixed status constants, countries, or controlled vocabularies). It does NOT contain the dynamic data instances described in your source text.
   - The Target Lookup Rule: BEFORE creating a `cd:` entity, check if the text mention refers to one of those static Reference Individuals existing verbatim in the provided ontology context (declared as `owl:NamedIndividual` or an explicit individual token).
   - Class vs. Individual Boundary: A Class in the ontology (e.g., `onto:Project`) is an abstract concept, NOT an instance. Finding a Class that matches the type of your text mention does NOT mean you found a matching individual. For any text occurrence, you must create a NEW `cd:` instance and type it with that Class.
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

8. Literals:
   - Use appropriate XSD datatypes: xsd:integer, xsd:decimal, xsd:float,
     xsd:date, xsd:dateTime. Dates use ISO 8601.
   - A node that stands for a date or a time (an instant, a day, a month, a
     year) carries its ISO 8601 value as a typed literal (xsd:dateTime,
     xsd:date, xsd:gYearMonth, xsd:gYear) through the ontology's property for
     it, else rdf:value. A label alone does not say which date a node is.
{literal_encoding_rules}

8a. OBJECT PROPERTIES TAKE IRIs, NOT STRINGS:
   - A property whose range is a class (declared `owl:ObjectProperty`, or with an
     `rdfs:range` pointing to a class) MUST have an IRI as its object, never a
     string literal.
   - Resolve the surface token to an individual in the provided ontology context by
     matching its `rdfs:label`, `skos:notation`, symbol, or code annotations,
     preserving case exactly (a lowercase symbol and its uppercase variant denote
     DIFFERENT individuals — match character-for-character).
   - WRONG:   `cd:item_1 onto:hasStatus "active"` — string on an object property
   - CORRECT: `cd:item_1 onto:hasStatus onto:Active` — when that individual's
     label/notation matches the text token

9. To extract data from tables, use CSV on the Web (CSVW) to describe tables.

10. {output_hygiene_rule}

{search_guidelines}

# FINAL STRUCTURAL VALIDATION CHECKLIST
Before finalizing output, verify:
- Are there any PascalCase IRIs acting as graph subjects or objects (excluding `rdf:type` targets)? If yes, fix them into `cd:` instances.
- Did you use `rdfs:Class` or `rdf:Property` anywhere as an instance type? If yes, replace it with its meaningful domain-specific concept class.
- For each `cd:` instance not linked to any other instance: does the text state a relation you missed? Add it only if the text states it (rule 3b); never invent one. If it is linked to nothing and holds no value, remove it (rule 3).
{quantity_checklist}{statement_checklist}"""

facts_literal_rules_turtle = """\
   - Dates: ISO 8601 with Turtle typing (e.g., "2024-01-15"^^xsd:date)
   - Numbers: typed literals (e.g., "42"^^xsd:integer, "99.95"^^xsd:decimal){currency_rule}"""

facts_literal_rules_jsonld = """\
   - Dates: {"@value": "2024-01-15", "@type": "xsd:date"} — never use Turtle ^^ syntax
   - Numbers: {"@value": "42", "@type": "xsd:integer"} etc.
   - Language tags: {"@value": "...", "@language": "en"}
   - IRI objects: {"@id": "cd:item_1"} — a bare string ("cd:item_1") is a literal, not an IRI"""

facts_output_hygiene_turtle = (
    "No comments in Turtle: output must contain only @prefix declarations and triples "
    "(no lines starting with #)."
)

facts_output_hygiene_jsonld = (
    "Output strictly valid JSON for graph fields: no comments, no trailing prose, "
    "no Turtle ^^ or @prefix syntax inside JSON."
)


_QUANTITY_FALLBACK_TEMPLATE = """       * If no such class is found in the domain ontology, type the value node
         with the fallback value class of TERMS and attach the numeric parts
         with its number and measurement-unit links; the value's label keeps
         the qualifier as the text writes it."""

_QUANTITY_FALLBACK_NONE = """       * If no such class is found in the domain ontology, keep the numeric
         parts as typed literals on the entity itself. Do NOT borrow a class
         from a vocabulary outside the provided context."""


def format_quantity_fallback_clause(vocabulary: dict[str, str]) -> str:
    """Render the bounded-quantity fallback for a configured vocabulary.

    The fallback is what the renderer reaches for when retrieval supplied no
    bounded-quantity class: the value class and links the TERMS bullet
    (:func:`quantity_terms`) names from the configured vocabulary.

    Args:
        vocabulary: Role -> term mapping (``FACTS_QUANTITY_FALLBACK_VOCABULARY``).
            Empty disables the fallback and keeps the renderer inside the
            provided context.

    Returns:
        str: The guideline bullet for the configured fallback.
    """
    if not vocabulary:
        return _QUANTITY_FALLBACK_NONE
    # The terms themselves are named once, in the TERMS bullet.
    return _QUANTITY_FALLBACK_TEMPLATE


# The vocabulary rules of the facts prompt, per FACTS_VOCABULARY. Both modes
# use the provided ontology first and the well-known vocabularies
# (ontocast.onto.well_known) where it has no term. ``strict`` stops there:
# nothing those cannot type or relate is extracted. ``open`` also admits any
# other established vocabulary the model knows.
VOCABULARY_MODES = ("strict", "open")

_PRECEDENCE_REASON = (
    "   - Prefer the ontology even when another term would also fit. The reason "
    "is mechanical, not stylistic: the deployment validates this graph against "
    "shapes written for the provided ontology and queries it in the ontology's "
    "terms; a substitute is invisible to every shape and every such query."
)
_NO_VACUOUS_TYPE = (
    "   - Never type an instance with `owl:Thing`, `rdfs:Resource` or "
    "`schema:Thing`, which state nothing."
)

_VOCABULARY_PRECEDENCE_STRICT = """2. VOCABULARY PRECEDENCE (strict vocabulary): use the ONTOLOGY section of this prompt first, for every class and property it has. Where it has none, use a term of these well-known vocabularies, which you know: {vocabularies}.
   - Use only terms that exist in these vocabularies, under their real namespace IRIs; never invent a term, and never mint one in an ontology's namespace.
   - No other vocabulary is available. A thing that neither the provided ontology nor a well-known vocabulary has a class for, or a relation neither has a property for, is not extracted.
{reason}
{vacuous}"""

_VOCABULARY_PRECEDENCE_OPEN = """2. VOCABULARY PRECEDENCE (open vocabulary): use the ONTOLOGY section of this prompt first, for every class and property it has. Where it has none, use a term of these well-known vocabularies: {vocabularies}. Where none of them has one, use a term of another established vocabulary you know well, under its real namespace IRI.
   - Vocabularies outside the ontology are a LAST RESORT, in that order, for concepts the provided ontology genuinely does not cover. They are not an equal alternative.
   - Use only terms that exist in a vocabulary; never invent a term, and never mint one in an ontology's namespace.
{reason}
{vacuous}"""

# One chapter for both modes: a mode narrows the terms, never the
# measurements. Every stated quantity has a carrier -- the observation form
# needs only an observation class and a result property, which the well-known
# vocabularies always offer.
_QUANTITY_CHAPTER = """3a. QUANTITIES: this text states numbers.
   - QUANTITATIVE COMPLETENESS: extract EVERY quantitative statement in the text — every measured value, duration, condition, concentration, ratio, threshold, and uncertainty, including a dimensionless ratio or a parameter of a model. A number attached to a stated quantity is a fact and must reach the graph with its verbatim value and unit; a bare citation, page, figure, or equation token is typography and must not (rule 4). Completeness of the measured content is judged, so silently skipping a stated measurement is a defect, not a simplification.
   - OBSERVATION FORM: every quantity can be expressed. Carry the value with the property the provided ontology offers for it, and where it offers none, state the quantity as an observation whose result is the value node itself (pointing straight at the typed value — never at a separate result node that holds the value), whose observed property names what was measured, and whose feature of interest is the instance the text says was measured (rule 3c).
   - EXACT QUANTITY: the observed property is an individual chosen under rule 6. Reuse a catalog individual only when it names exactly the quantity stated, its kind and what it is relative to. A change, difference or ratio between two states is a quantity of its own. Otherwise mint a `cd:` instance of the observable-property class TERMS names.
{terms}
   - A STRUCTURED VALUE (a quantity value, a range, a bound — a node that
     only holds a value) has no identity of its own: write it as a blank node
     (Turtle `[ … ]`, JSON-LD an object without `@id`), as the object of the
     property that carries it — the property of the entity, observation or
     result it measures that the provided ontology offers.
{carrier_fallback_clause}
     Never mint a `cd:`
     IRI for a value, and never leave a value standing alone: a value no
     statement carries is dropped downstream. Give it an `rdfs:label` with
     its verbatim text.
     CORRECT: `cd:item_1 onto:hasLength [ a onto:QuantityValue ; rdfs:label "about 20 m" ; … ] .`
     WRONG:   `cd:value_1 a onto:QuantityValue ; rdfs:label "about 20 m" .` — a value with no carrier
   - When a measurement appears with an epistemic qualifier — approximation
     (∼, ~, ≈, ca., about), bound (<, >, ≤, ≥, up to, at least, more than,
     exceeding), range (X–Y, X to Y), or uncertainty (X ± Y) — decompose
     it into a structured value (a blank node under its carrier):
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
   - NEVER encode a numeric measurement as xsd:string, even if the source
     text is approximate or bounded.
   - Prose restatements of a measurement in dcterms:description are redundant
     once typed numeric properties exist — omit them.
   - VERBATIM VALUES AND UNITS: transcribe every measurement with its exact
     source value and source unit. NEVER convert units — not between scale
     prefixes, not between related quantity kinds, not into a "preferred"
     unit — and never round or re-derive values —
     canonicalization happens downstream in code.
   - A measurement unit is an individual on a unit-valued property (rule 8a).
     A unit IRI comes from the provided context first. When no individual there
     matches the token, use the QUDT unit (`unit:`) for exactly that unit, only
     if you are certain of its QUDT name (`unit:M`, `unit:KiloM-PER-HR`,
     `unit:PERCENT`); never invent or approximate a unit name. When no
     measurement unit fits, add no unit link: the value's label keeps the
     verbatim text.
"""

_QUANTITY_CHECKLIST = (
    "- Is every structured value a blank node, the object of the property "
    "that carries it (rule 3a)?\n"
)

_ATTRIBUTION = """3d. ATTRIBUTED STATEMENTS: a statement the text attributes to other work — by a reference marker, or by saying that others reported it — is extracted like any other, and links to that work with `{attributed_to}`, from the node that states it: the statement node when there is one, else the instance it is about. The cited work is one `cd:` instance{work_type}, labelled with the reference as the text writes it, with its identifier when the text gives one.
"""

_CITED_WORK = (
    " The work a citation marker refers to is an entity (rule 3d); the marker is not."
)

_SENSOR_CLAUSE = (
    ", a thing made from another takes `prov:wasDerivedFrom`, and a\n"
    "    sensor's readings take `sosa:madeBySensor` on the observation"
)
_NO_SENSOR_CLAUSE = ", and a thing made from another takes `prov:wasDerivedFrom`"

_STATEMENT_CHECKLIST = (
    "- Does every statement (an instance of an observation class, or of any "
    "class under one) carry every role its class requires: the instance it is "
    "about (rule 3c) and the property it reports? If one is missing, add the "
    "link the text states.\n"
)

_CURRENCY_TURTLE = (
    '\n   - Currencies: give the amount as a number (e.g., "1000"^^xsd:decimal) '
    "with the currency's code as the text writes it (e.g., schema:priceCurrency)"
)

_SUBJECT = """3c. EVERY STATEMENT NAMES ITS SUBJECT: an observation — of the ontology's observation class or of any class under it — and any value that is not the object of the property carrying it, is a fact only together with what it is about. Link each one to that instance: `sosa:hasFeatureOfInterest` from the observation, or the ontology's property for the same link, in the direction it runs. The subject is the instance the sentence measures or describes; when the sentence does not repeat it, it is the instance the surrounding text is about — the thing under discussion that the measurement belongs to. That link is stated by the text, not guessed. Mint the subject when the text names it and it is not yet in the graph. An observation that names no subject is a defect."""

_CONNECT = """3b. CONNECT WHAT THE TEXT CONNECTS: when the text relates two instances — one is part of, belongs to, takes part in, or describes the other — assert that link with a property of the provided ontology, or, when it has none for that relation, {fallback} (rule 2). A `cd:` instance left with only its type, label and literal values often means a stated relation was missed; look for it. Assert only relations the text states or directly entails: never invent or guess a link. An instance the text relates to nothing these vocabularies can express, and that holds no value, is a mention (rule 3) — leave it out. When the text mentions one thing in several places, use one `cd:` IRI for it throughout this extraction instead of minting one per mention."""

_TYPING = """4. Enforce typing: all `cd:` entities (facts) are data instances and must be linked via `rdf:type` to a valid operational Class — from the provided ontology wherever one fits (e.g. `onto:Project`), and only otherwise {fallback} (e.g. `schema:Person`). An instance no class of these fits is not minted."""

_CARRIER = (
    """     If the provided ontology has no property to carry it, use {carrier}."""
)


def _term_name(graph: Graph | None, iri: str) -> str:
    """A prefixed name for ``iri``: a well-known vocabulary's prefix, else one
    ``graph`` binds; the IRI itself when neither applies."""
    vocabulary = well_known_vocabulary(iri)
    if vocabulary is not None:
        for namespace in vocabulary.namespaces:
            if iri.startswith(namespace):
                return f"{vocabulary.prefix}:{iri[len(namespace) :]}"
    if graph is not None:
        try:
            prefix, _, local = graph.namespace_manager.compute_qname(
                URIRef(iri), generate=False
            )
        except (KeyError, ValueError):
            return f"<{iri}>"
        return f"{prefix}:{local}"
    return f"<{iri}>"


#: The quantity roles of the TERMS bullet, in order: where the contract keeps
#: the role, and its name for several terms and for one.
_QUANTITY_ROLES = (
    ("statement_classes", "observation classes", "observation class"),
    ("subject", "subject links", "subject link"),
    ("property", "observed-property links", "observed-property link"),
    (
        "property_classes",
        "observable-property classes",
        "observable-property class",
    ),
    ("result", "result links", "result link"),
    ("value_classes", "value classes", "value class"),
)


def _namespace(iri: str) -> str:
    for separator in ("#", "/"):
        head, sep, local = iri.rpartition(separator)
        if sep and local:
            return head + sep
    return iri


def _registry_fallback() -> list[tuple[str, str]]:
    """``(role, IRI)`` the well-known registry offers for the observation form.

    The links are the statement roles the registry labels for the prompt; the
    class is the statement class the labelled property link serves.
    """
    labelled = {
        role.role: (iri, role)
        for iri, role in sorted(well_known_statement_roles().items())
        if role.label is not None and role.direction == "forward"
    }
    found: list[tuple[str, str]] = []
    if "property" in labelled:
        iri, role = labelled["property"]
        statement_classes = well_known_statement_classes()
        found.extend(
            ("statement_classes", _namespace(iri) + name)
            for name in sorted(role.classes)
            if _namespace(iri) + name in statement_classes
        )
    for key in ("subject", "property", "result"):
        if key not in labelled:
            continue
        iri, role = labelled[key]
        found.append((key, iri))
        if key == "property" and role.range_class is not None:
            found.append(("property_classes", _namespace(iri) + role.range_class))
    return found


def observable_property_classes(
    ontology_graph: Graph | None, contract: StatementContract
) -> list[str]:
    """The classes to mint an observed property in: the catalog's own in the
    unit's context, else the one the well-known registry offers."""
    own = own_terms(ontology_graph, contract.property_classes)
    if own:
        return own
    return [iri for key, iri in _registry_fallback() if key == "property_classes"]


def quantity_terms(
    ontology_graph: Graph | None,
    contract: StatementContract,
    vocabulary: Mapping[str, str] | None,
    *,
    limit: int = 6,
) -> str:
    """The TERMS bullet of the quantity chapter: catalog terms first.

    The catalog's terms are those of the unit's ontology context that the
    contract gives a quantity role and no well-known vocabulary owns. The
    fallback comes from the well-known registry (the statement class, the
    role links it offers and the class of an observed property) and from the
    configured quantity vocabulary (the value class, its number and its
    measurement unit).

    Args:
        ontology_graph: The unit's ontology context.
        contract: The catalog's statement contract.
        vocabulary: ``FACTS_QUANTITY_FALLBACK_VOCABULARY``.
        limit: Cap on the catalog terms named per role.

    Returns:
        The bullet, ending with a newline.
    """
    singular = {key: one for key, _, one in _QUANTITY_ROLES}

    catalog: list[str] = []
    for key, title, _ in _QUANTITY_ROLES:
        if key == "statement_classes":
            pool = contract.statement_classes
        elif key == "value_classes":
            pool = contract.value_classes
        elif key == "property_classes":
            pool = contract.property_classes
        else:
            pool = contract.links(key).forward
        terms = own_terms(ontology_graph, pool)[:limit]
        if terms:
            names = ", ".join(f"`{_term_name(ontology_graph, t)}`" for t in terms)
            catalog.append(f"{title} {names}")

    fallback = [
        f"{singular[key]} `{_term_name(None, iri)}`"
        for key, iri in _registry_fallback()
    ]
    for role, title in (
        ("value_class", "value class"),
        ("numeric_value", "number"),
        ("unit", "measurement unit"),
    ):
        term = (vocabulary or {}).get(role)
        if term:
            fallback.append(f"{title} `{term}`")

    head = (
        "the ONTOLOGY section's terms come first: " + "; ".join(catalog) + "."
        if catalog
        else "the ONTOLOGY section's terms come first, where it has them."
    )
    return (
        f"   - TERMS: {head} Where it has none for a role, use: "
        + "; ".join(fallback)
        + ".\n"
    )


def format_quantity_chapter(
    vocabulary: Mapping[str, str] | None,
    terms: str = "",
) -> str:
    """The quantity chapter (rule 3a), for a content unit whose text states a number.

    One chapter holds every quantity rule: completeness, the observation form,
    the exact-quantity rule for the observed property, structured values and
    their qualifiers, the fallback vocabulary and measurement units.

    Args:
        vocabulary: ``FACTS_QUANTITY_FALLBACK_VOCABULARY``; its ``carrier`` role
            names the property carrying a quantity whose entity has none in
            the ontology, and an empty mapping forbids the fallback.
        terms: The TERMS bullet (:func:`quantity_terms`).

    Returns:
        The chapter, ending with a newline.
    """
    carrier = (vocabulary or {}).get("carrier") or "the observation form above"
    return _QUANTITY_CHAPTER.format(
        terms=terms.rstrip("\n"),
        carrier_fallback_clause=_CARRIER.format(carrier=carrier),
        quantity_fallback_clause=format_quantity_fallback_clause(
            dict(vocabulary)
            if vocabulary is not None
            else DEFAULT_QUANTITY_FALLBACK_VOCABULARY
        ),
    )


def format_attribution_rule(citation_vocabulary: Mapping[str, str] | None) -> str:
    """Rule 3d: a statement attributed to other work links to that work.

    Empty when the citation vocabulary names no ``attributed_to`` term.
    """
    vocabulary = citation_vocabulary or {}
    attributed_to = vocabulary.get("attributed_to")
    if not attributed_to:
        return ""
    work_class = vocabulary.get("work_class")
    return _ATTRIBUTION.format(
        attributed_to=attributed_to,
        work_type=f" typed `{work_class}`" if work_class else "",
    )


def format_vocabulary_rules(
    vocabulary: str,
    additional_namespaces: tuple[str, ...] | list[str] = (),
    *,
    statement_rules: bool = False,
) -> dict[str, str]:
    """The prompt's vocabulary rules for one mode.

    Args:
        vocabulary: ``strict`` or ``open`` (``FACTS_VOCABULARY``).
        additional_namespaces: The deployment's own shared vocabularies
            (``FACTS_ADDITIONAL_STANDARD_NAMESPACES``), named beside the
            well-known ones.
        statement_rules: Whether the unit's statement rules apply
            (``UnitPromptScope.statements``); without them the vocabularies of
            statements and measurements are not named.

    Returns:
        dict[str, str]: Template fields ``vocabulary_precedence_rule``,
        ``connect_rule`` and ``typing_rule``.

    Raises:
        ValueError: If ``vocabulary`` is not a known mode.
    """
    if vocabulary not in VOCABULARY_MODES:
        raise ValueError(
            f"unknown facts vocabulary {vocabulary!r}; expected one of {VOCABULARY_MODES}"
        )
    vocabularies = well_known_prompt_list(measurement=statement_rules) + "".join(
        f", <{namespace}>" for namespace in additional_namespaces
    )
    strict = vocabulary == "strict"
    precedence = (
        _VOCABULARY_PRECEDENCE_STRICT if strict else _VOCABULARY_PRECEDENCE_OPEN
    )
    fallback = (
        "a term of a well-known vocabulary"
        if strict
        else "a term of a well-known or another established vocabulary"
    )
    return {
        "vocabulary_precedence_rule": precedence.format(
            vocabularies=vocabularies,
            reason=_PRECEDENCE_REASON,
            vacuous=_NO_VACUOUS_TYPE,
        ),
        "connect_rule": _CONNECT.format(fallback=f"with {fallback}"),
        "typing_rule": _TYPING.format(fallback=f"from {fallback}"),
    }


def format_facts_operational_guidelines(
    *,
    facts_namespace: str,
    domain_ontologies_clause: str,
    jsonld: bool,
    quantity_chapter: str = "",
    statement_rules: bool = False,
    citation_vocabulary: Mapping[str, str] | None = None,
    search_guidelines: str = "",
    vocabulary: str = "strict",
    additional_namespaces: tuple[str, ...] | list[str] = (),
) -> str:
    """Build operational guidelines for the active graph format.

    ``vocabulary`` (``FACTS_VOCABULARY``) picks the vocabulary rules: the
    provided ontology, then the well-known vocabularies and
    ``additional_namespaces``, and in open mode any other established
    vocabulary (:func:`format_vocabulary_rules`).

    Args:
        facts_namespace: The ``cd:`` namespace.
        domain_ontologies_clause: The domain ontologies of rule 1a.
        jsonld: Whether the wire format is JSON-LD.
        quantity_chapter: :func:`format_quantity_chapter` for a unit whose
            text states a number; empty leaves every quantity rule out.
        statement_rules: Whether the statement rules outside the chapter
            apply (``UnitPromptScope.statements``): the statement subject
            (rule 3c), the sensor link of rule 1e, the statement checklist
            line and the measurement vocabularies of rule 2. A quantity
            chapter implies them.
        citation_vocabulary: ``CHUNK_CITATION_VOCABULARY``; its
            ``attributed_to`` term gives the attribution rule. Empty for a
            unit the citation lane governs.
        search_guidelines: The web-grounding guideline, or empty.
        vocabulary: ``strict`` or ``open``.
        additional_namespaces: The deployment's own shared vocabularies.
    """
    statements = statement_rules or bool(quantity_chapter)
    literal_rules = (
        facts_literal_rules_jsonld
        if jsonld
        else facts_literal_rules_turtle.format(
            currency_rule=_CURRENCY_TURTLE if quantity_chapter else ""
        )
    )
    hygiene = facts_output_hygiene_jsonld if jsonld else facts_output_hygiene_turtle
    attribution = format_attribution_rule(citation_vocabulary)
    guidelines = facts_instruction_shared.format(
        **format_vocabulary_rules(
            vocabulary, additional_namespaces, statement_rules=statements
        ),
        domain_ontologies_clause=domain_ontologies_clause,
        facts_namespace=facts_namespace,
        literal_encoding_rules=literal_rules,
        output_hygiene_rule=hygiene,
        sensor_clause=_SENSOR_CLAUSE if statements else _NO_SENSOR_CLAUSE,
        link_kinds=(
            "provenance or observation property"
            if statements
            else "provenance property"
        ),
        quantity_chapter=quantity_chapter,
        statement_rule=_SUBJECT + "\n" if statements else "",
        quantity_checklist=_QUANTITY_CHECKLIST if quantity_chapter else "",
        statement_checklist=_STATEMENT_CHECKLIST if statements else "",
        attribution_rule=attribution,
        cited_work_clause=_CITED_WORK if attribution else "",
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
