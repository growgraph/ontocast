"""Prompt for the insert-only facts gap pass.

Runs after the facts render/critic loop, only when the unit has a gap, as its
:class:`~ontocast.tool.facts_validation.mandate.GapMandate` lists them:
statements missing a role their class requires (``missing_role`` findings),
instances linked to nothing, and, when measurement recovery is on,
measurements the numeric-coverage inventory still lists as absent. The TASK
chapter and the guidelines are rendered from the mandate, so they ask only for
what it admits. The render and critic prompts read the whole
ontology chapter; this pass instead gets a compact TERM SHEET -- the
quantity/observation/condition-shaped classes and the unit individuals of the
unit's ontology snapshot, the configured value terms and the subject
properties -- plus the unit's existing catalog-typed subjects,
so a recovered measurement can attach to what is already there instead of
minting a duplicate. What the prompt asks for comes from the mandate that
admits the fixes: the unit individuals, value terms, measurement and
measurement-unit guidelines and a value in the output example only when it
lists measurements; the exact-quantity guideline and the observable-property
class also when a statement lacks its property; the statement properties,
subject links and role guideline only to a call that repairs or recovers a
statement. The quantity classes follow the render's quantity chapter. Chapter order matches the render/critic templates:
preamble -> conformance -> term sheet (in place of the ontology chapter) ->
task -> guidelines -> user instruction -> text -> missing measurements ->
statement gaps -> detached instances.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence

from rdflib import OWL, RDF, RDFS, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.model import FactsUnitFinding
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import well_known_statement_roles
from ontocast.tool.facts_validation.mandate import GapMandate
from ontocast.tool.facts_validation.statements import StatementContract
from ontocast.util.numeric_inventory import NumericInventory, unit_surface_index

from .common import system_preamble_semantic
from .facts_guidelines import observable_property_classes

# Shares the render/critic preamble so a provider's prefix cache has a chance
# to serve this pass too, though the chapters after it are unrelated.
preamble = system_preamble_semantic

template_prompt = """
{preamble}

{conformance_chapter}

{term_sheet}

{catalog_subjects_chapter}

{task_chapter}

{completion_instruction}

{user_instruction}

{text_chapter}

{missing_measurements_chapter}

{role_gaps_chapter}

{detached_chapter}

{output_instruction}

{format_instructions}
"""

_GAP_KINDS = (
    ("measurements", "measurements the source text states that are missing"),
    ("roles", "statements that lack a role their class requires"),
    ("detached", "instances linked to nothing"),
)


def _gap_kinds(mandate: GapMandate) -> list[str]:
    present = {
        "measurements": bool(mandate.inventory.measurements),
        "roles": bool(mandate.role_gaps),
        "detached": bool(mandate.detached),
    }
    return [text for key, text in _GAP_KINDS if present[key]]


def build_task_chapter(mandate: GapMandate) -> str:
    """The TASK chapter: the gap kinds this call is for, and only those."""
    kinds = _gap_kinds(mandate)
    listed = (
        kinds[0] if len(kinds) == 1 else ", ".join(kinds[:-1]) + f" and {kinds[-1]}"
    )
    return (
        "# TASK\n\n"
        f"The facts graph already extracted from this unit has gaps: {listed}, "
        "listed below. Propose insert-only fixes that close them; a fix adding "
        "anything else is refused."
    )


_GUIDELINE_NAMESPACE = f"""Facts use the fixed namespace `{DEFAULT_IRI}` with the prefix `cd:`. Type
   every new subject via `rdf:type` with a class from the TERM SHEET
   when one fits; reach for a generic vocabulary only when it does not.
   Never mint a new subject under a domain ontology prefix."""

_GUIDELINE_MEASUREMENT = """A recovered measurement is a statement about something. Write it as a
   value on a property of a subject from the EXISTING SUBJECTS chapter, or
   as a statement (an observation class of the TERM SHEET) that names its
   subject with a subject property the TERM SHEET lists and holds the value
   as a blank node. Attach to an existing subject's IRI instead of minting
   a new one for the same entity. A value node nothing points to is not a
   fact."""

_GUIDELINE_LABEL = """Every new `cd:` subject carries `rdfs:label` naming it from the source
   text, in the same language."""

_GUIDELINE_UNIT = """A property whose range is a class (a unit-valued property among them)
   takes an IRI, never a string: resolve the unit token against the TERM
   SHEET's unit individuals by label, notation, or symbol, matching
   case-sensitively. When no listed unit matches, use the QUDT unit
   (`unit:`) for exactly that unit, only if you are certain of its QUDT name.
   When no measurement unit fits, add no unit link: the value's label keeps
   the verbatim text, and an invented or approximate unit IRI is never
   right. Keep the value and unit verbatim -- never convert, round, or
   re-derive."""

_GUIDELINE_EXACT_QUANTITY = """The observed property of a statement is an individual. Reuse a catalog
   individual only when it names exactly the quantity stated, its kind and
   what it is relative to. A change, difference or ratio between two states
   is a quantity of its own. Otherwise mint a `cd:` instance of the
   observable-property class the TERM SHEET names."""

_GUIDELINE_ARTIFACT = """Do NOT mint an entity for a bare number, a citation marker, or a
   typography artifact."""

_GUIDELINE_UNPLACED = """If a listed measurement cannot be placed as a fact -- the text gives it
   no subject -- leave it out rather than guessing a subject for it."""

_GUIDELINE_PREFIXES = """Declare every namespace prefix your fixes use that is not already bound
   in the chapters above."""

_GUIDELINE_ROLES = """For each statement under STATEMENT GAPS, add the triple giving it the
   role it lacks, in the direction its property runs: its subject with a
   subject property the TERM SHEET lists, its property with the property a
   statement takes, or the link named. The subject is the instance the
   sentence measures or describes; when the sentence does not repeat it, it
   is the instance the surrounding text is about. Prefer a listed candidate;
   mint a new `cd:` instance only for a thing the text names. Leave a gap
   the text does not fill."""

_GUIDELINE_DETACHED = """For each instance under DETACHED INSTANCES, add the one triple linking it
   to another instance of this unit that the text relates it to, with one of
   its candidate properties in the direction given (`from`: the detached
   instance is the subject; `to`: it is the object), or another object
   property whose domain and range fit both ends. Link to an existing
   instance; never mint one for this. Leave an instance the text relates to
   nothing."""


def completion_guidelines(mandate: GapMandate) -> str:
    """The COMPLETION GUIDELINES chapter for ``mandate``.

    Each guideline is given only for a gap the mandate lists, so the chapter
    asks for nothing :meth:`GapMandate.admits` refuses: the measurement and
    measurement-unit guidelines when it lists measurements, the
    exact-quantity guideline then or when a statement lacks its property, the
    role and detached-instance guidelines when it lists those gaps.
    """
    measurements = bool(mandate.inventory.measurements)
    guidelines = [
        _GUIDELINE_NAMESPACE,
        *([_GUIDELINE_MEASUREMENT] if measurements else []),
        _GUIDELINE_LABEL,
        *([_GUIDELINE_UNIT] if measurements else []),
        *([_GUIDELINE_EXACT_QUANTITY] if asks_property(mandate) else []),
        _GUIDELINE_ARTIFACT,
        *([_GUIDELINE_UNPLACED] if measurements else []),
        _GUIDELINE_PREFIXES,
        *([_GUIDELINE_ROLES] if mandate.role_gaps else []),
        *([_GUIDELINE_DETACHED] if mandate.detached else []),
    ]
    numbered = "\n\n".join(
        f"{index}. {text}" for index, text in enumerate(guidelines, start=1)
    )
    return (
        "\n\n\n# COMPLETION GUIDELINES\n\n"
        "This is an insert-only pass, not a re-extraction and not a critique. Every\n"
        "fix's `action` MUST be `ADD`; never REMOVE or REPLACE, and never fill\n"
        "`triple_ids` -- nothing here is addressed by id. Leave every existing triple\n"
        "exactly as it is.\n\n" + numbered + "\n"
    )


_OUTPUT_INSTRUCTION_LINK_TURTLE = """\n\n
# OUTPUT INSTRUCTION

Provide `proposed_fixes` as a JSON array of fix objects, every `action`
equal to `ADD`. `correct_value` is a **string** containing valid Turtle: any
`@prefix` declarations not already bound above, then the triples of one fix:
the link it adds, with the type and label of any node it introduces. Leave
`triple_ids` empty.
Example: "cd:item_1 onto:partOf cd:group_1 ."
"""

_OUTPUT_INSTRUCTION_LINK_JSONLD = """\n\n
# OUTPUT INSTRUCTION

Provide `proposed_fixes` as a JSON array of fix objects, every `action`
equal to `ADD`. `correct_value` is a **string** containing valid JSON for one
fix: an inline `@context` for any prefix not already bound above, then the
node the link starts from (`@id`) with the link it adds, and the type and
label of any node it introduces. Leave `triple_ids` empty.
Example: "{\\"@id\\": \\"cd:item_1\\", \\"onto:partOf\\": {\\"@id\\": \\"cd:group_1\\"}}"
"""

_OUTPUT_INSTRUCTION_TURTLE = """\n\n
# OUTPUT INSTRUCTION

Provide `proposed_fixes` as a JSON array of fix objects, every `action`
equal to `ADD`. `correct_value` is a **string** containing valid Turtle: any
`@prefix` declarations not already bound above, then the triples of one
recovered fact: a statement that names its subject and holds its value as a
blank node, or a value on a property of a subject named in EXISTING
SUBJECTS. Leave `triple_ids` empty.
Example: "cd:reading_1 a onto:Observation ; rdfs:label \\"example reading\\" ; onto:about cd:subject_1 ; onto:hasValue [ a onto:Quantity ; qudt:numericValue \\"1\\"^^xsd:decimal ; qudt:unit unit:EXAMPLE ] ."
"""

_OUTPUT_INSTRUCTION_JSONLD = """\n\n
# OUTPUT INSTRUCTION

Provide `proposed_fixes` as a JSON array of fix objects, every `action`
equal to `ADD`. `correct_value` is a **string** containing valid JSON for one
recovered fact: an inline `@context` for any prefix not already bound above,
then a statement node (`@id`/`@type`) that names its subject and holds its
value as a nested node without `@id`, or a value on a property of a subject
named in EXISTING SUBJECTS. Leave `triple_ids` empty.
Example: "{\\"@context\\": {\\"qudt\\": \\"http://qudt.org/schema/qudt/\\"}, \\"@id\\": \\"cd:reading_1\\", \\"@type\\": \\"onto:Observation\\", \\"rdfs:label\\": \\"example reading\\", \\"onto:about\\": {\\"@id\\": \\"cd:subject_1\\"}, \\"onto:hasValue\\": {\\"@type\\": \\"onto:Quantity\\", \\"qudt:numericValue\\": {\\"@value\\": \\"1\\", \\"@type\\": \\"xsd:decimal\\"}, \\"qudt:unit\\": {\\"@id\\": \\"unit:EXAMPLE\\"}}}"
"""


def asks_property(mandate: GapMandate) -> bool:
    """Whether the call may name a statement's observed property: it recovers
    measurements, or a statement lacks its property."""
    return bool(mandate.inventory.measurements) or any(
        gap.facet == "property" for gap in mandate.role_gaps
    )


def output_instruction_for(fmt: LLMGraphFormat, mandate: GapMandate) -> str:
    """The GRAPH FORMAT INSTRUCTION for ``correct_value``, by wire format.

    Its example is a recovered value when the mandate lists measurements, the
    only call that may add one; a link otherwise.
    """
    values = bool(mandate.inventory.measurements)
    if fmt == LLMGraphFormat.JSONLD:
        return _OUTPUT_INSTRUCTION_JSONLD if values else _OUTPUT_INSTRUCTION_LINK_JSONLD
    return _OUTPUT_INSTRUCTION_TURTLE if values else _OUTPUT_INSTRUCTION_LINK_TURTLE


# Generic, structure-only role hints -- the same kind of local-name pattern
# match numeric_inventory.py uses for "*Unit" classes. Never domain
# vocabulary: applied to whatever ontology graph is supplied, not embedded
# as content.
_TERM_SHEET_CLASS_HINT = re.compile(
    r"(quantity|measurement|observation|condition|value|characteristic|parameter)",
    re.IGNORECASE,
)


def _qname(graph: RDFGraph, term: URIRef) -> str:
    """Prefixed name for ``term`` under a prefix the graph binds, else ``<IRI>``.

    Never mints a prefix: one the prompt does not declare cannot be parsed back.
    """
    try:
        prefix, _, local = graph.namespace_manager.compute_qname(term, generate=False)
    except (KeyError, ValueError):
        return f"<{term}>"
    return f"{prefix}:{local}"


def _first_label(graph: RDFGraph, node: URIRef) -> str:
    """The node's least label, so a prompt does not depend on triple order."""
    return min((str(o) for o in graph.objects(node, RDFS.label)), default="")


def _quantity_like_classes(
    graph: RDFGraph, *, limit: int
) -> list[tuple[str, str, str]]:
    """(qname, label, IRI) of the classes whose name reads as quantity-shaped."""
    classes: set[URIRef] = set()
    for cls in graph.subjects(RDF.type, OWL.Class):
        if isinstance(cls, URIRef):
            classes.add(cls)
    for cls in graph.subjects(RDF.type, RDFS.Class):
        if isinstance(cls, URIRef):
            classes.add(cls)

    entries: list[tuple[str, str, str]] = []
    for cls in classes:
        label = _first_label(graph, cls)
        if not _TERM_SHEET_CLASS_HINT.search(f"{label} {cls}"):
            continue
        entries.append((_qname(graph, cls), label, str(cls)))
    entries.sort(key=lambda entry: entry[0])
    return entries[:limit]


def _unit_individuals(
    graph: RDFGraph, unit_properties: Collection[str], *, limit: int
) -> list[tuple[str, str]]:
    """(qname, surfaces) pairs for unit individuals findable via ``unit_properties``."""
    by_iri: dict[str, list[str]] = {}
    for surface, iris in unit_surface_index(graph, unit_properties).items():
        for iri in iris:
            by_iri.setdefault(iri, []).append(surface)

    entries: list[tuple[str, str]] = []
    for iri in sorted(by_iri):
        surfaces = sorted(by_iri[iri], key=len)[:3]
        entries.append((_qname(graph, URIRef(iri)), ", ".join(surfaces)))
    return entries[:limit]


def build_term_sheet(
    ontology_graph: RDFGraph,
    unit_properties: Collection[str] = (),
    *,
    value_terms: Mapping[str, str] | None = None,
    contract: StatementContract | None = None,
    max_classes: int = 40,
    max_units: int = 60,
    quantities: bool = True,
    statements: bool = True,
    values: bool = True,
    property_class: bool = False,
) -> str:
    """The TERM SHEET chapter, in place of the full ontology chapter.

    A compact vocabulary this pass can act on without reading the whole
    catalog: quantity/observation/condition-shaped classes, and the unit
    individuals reachable through ``unit_properties`` (typically the
    deployment's configured unit-role property, expanded to an IRI).

    Args:
        ontology_graph: The unit's ontology snapshot graph.
        unit_properties: IRIs of unit-role properties (e.g. ``qudt:unit``)
            whose range/``*Unit``-named classes hold the unit individuals.
        value_terms: The deployment's quantity vocabulary, role -> term as
            configured (value class, number, unit; a deployment without a
            unit role lists none).
        contract: The catalog's statement contract: its statement classes
            and the properties linking a statement to its subject.
        max_classes: Cap on listed classes.
        max_units: Cap on listed unit individuals.
        quantities: Whether the unit text states a number; without one the
            quantity classes are left out.
        statements: Whether the call repairs or recovers a statement; without
            one the statement properties and subject links are left out.
        values: Whether the call may add a value (it recovers measurements);
            without one the unit individuals and value terms are left out.
        property_class: Whether the call may name a statement's observed
            property; with one the observable-property class is named, the
            catalog's own first.

    Returns:
        The chapter text, or "" when there is nothing to list.
    """
    candidates = _quantity_like_classes(ontology_graph, limit=max_classes)
    statement_beside = (
        statements
        and contract is not None
        and contract.is_statement(iri for _, _, iri in candidates)
    )
    classes = candidates if quantities else []
    units = (
        _unit_individuals(ontology_graph, unit_properties, limit=max_units)
        if values
        else []
    )
    value_terms = value_terms if values else None
    subject = contract.links("subject") if contract is not None and statements else None
    forward = sorted(subject.forward) if subject is not None else []
    inverse = sorted(subject.inverse) if subject is not None else []
    observable = (
        observable_property_classes(ontology_graph, contract)
        if property_class and contract is not None
        else []
    )
    if not (
        classes
        or units
        or value_terms
        or forward
        or inverse
        or statement_beside
        or observable
    ):
        return ""

    lines = [
        "\n\n# TERM SHEET",
        "Compact vocabulary for this pass -- use these; never invent a class "
        "or a unit individual.",
    ]
    if classes:
        lines.append("\nQuantity/observation/condition classes:")
        for qname, label, _ in classes:
            lines.append(f'  {qname}  "{label}"' if label else f"  {qname}")
    if units:
        lines.append(
            "\nUnit individuals (object of a unit-valued property). Symbols are "
            "case-sensitive: use a unit only when a listed symbol matches the "
            "text exactly."
        )
        for qname, surfaces in units:
            lines.append(f"  {qname}  ({surfaces})" if surfaces else f"  {qname}")
    if value_terms:
        lines.append("\nValue node terms (role: term):")
        lines.extend(f"  {role}: {term}" for role, term in sorted(value_terms.items()))
    if statement_beside:
        lines.append("\nProperties a statement takes (well-known):")
        for iri, role in sorted(well_known_statement_roles().items()):
            if role.label is not None:
                lines.append(f'  {_qname(ontology_graph, URIRef(iri))}  "{role.label}"')
    if observable:
        lines.append(
            "\nObservable-property class (what a statement's observed property "
            "is an instance of): "
            + ", ".join(f"{_qname(ontology_graph, URIRef(c))}" for c in observable)
        )
    if forward:
        lines.append("\nSubject properties, from the statement to its subject:")
        lines.extend(f"  {_qname(ontology_graph, URIRef(p))}" for p in forward)
    if inverse:
        lines.append("\nSubject properties, from the subject to the statement:")
        lines.extend(f"  {_qname(ontology_graph, URIRef(p))}" for p in inverse)
    return "\n".join(lines) + "\n"


def build_catalog_subjects_chapter(
    fact_graph: RDFGraph,
    catalog_terms: Collection[str],
    *,
    limit: int = 40,
) -> str:
    """The EXISTING SUBJECTS chapter: unit subjects already typed by a catalog term.

    Lets the pass attach a recovered measurement to a subject that already
    exists rather than minting a duplicate for the same real-world entity.

    Args:
        fact_graph: The unit's current facts graph.
        catalog_terms: IRIs the unit's ontology snapshot declares or
            references (see ``collect_catalog_terms``).
        limit: Cap on listed subjects.

    Returns:
        The chapter text, or "" when no subject in ``fact_graph`` is typed
        by a catalog term.
    """
    terms = set(catalog_terms)
    if not terms:
        return ""

    types_of: dict[URIRef, set[str]] = {}
    for subject, _, obj in fact_graph.triples((None, RDF.type, None)):
        if (
            isinstance(subject, URIRef)
            and isinstance(obj, URIRef)
            and str(obj) in terms
        ):
            types_of.setdefault(subject, set()).add(_qname(fact_graph, obj))
    entries = [
        (
            _qname(fact_graph, subject),
            ", ".join(sorted(types)),
            _first_label(fact_graph, subject),
        )
        for subject, types in types_of.items()
    ]
    if not entries:
        return ""
    entries.sort(key=lambda row: row[0])
    entries = entries[:limit]

    lines = [
        "\n\n# EXISTING SUBJECTS",
        "Already in the graph -- attach a recovered fact to one of "
        "these when it is about the same entity, instead of minting a new "
        "subject:",
    ]
    for subject, type_qname, label in entries:
        row = f"  {subject}  a {type_qname}"
        if label:
            row += f'  rdfs:label "{label}"'
        lines.append(row)
    return "\n".join(lines) + "\n"


def build_role_gaps_chapter(
    fact_graph: RDFGraph,
    gaps: Sequence[FactsUnitFinding],
    *,
    limit: int = 30,
) -> str:
    """The STATEMENT GAPS chapter: statements missing a role, and their options.

    Lists each statement (type and label) with the role it lacks, and the
    candidate subjects the findings suggest. The properties playing each
    role are on the term sheet (:func:`build_term_sheet`).

    Args:
        fact_graph: The unit's current facts graph.
        gaps: ``missing_role`` findings.
        limit: Cap on listed statements and on listed candidates.

    Returns:
        The chapter text, or "" when there is no gap.
    """
    if not gaps:
        return ""

    def describe(node: URIRef) -> str:
        types = ", ".join(
            sorted(
                _qname(fact_graph, t)
                for t in fact_graph.objects(node, RDF.type)
                if isinstance(t, URIRef)
            )
        )
        label = _first_label(fact_graph, node)
        row = f"  {_qname(fact_graph, node)}"
        if types:
            row += f"  a {types}"
        if label:
            row += f'  rdfs:label "{label}"'
        return row

    def lacks(finding: FactsUnitFinding) -> str:
        if finding.facet == "link":
            return f"lacks {_qname(fact_graph, URIRef(finding.predicate))}"
        return f"lacks its {finding.facet}"

    lines = [
        "\n\n# STATEMENT GAPS",
        "Statements that state a value but lack a role their class requires:",
    ]
    candidates: list[str] = []
    for finding in list(gaps)[:limit]:
        lines.append(f"{describe(URIRef(finding.subject))}  -- {lacks(finding)}")
        candidates.extend(c for c in finding.suggestions if c not in candidates)
    if candidates:
        lines.append("\nCandidate subjects in the graph:")
        lines.extend(describe(URIRef(c)) for c in candidates[:limit])
    return "\n".join(lines) + "\n"


def build_detached_chapter(
    fact_graph: RDFGraph,
    detached: Sequence[tuple[URIRef, Sequence[tuple[URIRef, str]]]],
    *,
    limit: int = 20,
) -> str:
    """The DETACHED INSTANCES chapter: instances linked to nothing, and how they may link.

    Args:
        fact_graph: The unit's current facts graph.
        detached: Each instance with its candidate properties and their
            direction (``from``: the instance is the subject).
        limit: Cap on listed instances.

    Returns:
        The chapter text, or "" when there is none.
    """
    if not detached:
        return ""
    lines = [
        "\n\n# DETACHED INSTANCES",
        "Instances linked to no other instance:",
    ]
    for node, properties in list(detached)[:limit]:
        types = ", ".join(
            sorted(
                _qname(fact_graph, t)
                for t in fact_graph.objects(node, RDF.type)
                if isinstance(t, URIRef)
            )
        )
        row = f"  {_qname(fact_graph, node)}"
        if types:
            row += f"  a {types}"
        label = _first_label(fact_graph, node)
        if label:
            row += f'  rdfs:label "{label}"'
        lines.append(row)
        if properties:
            lines.append(
                "    candidate properties: "
                + ", ".join(
                    f"{_qname(fact_graph, p)} ({direction})"
                    for p, direction in properties
                )
            )
    return "\n".join(lines) + "\n"


def build_missing_measurements_chapter(inventory: NumericInventory) -> str:
    """The MISSING MEASUREMENTS chapter: each value, unit, and source context.

    Args:
        inventory: The unit's numeric-coverage inventory (see
            ``ontocast.tool.facts_validation.unit_findings.unit_numeric_inventory``),
            already restricted to values absent from the graph.

    Returns:
        The chapter text, or "" when nothing is missing.
    """
    if not inventory.measurements:
        return ""
    lines = [
        "\n\n# MISSING MEASUREMENTS",
        "Numbers stated in the text with a unit, absent from the graph. Add "
        "a fact for each you can place; leave out one you cannot.",
        "",
    ]
    for mention in inventory.measurements:
        lines.append(
            f'  "{mention.value} {mention.unit}"  in: "...{mention.context}..."'
        )
    return "\n".join(lines) + "\n"
