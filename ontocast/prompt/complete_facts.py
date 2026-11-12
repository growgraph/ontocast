"""Prompt for the insert-only facts gap pass.

Runs after the facts render/critic loop, only when the unit has a gap: the
numeric-coverage inventory (:mod:`ontocast.util.numeric_inventory`) still
lists measurements absent from the unit graph, or a statement or value is
linked to nothing (``statement_without_subject`` findings). The render and critic prompts read the whole
ontology chapter; this pass instead gets a compact TERM SHEET -- the
quantity/observation/condition-shaped classes and the unit individuals of the
unit's ontology snapshot, the configured value terms and the subject
properties -- plus the unit's existing catalog-typed subjects,
so a recovered measurement can attach to what is already there instead of
minting a duplicate. Chapter order matches the render/critic templates:
preamble -> conformance -> term sheet (in place of the ontology chapter) ->
task -> guidelines -> user instruction -> text -> missing measurements ->
subject gaps.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence

from rdflib import OWL, RDF, RDFS, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.model import FactsUnitFinding
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import well_known_statement_properties
from ontocast.tool.facts_validation.terms import SubjectContract
from ontocast.util.numeric_inventory import NumericInventory, unit_surface_index

from .common import system_preamble_semantic

# Shares the render/critic preamble so a provider's prefix cache has a chance
# to serve this pass too, though the chapters after it are unrelated.
preamble = system_preamble_semantic

template_prompt = """
{preamble}

{conformance_chapter}

{term_sheet}

{catalog_subjects_chapter}

# TASK

The facts graph already extracted from this unit has gaps: measurements the
source text states that are missing, or statements that name nothing they are
about. Propose insert-only fixes that close them.

{completion_instruction}

{user_instruction}

{text_chapter}

{missing_measurements_chapter}

{subject_gaps_chapter}

{output_instruction}

{format_instructions}
"""

completion_instruction = f"""\n\n
# COMPLETION GUIDELINES

This is an insert-only pass, not a re-extraction and not a critique. Every
fix's `action` MUST be `ADD`; never REMOVE or REPLACE, and never fill
`triple_ids` -- nothing here is addressed by id. Leave every existing triple
exactly as it is; propose nothing beyond what recovers a listed measurement.

1. Facts use the fixed namespace `{DEFAULT_IRI}` with the prefix `cd:`. Type
   every new subject via `rdf:type` with a class from the TERM SHEET
   when one fits; reach for a generic vocabulary only when it does not.
   Never mint a new subject under a domain ontology prefix.

2. A recovered measurement is a statement about something. Write it as a
   value on a property of a subject from the EXISTING SUBJECTS chapter, or
   as a statement (an observation class of the TERM SHEET) that names its
   subject with a subject property the TERM SHEET lists and holds the value
   as a blank node. Attach to an existing subject's IRI instead of minting
   a new one for the same entity. A value node nothing points to is not a
   fact.

3. Every new `cd:` subject carries `rdfs:label` naming it from the source
   text, in the same language.

4. A property whose range is a class (a unit-valued property among them)
   takes an IRI, never a string: resolve the unit token against the TERM
   SHEET's unit individuals by label, notation, or symbol, matching
   case-sensitively. When no listed unit matches, or the sheet lists none,
   keep the unit token as text on the value node (`rdfs:comment`) and add
   no unit link -- never a unit IRI the TERM SHEET does not list. Keep the
   value and unit verbatim -- never convert, round, or re-derive.

5. Do NOT mint an entity for a bare number, a citation marker, or a
   typography artifact. If a listed measurement cannot be placed as a fact
   -- the text gives it no subject -- leave it out rather than guessing a
   subject for it.

6. Declare every namespace prefix your fixes use that is not already bound
   in the chapters above.

7. For each statement under SUBJECT GAPS, add the one triple linking it to
   what the text says it is about, using a subject property the TERM SHEET
   lists, in the direction it runs. The subject is the instance the sentence measures or
   describes; when the sentence does not repeat it, it is the instance the
   surrounding text is about. Prefer a listed candidate; mint a new `cd:`
   subject only for a thing the text names. Leave a statement whose subject
   the text does not give.
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


def output_instruction_for(fmt: LLMGraphFormat) -> str:
    """The GRAPH FORMAT INSTRUCTION for ``correct_value``, by wire format."""
    if fmt == LLMGraphFormat.JSONLD:
        return _OUTPUT_INSTRUCTION_JSONLD
    return _OUTPUT_INSTRUCTION_TURTLE


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
    contract: SubjectContract | None = None,
    max_classes: int = 40,
    max_units: int = 60,
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
        contract: The catalog's subject properties, each direction.
        max_classes: Cap on listed classes.
        max_units: Cap on listed unit individuals.

    Returns:
        The chapter text, or "" when there is nothing to list.
    """
    classes = _quantity_like_classes(ontology_graph, limit=max_classes)
    units = _unit_individuals(ontology_graph, unit_properties, limit=max_units)
    forward = sorted(contract.forward) if contract is not None else []
    inverse = sorted(contract.inverse) if contract is not None else []
    if not (classes or units or value_terms or forward or inverse):
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
    if contract is not None and contract.statement_classes & {
        iri for _, _, iri in classes
    }:
        lines.append("\nProperties a statement takes (well-known):")
        for iri, label in sorted(well_known_statement_properties().items()):
            lines.append(f'  {_qname(ontology_graph, URIRef(iri))}  "{label}"')
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
        "Already in the graph -- attach a recovered measurement to one of "
        "these when it is about the same entity, instead of minting a new "
        "subject:",
    ]
    for subject, type_qname, label in entries:
        row = f"  {subject}  a {type_qname}"
        if label:
            row += f'  rdfs:label "{label}"'
        lines.append(row)
    return "\n".join(lines) + "\n"


def build_subject_gaps_chapter(
    fact_graph: RDFGraph,
    gaps: Sequence[FactsUnitFinding],
    *,
    limit: int = 30,
) -> str:
    """The SUBJECT GAPS chapter: statements linked to nothing, and their options.

    Lists each statement (type and label) and the candidate subjects the
    findings suggest. The properties linking a statement to its subject are
    on the term sheet (:func:`build_term_sheet`).

    Args:
        fact_graph: The unit's current facts graph.
        gaps: ``statement_without_subject`` findings.
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

    lines = [
        "\n\n# SUBJECT GAPS",
        "Statements that state a value but name nothing they are about:",
    ]
    candidates: list[str] = []
    for finding in list(gaps)[:limit]:
        lines.append(describe(URIRef(finding.subject)))
        candidates.extend(c for c in finding.suggestions if c not in candidates)
    if candidates:
        lines.append("\nCandidate subjects in the graph:")
        lines.extend(describe(URIRef(c)) for c in candidates[:limit])
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
