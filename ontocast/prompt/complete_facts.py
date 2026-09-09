"""Prompt for the insert-only facts completion pass.

Runs after the facts render/critic loop, only when the numeric-coverage
inventory (:mod:`ontocast.util.numeric_inventory`) still lists measurements
absent from the unit graph. The render and critic prompts read the whole
ontology chapter; this pass instead gets a compact TERM SHEET -- the
quantity/observation/condition-shaped classes and the unit individuals of the
unit's ontology snapshot -- plus the unit's existing catalog-typed subjects,
so a recovered measurement can attach to what is already there instead of
minting a duplicate. Chapter order matches the render/critic templates:
preamble -> conformance -> term sheet (in place of the ontology chapter) ->
task -> guidelines -> user instruction -> text -> missing measurements.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from rdflib import OWL, RDF, RDFS, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.rdfgraph import RDFGraph
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

The facts graph already extracted from this unit is missing measurements the
source text states. Propose insert-only fixes that add them.

{completion_instruction}

{user_instruction}

{text_chapter}

{missing_measurements_chapter}

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
   every new subject via `rdf:type` with a class from the TERM SHEET below
   when one fits; reach for a generic vocabulary only when it does not.
   Never mint a new subject under a domain ontology prefix.

2. Prefer the EXISTING SUBJECTS chapter: when a recovered measurement is a
   property of a subject already in the graph, attach the new triple(s) to
   that subject's IRI instead of minting a new one for the same entity.

3. Every new `cd:` subject carries `rdfs:label` naming it from the source
   text, in the same language.

4. A property whose range is a class (a unit-valued property among them)
   takes an IRI, never a string: resolve the unit token against the TERM
   SHEET's unit individuals by label, notation, or symbol, matching
   case-sensitively. Keep the value and unit verbatim -- never convert,
   round, or re-derive.

5. Do NOT mint an entity for a bare number, a citation marker, or a
   typography artifact. If a listed measurement cannot be placed as a fact,
   leave it out rather than guessing a subject for it.

6. Declare every namespace prefix your fixes use that is not already bound
   in the chapters above.
"""

_OUTPUT_INSTRUCTION_TURTLE = """\n\n
# OUTPUT INSTRUCTION

Provide `proposed_fixes` as a JSON array of fix objects, every `action`
equal to `ADD`. `correct_value` is a **string** containing valid Turtle: any
`@prefix` declarations not already bound above, then one or more triples
about a single new subject (or a statement attaching a value to an existing
subject named in EXISTING SUBJECTS). Leave `triple_ids` empty.
Example: "cd:melting_point_1 a onto:Measurement ; qudt:numericValue \\"96\\"^^xsd:decimal ; qudt:unit unit:MilliEV ."
"""

_OUTPUT_INSTRUCTION_JSONLD = """\n\n
# OUTPUT INSTRUCTION

Provide `proposed_fixes` as a JSON array of fix objects, every `action`
equal to `ADD`. `correct_value` is a **string** containing valid JSON for one
subject node: an inline `@context` for any prefix not already bound above,
then `@id`/`@type` plus the recovered measurement's properties (or a
statement attaching a value to an existing subject named in EXISTING
SUBJECTS). Leave `triple_ids` empty.
Example: "{\\"@context\\": {\\"qudt\\": \\"http://qudt.org/schema/qudt/\\"}, \\"@id\\": \\"cd:melting_point_1\\", \\"@type\\": \\"onto:Measurement\\", \\"qudt:numericValue\\": {\\"@value\\": \\"96\\", \\"@type\\": \\"xsd:decimal\\"}, \\"qudt:unit\\": {\\"@id\\": \\"unit:MilliEV\\"}}"
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
    """Prefixed name for ``term``, falling back to the full IRI."""
    try:
        return graph.namespace_manager.qname(term)
    except Exception:
        return str(term)


def _quantity_like_classes(graph: RDFGraph, *, limit: int) -> list[tuple[str, str]]:
    """(qname, label) pairs for classes whose name reads as quantity-shaped."""
    classes: set[URIRef] = set()
    for cls in graph.subjects(RDF.type, OWL.Class):
        if isinstance(cls, URIRef):
            classes.add(cls)
    for cls in graph.subjects(RDF.type, RDFS.Class):
        if isinstance(cls, URIRef):
            classes.add(cls)

    entries: list[tuple[str, str]] = []
    for cls in classes:
        label = next((str(o) for o in graph.objects(cls, RDFS.label)), "")
        if not _TERM_SHEET_CLASS_HINT.search(f"{label} {cls}"):
            continue
        entries.append((_qname(graph, cls), label))
    entries.sort(key=lambda pair: pair[0])
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
        max_classes: Cap on listed classes.
        max_units: Cap on listed unit individuals.

    Returns:
        The chapter text, or "" when the snapshot offers neither.
    """
    classes = _quantity_like_classes(ontology_graph, limit=max_classes)
    units = _unit_individuals(ontology_graph, unit_properties, limit=max_units)
    if not classes and not units:
        return ""

    lines = [
        "\n\n# TERM SHEET",
        "Compact vocabulary for this pass -- use these; never invent a class "
        "or a unit individual.",
    ]
    if classes:
        lines.append("\nQuantity/observation/condition classes:")
        for qname, label in classes:
            lines.append(f'  {qname}  "{label}"' if label else f"  {qname}")
    if units:
        lines.append("\nUnit individuals (object of a unit-valued property):")
        for qname, surfaces in units:
            lines.append(f"  {qname}  ({surfaces})" if surfaces else f"  {qname}")
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

    entries: list[tuple[str, str, str]] = []
    seen: set[URIRef] = set()
    for subject, _, obj in fact_graph.triples((None, RDF.type, None)):
        if not isinstance(subject, URIRef) or not isinstance(obj, URIRef):
            continue
        if subject in seen or str(obj) not in terms:
            continue
        seen.add(subject)
        label = next((str(o) for o in fact_graph.objects(subject, RDFS.label)), "")
        entries.append((_qname(fact_graph, subject), _qname(fact_graph, obj), label))
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
