"""Domain-agnostic numeric-mention inventory for coverage checking.

Compares numbers stated in a source text against numeric literals present
in the extracted graph. The comparison is deliberately verbatim-oriented:
extraction is expected to transcribe source values exactly (units are
normalized downstream in code, never by the LLM), so a text number missing
from the graph is a candidate extraction gap.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from weakref import WeakKeyDictionary

from rdflib import RDF, RDFS, SKOS, Graph, Literal, URIRef
from rdflib.namespace import DCTERMS

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.util.measurement_lexicon import Mention, unit_adjacent_numbers

logger = logging.getLogger(__name__)

# Numbers incl. decimals; exponents like 1.4e3; range/uncertainty separators
# are handled by extracting each side separately.
_NUMBER_PATTERN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![\w])")

# Bare 4-digit integers in this span are treated as publication years and left
# out of coverage checking, because citation/date noise otherwise dominates the
# advisory findings. The cost is real and accepted: a genuine bare quantity in
# this range (a 2000 K temperature, 1950 rpm) is exempted too. Only bare
# integers are affected -- anything with a decimal point, exponent, or unit
# attachment still counts. Disable per call with ``ignore_year_like=False``.
_YEAR_RANGE = (1900, 2100)


def canonical_number(text: str) -> str | None:
    """Return the canonical decimal form of a numeric string, or None."""
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return format(value.normalize(), "f")


#: Characters that join a digit group to the rest of an identifier rather than
#: to a magnitude. Deliberately narrow. A decimal point is excluded because the
#: number pattern already consumes "3.14" as one match, and a hyphen is
#: excluded because "10-15 meV" is a range whose sides are both real values --
#: the module reads each side separately by design.
_IDENTIFIER_ADJACENT = frozenset("/:")


def _is_identifier_fragment(text: str, start: int, end: int) -> bool:
    """Whether the digit group at ``text[start:end]`` belongs to an identifier.

    The signal is adjacency to an identifier separator: "600/92" is one file
    number, "10.1234/example" one DOI. A magnitude with its unit ("8.5 nm") is
    untouched, and so is a range.

    Args:
        text: The text the match came from.
        start: Match start offset.
        end: Match end offset.

    Returns:
        True when the digit group sits against an identifier separator.
    """
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return before in _IDENTIFIER_ADJACENT or after in _IDENTIFIER_ADJACENT


def extract_numeric_tokens(
    text: str,
    *,
    ignore_year_like: bool = True,
    ignore_identifier_fragments: bool = False,
) -> set[str]:
    """Extract canonical numeric tokens from free text.

    Args:
        text: Source text.
        ignore_year_like: Drop bare integers in the 1900-2100 range (years,
            citation artifacts). Values that also occur with a decimal point
            are kept.
        ignore_identifier_fragments: Drop digit groups sitting against an
            identifier separator -- see :func:`_is_identifier_fragment`. A
            digit group standing alone as its own token is *not* covered:
            nothing around it distinguishes a file-number component from a
            small quantity, and guessing there would cost real values.

    Returns:
        Set of canonical decimal strings.
    """
    tokens: set[str] = set()
    for match in _NUMBER_PATTERN.finditer(text):
        raw = match.group(1)
        if ignore_identifier_fragments and _is_identifier_fragment(
            text, match.start(1), match.end(1)
        ):
            continue
        canonical = canonical_number(raw)
        if canonical is None:
            continue
        if (
            ignore_year_like
            and "." not in raw
            and "e" not in raw.lower()
            and _YEAR_RANGE[0] <= int(canonical.split(".")[0] or 0) <= _YEAR_RANGE[1]
            and canonical.isdigit()
        ):
            continue
        tokens.add(canonical)
    return tokens


#: Predicates whose literals describe a node rather than assert a value. A
#: number inside one of these is prose, not data.
_ANNOTATION_PREDICATES = frozenset({RDFS.label, RDFS.comment, DCTERMS.description})
_SKOS_NAMESPACE = str(SKOS)


def _is_annotation(predicate: object) -> bool:
    return predicate in _ANNOTATION_PREDICATES or str(predicate).startswith(
        _SKOS_NAMESPACE
    )


def numeric_literals_in_graph(
    graph: RDFGraph, *, include_annotations: bool = False
) -> set[str]:
    """Collect canonical numeric values appearing in graph literals.

    By default numbers inside labels, comments, SKOS notes and descriptions
    do **not** count as present. Counting them let a placeholder node
    labelled with the missing number silence the coverage finding that asked
    for it, so the lane measured whether a number had been *mentioned*
    rather than whether it had been extracted; a value that exists only
    inside a label is invisible to every query and to SHACL alike.

    Args:
        graph: The graph to inventory.
        include_annotations: Count numbers inside annotation literals too.
    """
    values: set[str] = set()
    for _, predicate, obj in graph:
        if not isinstance(obj, Literal):
            continue
        if not include_annotations and _is_annotation(predicate):
            continue
        text = str(obj)
        canonical = canonical_number(text.strip())
        if canonical is not None:
            values.add(canonical)
            continue
        for match in _NUMBER_PATTERN.finditer(text):
            canonical = canonical_number(match.group(1))
            if canonical is not None:
                values.add(canonical)
    return values


@dataclass(frozen=True)
class NumericInventory:
    """Numbers stated in a text, split by whether a unit stands next to them.

    ``measurements`` are unit-adjacent mentions in text order, one per
    distinct value, each carrying the unit token and the phrase it occurs in;
    a stated measurement is a fact the graph is expected to hold, and the
    context is what lets a later pass place it. ``unclassified`` are the bare
    numbers, shortest-first: a value whose unit sits elsewhere in the
    sentence, or typography, and nothing in the text alone says which.
    """

    measurements: list[Mention] = field(default_factory=list)
    unclassified: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.measurements and not self.unclassified

    def measurement_values(self) -> list[str]:
        """Canonical values of the measurements, in text order."""
        values = [canonical_number(m.value) for m in self.measurements]
        return [value for value in values if value is not None]


def inventory_numeric_mentions(
    text: str,
    *,
    unit_surfaces: Collection[str] = frozenset(),
    ignore_year_like: bool = True,
    ignore_identifier_fragments: bool = False,
) -> NumericInventory:
    """Split the numbers of ``text`` into measurements and bare numbers.

    A number is a measurement when a unit surface stands next to it -- from
    the built-in lexicon or from ``unit_surfaces``, typically the labels and
    symbols of the unit individuals in the unit's ontology context. The
    year-like and identifier guards apply to the bare numbers only: a number
    written with its unit is a measurement whatever its magnitude.

    Args:
        text: Source text of the unit.
        unit_surfaces: Extra unit surfaces beyond the built-in lexicon.
        ignore_year_like: Drop bare integers in the publication-year span.
        ignore_identifier_fragments: Drop bare digit groups that are parts of
            an identifier.

    Returns:
        The inventory; measurements in text order, bare numbers shortest-first.
    """
    measurements: list[Mention] = []
    seen: set[str] = set()
    for mention in unit_adjacent_numbers(text, unit_surfaces):
        canonical = canonical_number(mention.value)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        measurements.append(mention)
    bare = (
        extract_numeric_tokens(
            text,
            ignore_year_like=ignore_year_like,
            ignore_identifier_fragments=ignore_identifier_fragments,
        )
        - seen
    )
    return NumericInventory(
        measurements=measurements,
        unclassified=sorted(bare, key=lambda value: (len(value), value)),
    )


def missing_numeric_inventory(
    text: str,
    graph: RDFGraph,
    *,
    unit_surfaces: Collection[str] = frozenset(),
    ignore_year_like: bool = True,
    ignore_identifier_fragments: bool = False,
    limit: int = 30,
) -> NumericInventory:
    """The inventory of ``text`` restricted to values absent from the graph.

    Capped at ``limit`` over both lists, measurements first: they are the
    numbers a later pass can act on, so when the cap bites it is the bare
    numbers that are dropped. A warning records how many were.

    Args:
        text: Source text for the unit.
        graph: Graph extracted from that text.
        unit_surfaces: Extra unit surfaces beyond the built-in lexicon.
        ignore_year_like: Drop bare integers in the publication-year span.
        ignore_identifier_fragments: Drop bare digit groups that are parts of
            an identifier. Offering them invites the critic to structure a
            file number or a citation into numeric properties, which the
            downstream multi-value check then flags.
        limit: Maximum mentions across both lists.

    Returns:
        The missing measurements in text order and the missing bare numbers
        shortest-first.
    """
    present = numeric_literals_in_graph(graph)
    inventory = inventory_numeric_mentions(
        text,
        unit_surfaces=unit_surfaces,
        ignore_year_like=ignore_year_like,
        ignore_identifier_fragments=ignore_identifier_fragments,
    )
    measurements = [
        mention
        for mention in inventory.measurements
        if canonical_number(mention.value) not in present
    ]
    unclassified = [value for value in inventory.unclassified if value not in present]
    total = len(measurements) + len(unclassified)
    if total > limit:
        logger.warning(
            "Numeric coverage: %d missing mention(s) truncated to %d for the prompt",
            total,
            limit,
        )
        measurements = measurements[:limit]
        unclassified = unclassified[: max(0, limit - len(measurements))]
    return NumericInventory(measurements=measurements, unclassified=unclassified)


def missing_numeric_mentions(
    text: str,
    graph: RDFGraph,
    *,
    ignore_year_like: bool = True,
    ignore_identifier_fragments: bool = False,
    limit: int = 30,
    unit_surfaces: Collection[str] = frozenset(),
) -> list[str]:
    """Return canonical numbers stated in text but absent from the graph.

    Measurements come first in text order, then bare numbers shortest-first;
    see :func:`missing_numeric_inventory` for the split and the cap.

    Args:
        text: Source text for the unit.
        graph: Graph extracted from that text.
        ignore_year_like: Drop bare integers in the publication-year span.
        ignore_identifier_fragments: Drop bare digit groups that are parts of
            an identifier.
        limit: Maximum mentions returned.
        unit_surfaces: Extra unit surfaces beyond the built-in lexicon.

    Returns:
        Canonical decimal strings, capped at ``limit``.
    """
    inventory = missing_numeric_inventory(
        text,
        graph,
        unit_surfaces=unit_surfaces,
        ignore_year_like=ignore_year_like,
        ignore_identifier_fragments=ignore_identifier_fragments,
        limit=limit,
    )
    return inventory.measurement_values() + inventory.unclassified


#: Surface-bearing predicates a unit individual is named by. Names every RDF
#: vocabulary shares, plus the code/symbol predicates recognised by local
#: name so no catalog vocabulary is compiled in.
_NAME_PREDICATES = (RDFS.label, SKOS.prefLabel, SKOS.altLabel, SKOS.notation)
_CODE_LOCAL_NAMES = ("symbol", "code", "notation", "abbreviation")
#: Longest surface worth matching against a number-adjacent token.
_MAX_SURFACE_CHARS = 12

_SurfaceIndex = dict[str, tuple[str, ...]]
_surface_memo: WeakKeyDictionary[Graph, tuple[int, frozenset[str], _SurfaceIndex]] = (
    WeakKeyDictionary()
)


def _local(iri: str) -> str:
    for separator in ("#", "/"):
        head, sep, local = iri.rpartition(separator)
        if sep and local:
            return local
    return iri


def _unit_classes(graph: Graph, unit_properties: Collection[str]) -> set[URIRef]:
    """Classes whose individuals are units: unit-property ranges, ``*Unit``
    classes, and everything below them."""
    seeds: set[URIRef] = set()
    for prop in unit_properties:
        for rng in graph.objects(URIRef(prop), RDFS.range):
            if isinstance(rng, URIRef):
                seeds.add(rng)
    for cls in set(graph.objects(None, RDF.type)) | set(
        graph.subjects(RDFS.subClassOf)
    ):
        if isinstance(cls, URIRef) and _local(str(cls)).endswith("Unit"):
            seeds.add(cls)
    closure = set(seeds)
    frontier = list(seeds)
    while frontier:
        current = frontier.pop()
        for sub in graph.subjects(RDFS.subClassOf, current):
            if isinstance(sub, URIRef) and sub not in closure:
                closure.add(sub)
                frontier.append(sub)
    return closure


def _build_surface_index(
    graph: Graph, unit_properties: Collection[str]
) -> _SurfaceIndex:
    classes = _unit_classes(graph, unit_properties)
    if not classes:
        return {}
    individuals = {
        subject
        for cls in classes
        for subject in graph.subjects(RDF.type, cls)
        if isinstance(subject, URIRef)
    }
    index: dict[str, set[str]] = {}
    for individual in individuals:
        for predicate, value in graph.predicate_objects(individual):
            if not isinstance(value, Literal):
                continue
            local = _local(str(predicate)).lower()
            if predicate not in _NAME_PREDICATES and not any(
                token in local for token in _CODE_LOCAL_NAMES
            ):
                continue
            text = str(value).strip()
            if (
                not text
                or len(text) > _MAX_SURFACE_CHARS
                or any(ch.isspace() for ch in text)
                or canonical_number(text) is not None
            ):
                continue
            index.setdefault(text, set()).add(str(individual))
    return {surface: tuple(sorted(iris)) for surface, iris in index.items()}


def unit_surface_index(
    ontology_graph: Graph | None, unit_properties: Collection[str] = ()
) -> _SurfaceIndex:
    """Surface form -> unit individuals declaring it, for one ontology graph.

    Unit individuals are found through the ranges of the configured unit-role
    properties and through classes named ``*Unit``, with their subclasses.
    Surfaces are labels, notations and code/symbol literals short enough to
    stand next to a number. Memoised per graph object (validated by size, so
    a graph mutated in place is re-indexed), because the snapshot is shared
    by reference across a whole fan-out and this walks it whole.

    Args:
        ontology_graph: The unit's ontology context; ``None`` yields nothing.
        unit_properties: IRIs of the unit-role properties (``qudt:unit``).

    Returns:
        Surface -> sorted unit IRIs. Treat as read-only.
    """
    if ontology_graph is None or len(ontology_graph) == 0:
        return {}
    key = frozenset(unit_properties)
    try:
        cached = _surface_memo.get(ontology_graph)
    except TypeError:
        cached = None
    if cached is not None and cached[0] == len(ontology_graph) and cached[1] == key:
        return cached[2]
    index = _build_surface_index(ontology_graph, key)
    try:
        _surface_memo[ontology_graph] = (len(ontology_graph), key, index)
    except TypeError:
        pass
    return index


def unit_surfaces_in_ontology(
    ontology_graph: Graph | None, unit_properties: Collection[str] = ()
) -> frozenset[str]:
    """The unit surfaces of an ontology graph; see :func:`unit_surface_index`."""
    return frozenset(unit_surface_index(ontology_graph, unit_properties))
