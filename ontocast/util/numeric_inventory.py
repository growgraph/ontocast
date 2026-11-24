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
import weakref
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from rdflib import RDF, RDFS, SKOS, Graph, Literal, URIRef

from ontocast.onto.annotation import is_annotation
from ontocast.onto.property_sides import side_classes
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.util.measurement_lexicon import (
    Mention,
    _lookup_keys,
    unit_adjacent_numbers,
)

logger = logging.getLogger(__name__)

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

    Reads the numbers :func:`stated_number_spans` reads, the one reader of
    digits in text, less a number glued to a letter after it or to a point
    before it: with a unit ("5kg") it is a measurement, read with its unit by
    :func:`inventory_numeric_mentions`; otherwise it is part of a name ("3D",
    "v1.2").

    Args:
        text: Source text.
        ignore_year_like: Drop bare four-digit integers in the 1900-2100 range
            (years, citation artifacts).
        ignore_identifier_fragments: Drop digit groups sitting against an
            identifier separator -- see :func:`_is_identifier_fragment`. A
            digit group standing alone as its own token is *not* covered:
            nothing around it distinguishes a file-number component from a
            small quantity, and guessing there would cost real values.

    Returns:
        Set of canonical decimal strings.
    """
    tokens: set[str] = set()
    for value, start, end in stated_number_spans(text):
        # Glued to a letter after it, or to a point before it ("v1.2", "No.5").
        if (end < len(text) and text[end].isalpha()) or text[start - 1 : start] == ".":
            continue
        if ignore_identifier_fragments and _is_identifier_fragment(text, start, end):
            continue
        raw = text[start:end]
        if (
            ignore_year_like
            and raw.isdigit()
            and len(raw) == 4
            and _YEAR_RANGE[0] <= int(raw) <= _YEAR_RANGE[1]
        ):
            continue
        tokens.add(value)
    return tokens


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
        if not include_annotations and is_annotation(predicate):
            continue
        text = str(obj)
        canonical = canonical_number(text.strip())
        if canonical is not None:
            values.add(canonical)
            continue
        values |= stated_numbers(text)
    return values


@dataclass(frozen=True)
class NumericInventory:
    """Numbers stated in a text, split by whether a unit stands next to them.

    ``measurements`` are unit-adjacent mentions in text order, one per
    distinct ``(value, unit)``, each carrying the unit token and the phrase
    it occurs in; a stated measurement is a fact the graph is expected to
    hold, and the context is what lets a later pass place it.
    ``unclassified`` are the bare numbers, shortest-first: a value whose
    unit sits elsewhere in the sentence, or typography, and nothing in the
    text alone says which.
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
    seen: set[tuple[str, str]] = set()
    measurement_numbers: set[str] = set()
    # One reader: a measurement is a number :func:`stated_number_spans` reads,
    # so every number listed here is one the text is judged to state.
    stated = {(value, start) for value, start, _ in stated_number_spans(text)}
    for mention in unit_adjacent_numbers(text, unit_surfaces):
        canonical = canonical_number(mention.value)
        if canonical is None or (canonical, mention.start) not in stated:
            continue
        key = (canonical, mention.unit)
        if key in seen:
            continue
        seen.add(key)
        measurement_numbers.add(canonical)
        measurements.append(mention)
    bare = (
        extract_numeric_tokens(
            text,
            ignore_year_like=ignore_year_like,
            ignore_identifier_fragments=ignore_identifier_fragments,
        )
        - measurement_numbers
    )
    return NumericInventory(
        measurements=measurements,
        unclassified=sorted(bare, key=lambda value: (len(value), value)),
    )


def states_quantity(
    text: str,
    *,
    unit_surfaces: Collection[str] = frozenset(),
    ignore_identifier_fragments: bool = False,
) -> bool:
    """Whether ``text`` states a quantity: its inventory holds a number.

    A number with a unit or a bare one counts (:func:`inventory_numeric_mentions`);
    a year-like bare integer and, with ``ignore_identifier_fragments``, a
    digit group of an identifier do not. Number words are not numbers.
    """
    return not inventory_numeric_mentions(
        text,
        unit_surfaces=unit_surfaces,
        ignore_identifier_fragments=ignore_identifier_fragments,
    ).is_empty


#: Separators inside a unit kept as text (``"unit: °C; direction: east"``).
_UNIT_TEXT_SPLIT = re.compile(r"[\s;,:()\[\]]+")


def _unit_text_keys(text: str) -> set[str]:
    """Lookup keys of every token of a unit kept as text on a value node."""
    keys: set[str] = set()
    for token in _UNIT_TEXT_SPLIT.split(text):
        token = token.strip(".")
        if token:
            keys.update(_lookup_keys(token))
    return keys


def measurement_pairs_in_graph(
    graph: RDFGraph,
    ontology_graph: Graph | None = None,
    *,
    unit_properties: Collection[str] = (),
) -> set[tuple[str, str]]:
    """``(canonical_number, unit_surface_key)`` pairs structured in ``graph``.

    Any node, blank or named, contributes every number it states outside an
    annotation (a value, a bound) paired with each unit it states: an object
    of a configured unit property, or, where the context has no unit
    individual to fit, the unit's token kept in a literal on the same node.
    No value property is assumed, so a catalog's own value vocabulary counts
    as the fallback vocabulary does, and so does a number the node holds that
    its own text writes with a unit ("0.25°"). A number with no unit on its
    node, or a number written only in text, does not cover a unit-adjacent
    mention. Unit IRIs map to
    surfaces through :func:`unit_surface_index`; each surface is expanded
    with :func:`~ontocast.util.measurement_lexicon._lookup_keys` so the same
    case and plural rules that recognise a mention also recognise its
    extracted counterpart. When the ontology has no surface for an IRI, the
    IRI's local name is tried as a last resort.

    Args:
        graph: The extracted facts graph.
        ontology_graph: The unit's ontology context (for unit surfaces).
        unit_properties: IRIs of the unit-role properties.

    Returns:
        Pairs keyed for membership tests against mention lookup keys.
    """
    unit_props = set(unit_properties)
    iri_to_surfaces: dict[str, set[str]] = {}
    for surface, iris in unit_surface_index(ontology_graph, unit_properties).items():
        for iri in iris:
            iri_to_surfaces.setdefault(iri, set()).add(surface)

    pairs: set[tuple[str, str]] = set()
    for subject in set(graph.subjects()):
        numbers: set[str] = set()
        unit_keys: set[str] = set()
        texts: list[str] = []
        for _, predicate, obj in graph.triples((subject, None, None)):
            pred = str(predicate)
            if pred in unit_props and isinstance(obj, URIRef):
                for surface in iri_to_surfaces.get(str(obj)) or {_local(str(obj))}:
                    unit_keys.update(_lookup_keys(surface.strip().rstrip(".,;:")))
            elif not isinstance(obj, Literal):
                continue
            elif not is_annotation(predicate) and (
                canonical := canonical_number(str(obj).strip())
            ):
                numbers.add(canonical)
            else:
                unit_keys.update(_unit_text_keys(str(obj)))
                texts.append(str(obj))
        for number in numbers:
            for key in unit_keys:
                pairs.add((number, key))
        # A label such as "0.25°" states the unit glued to the number the node holds.
        for text in texts:
            for mention in unit_adjacent_numbers(text):
                number = canonical_number(mention.value)
                if number in numbers:
                    token = mention.unit.strip().rstrip(".,;:")
                    pairs.update((number, key) for key in _lookup_keys(token))
    return pairs


def _measurement_covered(mention: Mention, present_pairs: set[tuple[str, str]]) -> bool:
    """Whether a unit-adjacent mention is covered by a structured graph pair."""
    canonical = canonical_number(mention.value)
    if canonical is None:
        return False
    token = mention.unit.strip().rstrip(".,;:")
    return any((canonical, key) in present_pairs for key in _lookup_keys(token))


def missing_numeric_inventory(
    text: str,
    graph: RDFGraph,
    *,
    unit_surfaces: Collection[str] = frozenset(),
    ontology_graph: Graph | None = None,
    unit_properties: Collection[str] = (),
    ignore_year_like: bool = True,
    ignore_identifier_fragments: bool = False,
    limit: int = 30,
) -> NumericInventory:
    """The inventory of ``text`` restricted to values absent from the graph.

    Measurements are judged against structured ``(number, unit)`` pairs in
    the graph: a bare numeric literal does not clear a unit-adjacent mention.
    Bare numbers still use the number-only presence set. Capped at ``limit``
    over both lists, measurements first: they are the numbers a later pass
    can act on, so when the cap bites it is the bare numbers that are
    dropped. A warning records how many were.

    Args:
        text: Source text for the unit.
        graph: Graph extracted from that text.
        unit_surfaces: Extra unit surfaces beyond the built-in lexicon.
        ontology_graph: Ontology context used to map unit IRIs to surfaces.
        unit_properties: IRIs of the unit-role properties.
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
    present_numbers = numeric_literals_in_graph(graph)
    present_pairs = measurement_pairs_in_graph(
        graph,
        ontology_graph,
        unit_properties=unit_properties,
    )
    inventory = inventory_numeric_mentions(
        text,
        unit_surfaces=unit_surfaces,
        ignore_year_like=ignore_year_like,
        ignore_identifier_fragments=ignore_identifier_fragments,
    )
    measurements = [
        mention
        for mention in inventory.measurements
        if not _measurement_covered(mention, present_pairs)
    ]
    unclassified = [
        value for value in inventory.unclassified if value not in present_numbers
    ]
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
    ontology_graph: Graph | None = None,
    unit_properties: Collection[str] = (),
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
        ontology_graph: Ontology context used to map unit IRIs to surfaces.
        unit_properties: IRIs of the unit-role properties.

    Returns:
        Canonical decimal strings, capped at ``limit``.
    """
    inventory = missing_numeric_inventory(
        text,
        graph,
        unit_surfaces=unit_surfaces,
        ontology_graph=ontology_graph,
        unit_properties=unit_properties,
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


class _GraphMemo:
    """Indexes of one graph object, kept while its content is unchanged.

    Keyed by the graph's identity, never by graph equality (two graphs with
    one identifier are equal), and checked against a digest of its triples,
    so a graph mutated in place is indexed again. Entries go with the graph.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[int, dict[frozenset[str], _SurfaceIndex]]] = {}

    def get(
        self, graph: Graph, key: frozenset[str], build: Callable[[], _SurfaceIndex]
    ) -> _SurfaceIndex:
        digest = hash(frozenset(graph))
        entry = self._entries.get(id(graph))
        if entry is None or entry[0] != digest:
            if entry is None:
                try:
                    weakref.finalize(graph, self._entries.pop, id(graph), None)
                except TypeError:  # not weakly referenceable: not kept
                    return build()
            entry = (digest, {})
            self._entries[id(graph)] = entry
        index = entry[1].get(key)
        if index is None:
            index = build()
            entry[1][key] = index
        return index


_surface_memo = _GraphMemo()


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
        seeds |= side_classes(URIRef(prop), [graph], "range")
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
    stand next to a number. Memoised per graph object and its content
    (:class:`_GraphMemo`), because the snapshot is shared by reference across
    a whole fan-out and this walks it whole.

    Args:
        ontology_graph: The unit's ontology context; ``None`` yields nothing.
        unit_properties: IRIs of the unit-role properties (``qudt:unit``).

    Returns:
        Surface -> sorted unit IRIs. Treat as read-only.
    """
    if ontology_graph is None or len(ontology_graph) == 0:
        return {}
    key = frozenset(unit_properties)
    return _surface_memo.get(
        ontology_graph, key, lambda: _build_surface_index(ontology_graph, key)
    )


def unit_surfaces_in_ontology(
    ontology_graph: Graph | None, unit_properties: Collection[str] = ()
) -> frozenset[str]:
    """The unit surfaces of an ontology graph; see :func:`unit_surface_index`."""
    return frozenset(unit_surface_index(ontology_graph, unit_properties))


#: Predicates whose literals are an individual's cues, beside the code and
#: symbol literals recognised by local name: what a text writes next to a
#: number to say which individual it means.
_CUE_PREDICATES = (SKOS.altLabel, SKOS.notation)

_cue_memo = _GraphMemo()


def _cue_index(graph: Graph, classes: frozenset[str]) -> _SurfaceIndex:
    closure = {URIRef(cls) for cls in classes}
    frontier = list(closure)
    while frontier:
        current = frontier.pop()
        for sub in graph.subjects(RDFS.subClassOf, current):
            if isinstance(sub, URIRef) and sub not in closure:
                closure.add(sub)
                frontier.append(sub)
    index: dict[str, set[str]] = {}
    for cls in closure:
        for individual in graph.subjects(RDF.type, cls):
            if not isinstance(individual, URIRef):
                continue
            for predicate, value in graph.predicate_objects(individual):
                if not isinstance(value, Literal):
                    continue
                local = _local(str(predicate)).lower()
                if predicate not in _CUE_PREDICATES and not any(
                    token in local for token in _CODE_LOCAL_NAMES
                ):
                    continue
                text = str(value).strip()
                if text and canonical_number(text) is None:
                    index.setdefault(text, set()).add(str(individual))
    return {cue: tuple(sorted(iris)) for cue, iris in index.items()}


def cue_index(ontology_graph: Graph | None, classes: Collection[str]) -> _SurfaceIndex:
    """Cue -> the individuals of ``classes`` declaring it, case preserved.

    The individuals are those typed with one of ``classes`` or with a
    subclass the graph states. Their cues are ``skos:altLabel`` and
    ``skos:notation`` literals and code or symbol literals (a predicate whose
    local name says so); a number is never a cue. Case is kept: two
    individuals may differ by the case of a symbol alone. Memoised per graph
    object and its content (:class:`_GraphMemo`).

    Args:
        ontology_graph: The unit's ontology context; ``None`` yields nothing.
        classes: Class IRIs whose individuals are indexed.

    Returns:
        Cue -> sorted individual IRIs. Treat as read-only.
    """
    if ontology_graph is None or len(ontology_graph) == 0 or not classes:
        return {}
    key = frozenset(classes)
    return _cue_memo.get(ontology_graph, key, lambda: _cue_index(ontology_graph, key))


#: A number written in digits: integer, decimal or exponent form, or with a
#: leading point. A unit or a word may follow it directly ("5kg", "20s").
_STATED_PIECE = re.compile(r"(?<![\w])(\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?!\d)")
#: Digits grouped by thousands: with commas and a decimal point; with points
#: and a decimal comma; or with spaces or apostrophes and either decimal mark.
_STATED_GROUPED = (
    (
        re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)"),
        ",",
    ),
    (
        re.compile(r"(?<![\w.,])\d{1,3}(?:(?:\.\d{3})+,\d+|(?:\.\d{3}){2,})(?![\d.,])"),
        ".",
    ),
    (
        re.compile(
            r"(?<![\w.,'’])\d{1,3}(?:['’ \u00a0\u2009\u202f]\d{3})+(?:[.,]\d+)?(?!\d)"
        ),
        "'’ \u00a0\u2009\u202f",
    ),
)
#: A decimal comma.
_STATED_DECIMAL_COMMA = re.compile(r"(?<![\w.,])\d+,\d+(?![\d,])")
#: Scientific notation: ``1.4 × 10^3``, ``10^-15``, ``10⁻³``, and with a
#: mantissa, an exponent written with its sign alone: ``1.5 × 10−3``.
_STATED_SCIENTIFIC = re.compile(
    r"(?<![\w.])(?:(\d+(?:[.,]\d+)?)\s*[x×·⋅*]\s*)?10"
    r"(?:\s*(?:\^|\*\*)\s*\(?\s*([-−–+]?\d+)\s*\)?|([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
    r"|([-−–]\d+))(?!\d)"
)
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
#: How far a power of ten may rescale a stated number and still ground it.
_MAX_SHIFT = 12


def _magnitude(text: str) -> str | None:
    """The canonical absolute value of a numeric string, or ``None``."""
    canonical = canonical_number(text)
    return None if canonical is None else canonical.lstrip("-")


def stated_number_spans(text: str) -> list[tuple[str, int, int]]:
    """Every number ``text`` writes in digits, with where it is written.

    Each reading counts: a number grouped by thousands (``1,000``,
    ``1'000``, ``1.234,5``) is read whole and in its digit groups, a decimal
    comma (``6,8``) both ways, and scientific notation (``1.4 × 10^3``,
    ``1.5 × 10−3``) as its value. A unit written against the number
    (``5kg``) does not hide it. Values are canonical and unsigned: the sides
    of a range (``10-15``) are two numbers. Number words are not numbers.

    Returns:
        ``(canonical value, start, end)`` per reading, in text order.
    """
    spans: list[tuple[str, int, int]] = []
    for match in _STATED_PIECE.finditer(text):
        value = _magnitude(match.group(0))
        if value is not None:
            spans.append((value, match.start(), match.end()))
    for pattern, separators in _STATED_GROUPED:
        for match in pattern.finditer(text):
            digits = "".join(ch for ch in match.group(0) if ch not in separators)
            value = _magnitude(digits.replace(",", "."))
            if value is not None:
                spans.append((value, match.start(), match.end()))
    for match in _STATED_DECIMAL_COMMA.finditer(text):
        value = _magnitude(match.group(0).replace(",", "."))
        if value is not None:
            spans.append((value, match.start(), match.end()))
    for match in _STATED_SCIENTIFIC.finditer(text):
        if match.group(4) and not match.group(1):
            continue  # "10-15" with no mantissa is a range
        mantissa = (match.group(1) or "1").replace(",", ".")
        exponent = match.group(2) or match.group(3) or match.group(4) or ""
        exponent = exponent.translate(_SUPERSCRIPT)
        exponent = exponent.replace("−", "-").replace("–", "-")
        try:
            value = Decimal(mantissa) * (Decimal(10) ** int(exponent))
        except (InvalidOperation, ValueError):
            continue
        magnitude = _magnitude(str(value))
        if magnitude is not None:
            spans.append((magnitude, match.start(), match.end()))
    spans.sort(key=lambda span: (span[1], span[2]))
    return spans


def stated_numbers(text: str) -> set[str]:
    """The canonical unsigned numbers ``text`` writes; see :func:`stated_number_spans`."""
    return {value for value, _, _ in stated_number_spans(text)}


#: Languages whose number words :func:`worded_numbers` reads.
NUMBER_WORD_LANGUAGES = ("en", "fr", "es", "pt", "de", "it", "nl")
#: A hyphen joining two letters.
_WORD_HYPHEN = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")


def worded_numbers(mention: str) -> set[str]:
    """The canonical unsigned numbers ``mention`` writes, in digits or in words.

    Number words are read compositionally ("twenty-five", "three hundred
    thousand", "fifth") in every language of :data:`NUMBER_WORD_LANGUAGES`.
    Call it on the short mention a number belongs to, never on running text:
    a word of one language can be a number in another ("once" is 11 in
    Spanish), which only a mention that also claims that number makes
    harmless. A hyphen between words is also read as a space, so a compound
    adjective ("two-month") states its number. Vague quantities ("several",
    "a few") are not numbers.
    """
    from text_to_num import alpha2digit

    found = stated_numbers(mention)
    for reading in {mention, _WORD_HYPHEN.sub(" ", mention)}:
        for language in NUMBER_WORD_LANGUAGES:
            found |= stated_numbers(alpha2digit(reading, language, threshold=0))
    return found


def _scaled(number: str) -> tuple[tuple[int, ...], int] | None:
    """``number`` as its significant digits and power of ten, or ``None``."""
    try:
        value = Decimal(number).copy_abs().normalize()
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite():
        return None
    _, digits, exponent = value.as_tuple()
    return tuple(digits), int(exponent)


def grounds(number: str, stated: Collection[str]) -> bool:
    """Whether ``stated`` numbers ground ``number``.

    A number is grounded when it equals a stated number times a power of ten,
    from 10^-12 to 10^12 (``5`` grounds ``5000000`` and ``0.05``): a scale
    word, a percent and scientific notation rescale what is written. The
    comparison is exact. A fraction or a unit conversion is not grounded.

    Args:
        number: A canonical number, signed or not.
        stated: Canonical numbers a text writes (:func:`stated_numbers`).
    """
    own = _scaled(number)
    if own is None:
        return False
    for candidate in stated:
        other = _scaled(candidate)
        if (
            other is not None
            and other[0] == own[0]
            and abs(own[1] - other[1]) <= _MAX_SHIFT
        ):
            return True
    return False
