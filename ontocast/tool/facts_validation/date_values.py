"""Date values for nodes that stand for a date, read from their labels.

A node standing for a date (an instant, a day, a month, a year) must say which
date it is with a typed ISO 8601 literal: a label alone is text no query can
compare or order. The render is told so; this pass makes it hold in code. A
node is taken to stand for a date when it is the object of a temporal property,
is typed with an OWL-Time class, or carries a label that is a whole calendar
date or month. When its label parses as a date and it carries no date-typed
literal, the parsed value is added: through the one catalog property whose
domain fits the node and whose range is the value's datatype, else OWL-Time's
``inXSD*`` property for a ``time:Instant``, else ``rdf:value``. Nothing is
removed, and a label that does not parse is left as it is. Month names are
read in English.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, datetime

from rdflib import RDF, RDFS, BNode, Literal, URIRef
from rdflib.namespace import TIME, XSD
from rdflib.term import Node

from ontocast.onto.model import FactsGateRepairKind, GraphRepairRecord
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import (
    well_known_datatype_ranges,
    well_known_temporal_properties,
)
from ontocast.onto.xsd_lexical import CALENDAR_DATATYPES
from ontocast.tool.facts_validation.terms import fact_instances, superclass_closure

#: Datatypes that state a date or a time; a node holding one is dated.
DATE_DATATYPES: frozenset[URIRef] = CALENDAR_DATATYPES

#: Properties of the well-known vocabularies whose object is a time.
_TEMPORAL_PROPERTIES: frozenset[URIRef] = frozenset(
    URIRef(iri) for iri in well_known_temporal_properties()
)

#: OWL-Time's value property per datatype, for a node typed ``time:Instant``:
#: the inverse of its published ranges.
_INSTANT_PROPERTY: dict[URIRef, URIRef] = {
    URIRef(datatype): URIRef(prop)
    for prop, datatype in well_known_datatype_ranges().items()
    if prop.startswith(str(TIME))
}

_MONTHS: dict[str, int] = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH = r"(?P<month>[a-z]+)\.?"
_YEAR = r"(?P<year>\d{4})"
_DAY = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
_MONTH_YEAR = re.compile(rf"{_MONTH}\s+{_YEAR}")
_DAY_MONTH_YEAR = re.compile(rf"{_DAY}\s+{_MONTH},?\s+{_YEAR}")
_MONTH_DAY_YEAR = re.compile(rf"{_MONTH}\s+{_DAY},?\s+{_YEAR}")
_ISO_YEAR = re.compile(r"\d{4}")
_ISO_YEAR_MONTH = re.compile(r"(\d{4})-(\d{2})")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_DATE_TIME = re.compile(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}.*")


def parse_date_label(label: str, *, allow_year: bool) -> Literal | None:
    """The typed ISO 8601 literal a label names as a whole, or ``None``.

    Reads ISO forms (``2021``, ``2021-06``, ``2021-06-27``, a date-time) and
    English calendar forms (``June 2021``, ``27 June 2021``, ``June 27, 2021``,
    abbreviated months too). The whole label must be the date: ``mid-June
    2021`` or ``summer 2021`` name no single value and give ``None``.

    Args:
        label: The node's label.
        allow_year: Whether a bare four-digit year counts; it is a date only
            where the node's position already says it stands for a time.

    Returns:
        Literal | None: An ``xsd:gYear``, ``xsd:gYearMonth``, ``xsd:date`` or
        ``xsd:dateTime`` literal.
    """
    text = " ".join(label.split()).strip().lower()
    if _ISO_YEAR.fullmatch(text):
        return Literal(text, datatype=XSD.gYear) if allow_year else None
    if match := _ISO_YEAR_MONTH.fullmatch(text):
        if 1 <= int(match.group(2)) <= 12:
            return Literal(text, datatype=XSD.gYearMonth)
        return None
    if _ISO_DATE.fullmatch(text):
        try:
            return Literal(date.fromisoformat(text).isoformat(), datatype=XSD.date)
        except ValueError:
            return None
    if _ISO_DATE_TIME.fullmatch(text):
        try:
            return Literal(
                datetime.fromisoformat(text.upper()).isoformat(),
                datatype=XSD.dateTime,
            )
        except ValueError:
            pass
    if match := _MONTH_YEAR.fullmatch(text):
        month = _MONTHS.get(match.group("month"))
        if month is not None:
            return Literal(
                f"{match.group('year')}-{month:02d}", datatype=XSD.gYearMonth
            )
        return None
    for pattern in (_DAY_MONTH_YEAR, _MONTH_DAY_YEAR):
        if match := pattern.fullmatch(text):
            month = _MONTHS.get(match.group("month"))
            if month is None:
                return None
            try:
                day = date(int(match.group("year")), month, int(match.group("day")))
            except ValueError:
                return None
            return Literal(day.isoformat(), datatype=XSD.date)
    return None


def _temporal_properties(ontology_graph: RDFGraph) -> set[URIRef]:
    """The well-known temporal properties plus the catalog's own.

    A catalog property is temporal when its declared range is an OWL-Time
    class or a date datatype.
    """
    found = set(_TEMPORAL_PROPERTIES)
    for predicate, _, range_iri in ontology_graph.triples((None, RDFS.range, None)):
        if not isinstance(predicate, URIRef) or not isinstance(range_iri, URIRef):
            continue
        if range_iri in DATE_DATATYPES or str(range_iri).startswith(str(TIME)):
            found.add(predicate)
            continue
        if any(
            str(parent).startswith(str(TIME))
            for parent in superclass_closure(range_iri, ontology_graph)
        ):
            found.add(predicate)
    return found


def _temporal_type(
    node: Node,
    graph: RDFGraph,
    ontology_graph: RDFGraph,
) -> tuple[bool, bool]:
    """Whether ``node`` is typed with an OWL-Time class, and with ``time:Instant``."""
    temporal = instant = False
    for type_iri in graph.objects(node, RDF.type):
        if not isinstance(type_iri, URIRef):
            continue
        closure = superclass_closure(type_iri, ontology_graph) | {type_iri}
        if any(str(parent).startswith(str(TIME)) for parent in closure):
            temporal = True
        if TIME.Instant in closure:
            instant = True
    return temporal, instant


def _date_property(
    node: Node, datatype: URIRef, graph: RDFGraph, ontology_graph: RDFGraph
) -> URIRef | None:
    """The one context property that holds a ``datatype`` date of ``node``'s kind.

    A datatype property whose range is ``datatype`` and whose domain is one of
    the node's types or their superclasses. ``None`` when no property, or more
    than one, fits.
    """
    kinds: set[URIRef] = set()
    for type_iri in graph.objects(node, RDF.type):
        if isinstance(type_iri, URIRef):
            kinds |= superclass_closure(type_iri, ontology_graph) | {type_iri}
    fitting = {
        predicate
        for predicate in ontology_graph.subjects(RDFS.range, datatype)
        if isinstance(predicate, URIRef)
        and any(
            domain in kinds for domain in ontology_graph.objects(predicate, RDFS.domain)
        )
    }
    return next(iter(fitting)) if len(fitting) == 1 else None


def _is_dated(node: Node, graph: RDFGraph) -> bool:
    return any(
        isinstance(obj, Literal) and obj.datatype in DATE_DATATYPES
        for obj in graph.objects(node, None)
    )


def add_date_values(
    graph: RDFGraph,
    ontology_graph: RDFGraph,
    fact_namespaces: Sequence[str],
) -> list[GraphRepairRecord]:
    """Give every undated node that stands for a date the value its label names.

    Args:
        graph: The aggregated facts graph; edited in place.
        ontology_graph: The ontology context the facts were validated against.
        fact_namespaces: Namespaces holding minted instances.

    Returns:
        list[GraphRepairRecord]: One ``date_value_added`` record per node, its
        target the added literal.
    """
    temporal_properties = _temporal_properties(ontology_graph)
    in_position: set[Node] = {
        obj
        for predicate in temporal_properties
        for obj in graph.objects(None, predicate)
        if isinstance(obj, BNode | URIRef)
    }
    instances = fact_instances(graph, fact_namespaces)
    # A structured value under a temporal property is a blank node.
    candidates = instances | {node for node in in_position if isinstance(node, BNode)}
    records: list[GraphRepairRecord] = []
    for node in sorted(candidates, key=str):
        if _is_dated(node, graph):
            continue
        temporal, instant = _temporal_type(node, graph, ontology_graph)
        positioned = node in in_position or temporal
        labels = sorted(
            str(label)
            for label in graph.objects(node, RDFS.label)
            if isinstance(label, Literal)
        )
        value = next(
            (
                parsed
                for label in labels
                if (parsed := parse_date_label(label, allow_year=positioned))
                is not None
            ),
            None,
        )
        if value is None:
            continue
        datatype = URIRef(str(value.datatype))
        predicate = (
            _date_property(node, datatype, graph, ontology_graph)
            or (_INSTANT_PROPERTY.get(datatype) if instant else None)
            or RDF.value
        )
        graph.add((node, predicate, value))
        records.append(
            GraphRepairRecord(
                kind=FactsGateRepairKind.DATE_VALUE_ADDED,
                source=str(node),
                target=f'"{value}"^^<{value.datatype}>',
            )
        )
    return records


__all__ = ["DATE_DATATYPES", "add_date_values", "parse_date_label"]
