"""A property's domain and range, read from every predicate that states one.

Two kinds of statement give a side:

- ``rdfs:domain`` / ``rdfs:range`` entail: every subject of the property is
  an instance of the domain, every object of the range;
- ``schema:domainIncludes`` / ``schema:rangeIncludes`` list expected classes.
  They entail nothing, and they add up: each document that uses the property
  for a new kind of subject may add its class.

Readers take a property's sides from here, so both kinds are read the same
way everywhere. A reader that infers a type from a side reads the ``rdfs``
sides only (or a listed side that names one class); a reader that checks fit,
ranks candidates or shows the sides reads both.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from rdflib import RDF, RDFS, XSD, Graph, URIRef

Side = Literal["domain", "range"]
SideSource = Literal["rdfs", "includes"]

SCHEMA_NAMESPACES: tuple[str, ...] = ("https://schema.org/", "http://schema.org/")

#: The predicate a writer uses to list an expected class of a side.
DOMAIN_INCLUDES = URIRef("https://schema.org/domainIncludes")
RANGE_INCLUDES = URIRef("https://schema.org/rangeIncludes")

_RDFS_PREDICATE: dict[Side, URIRef] = {"domain": RDFS.domain, "range": RDFS.range}
_INCLUDES_NAME: dict[Side, str] = {"domain": "domainIncludes", "range": "rangeIncludes"}


def includes_predicates(side: Side) -> tuple[URIRef, ...]:
    """``schema:<side>Includes`` in both spellings of the schema.org namespace."""
    return tuple(URIRef(ns + _INCLUDES_NAME[side]) for ns in SCHEMA_NAMESPACES)


def side_predicates(side: Side) -> tuple[URIRef, ...]:
    """Every predicate that states ``side``: the ``rdfs`` one first."""
    return (_RDFS_PREDICATE[side], *includes_predicates(side))


DOMAIN_PREDICATES: frozenset[URIRef] = frozenset(side_predicates("domain"))
RANGE_PREDICATES: frozenset[URIRef] = frozenset(side_predicates("range"))
#: Every predicate that states a domain or a range.
SIDE_PREDICATES: frozenset[URIRef] = DOMAIN_PREDICATES | RANGE_PREDICATES


def side_of(predicate: URIRef) -> Side | None:
    """The side ``predicate`` states, or ``None`` for any other predicate."""
    if predicate in DOMAIN_PREDICATES:
        return "domain"
    if predicate in RANGE_PREDICATES:
        return "range"
    return None


def source_of(predicate: URIRef) -> SideSource | None:
    """``rdfs`` or ``includes`` for a side predicate, else ``None``."""
    if predicate in (RDFS.domain, RDFS.range):
        return "rdfs"
    if predicate in SIDE_PREDICATES:
        return "includes"
    return None


def property_sides(
    prop: URIRef, graphs: Iterable[Graph], side: Side
) -> dict[URIRef, SideSource]:
    """The named classes (or datatypes) of ``prop``'s ``side``, with their source.

    A class stated both ways counts as ``rdfs``.
    """
    found: dict[URIRef, SideSource] = {}
    for graph in graphs:
        for predicate in side_predicates(side):
            source = source_of(predicate)
            for value in graph.objects(prop, predicate):
                if isinstance(value, URIRef) and source is not None:
                    if found.get(value) != "rdfs":
                        found[value] = source
    return found


def side_classes(prop: URIRef, graphs: Iterable[Graph], side: Side) -> set[URIRef]:
    """The named values of ``prop``'s ``side``, from either source."""
    return set(property_sides(prop, graphs, side))


def sides_by_property(
    graph: Graph, side: Side, *, source: SideSource | None = None
) -> dict[URIRef, set[URIRef]]:
    """Property -> the named values of its ``side`` in ``graph``.

    Args:
        graph: Where the sides are read.
        side: ``domain`` or ``range``.
        source: Only the ``rdfs`` or only the ``includes`` statements; both
            when ``None``.
    """
    found: dict[URIRef, set[URIRef]] = {}
    for predicate in side_predicates(side):
        if source is not None and source_of(predicate) != source:
            continue
        for prop, value in graph.subject_objects(predicate):
            if isinstance(prop, URIRef) and isinstance(value, URIRef):
                found.setdefault(prop, set()).add(value)
    return found


def properties_with_side(
    graph: Graph, value: URIRef, side: Side | None = None
) -> set[URIRef]:
    """The properties whose ``side`` (either when ``None``) names ``value``."""
    sides: tuple[Side, ...] = (side,) if side is not None else ("domain", "range")
    return {
        prop
        for each in sides
        for predicate in side_predicates(each)
        for prop in graph.subjects(predicate, value)
        if isinstance(prop, URIRef)
    }


def is_datatype_value(value: URIRef, graph: Graph | None = None) -> bool:
    """An XSD datatype, a literal type, or a class ``graph`` declares a datatype."""
    if str(value).startswith(str(XSD)) or value in (
        RDFS.Literal,
        RDF.langString,
        RDF.PlainLiteral,
        RDFS.Datatype,
    ):
        return True
    return graph is not None and (value, RDF.type, RDFS.Datatype) in graph


__all__ = [
    "DOMAIN_INCLUDES",
    "DOMAIN_PREDICATES",
    "RANGE_INCLUDES",
    "RANGE_PREDICATES",
    "SCHEMA_NAMESPACES",
    "SIDE_PREDICATES",
    "Side",
    "SideSource",
    "includes_predicates",
    "is_datatype_value",
    "properties_with_side",
    "property_sides",
    "side_classes",
    "side_of",
    "side_predicates",
    "sides_by_property",
    "source_of",
]
