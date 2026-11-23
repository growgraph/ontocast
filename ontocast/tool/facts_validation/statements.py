"""The statement model every facts rule reads.

A *statement* is an instance of a statement class: it is about something
else, and its links play roles (``subject``, ``property``, ``result``,
``context``). A *value* is a structured node a statement carries, blank or
named; its *components* are its numbers and the individuals it names.

:class:`StatementContract` is built once per catalog state by
:meth:`~ontocast.tool.facts_validation.terms.CatalogRoles.statement_contract`;
:func:`value_components` reads one value node through it. Neither names a
vocabulary term: the roles and classes come from the well-known registry and
the catalog.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

from pydantic import BaseModel, ConfigDict, Field
from rdflib import OWL, RDF, Graph, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.annotation import is_annotation
from ontocast.onto.well_known import StatementRoleName, well_known_vocabulary
from ontocast.tool.agg.signatures import canonical_literal


class RoleLinks(BaseModel):
    """The properties playing one statement role, closed over the catalog.

    Attributes:
        forward: Properties running from a statement.
        inverse: Properties running to a statement.
    """

    model_config = ConfigDict(frozen=True)

    forward: frozenset[str] = frozenset()
    inverse: frozenset[str] = frozenset()


class StatementContract(BaseModel):
    """What the catalog says a statement and a value are.

    Attributes:
        statement_classes: Statement class -> the roles its instances must
            carry; every catalog class under a well-known statement class
            inherits that class's roles.
        required_links: Statement class -> properties a catalog restriction
            (``owl:someValuesFrom``, or a minimum or exact cardinality of at
            least one) requires on it, its superclasses' included.
        value_classes: Classes whose instances are values, with every catalog
            class under one.
        roles: Role -> its links (:meth:`CatalogRoles.role_links`).
        super_properties: Property -> its catalog super-properties, for each
            property that has one.
        closed_ranges: Property -> the classes of its range whose members are
            known (catalog individuals or a shipped term list).
        link_classes: Role link -> the statement classes its declared side
            (domain from the statement, range to it) admits; a link absent
            here admits every class.
        anchor_links: Properties declaring what a property node measures (a
            quantity kind, an applicable measurement unit), with every catalog
            sub-property of one.
        numeric_properties: Properties whose declared datatype range (the
            catalog's or a well-known vocabulary's) is numeric.
        property_classes: Classes whose instances a statement's ``property``
            link names (what is observed or acted on): the ranges the
            catalog and the well-known vocabularies declare for those links,
            with every catalog class under one.
    """

    model_config = ConfigDict(frozen=True)

    statement_classes: dict[str, frozenset[StatementRoleName]] = Field(
        default_factory=dict
    )
    required_links: dict[str, frozenset[str]] = Field(default_factory=dict)
    value_classes: frozenset[str] = frozenset()
    roles: dict[StatementRoleName, RoleLinks] = Field(default_factory=dict)
    super_properties: dict[str, frozenset[str]] = Field(default_factory=dict)
    closed_ranges: dict[str, frozenset[str]] = Field(default_factory=dict)
    link_classes: dict[str, frozenset[str]] = Field(default_factory=dict)
    anchor_links: frozenset[str] = frozenset()
    numeric_properties: frozenset[str] = frozenset()
    property_classes: frozenset[str] = frozenset()

    def links(self, role: StatementRoleName) -> RoleLinks:
        """The links of ``role``; none when the catalog states none."""
        return self.roles.get(role, RoleLinks())

    def required_roles(
        self, types: Iterable[Node | str]
    ) -> frozenset[StatementRoleName]:
        """The roles an instance of ``types`` must carry."""
        found: set[StatementRoleName] = set()
        for cls in types:
            found |= self.statement_classes.get(str(cls), frozenset())
        return frozenset(found)

    def is_statement(self, types: Iterable[Node | str]) -> bool:
        """Whether one of ``types`` is a statement class."""
        return any(str(cls) in self.statement_classes for cls in types)

    def is_value(self, types: Iterable[Node | str]) -> bool:
        """Whether one of ``types`` is a value class."""
        return any(str(cls) in self.value_classes for cls in types)

    def serves(self, link: str, types: Iterable[Node | str]) -> bool:
        """Whether ``link`` can play its role for an instance of ``types``.

        It can when its declared side admits one of the types; a link with no
        declared side admits every type.
        """
        admitted = self.link_classes.get(link)
        return admitted is None or any(str(cls) in admitted for cls in types)

    def properties_of(self, prop: str) -> frozenset[str]:
        """``prop`` and its catalog super-properties."""
        return frozenset({prop}) | self.super_properties.get(prop, frozenset())


def own_terms(graph: Graph | None, terms: Iterable[str]) -> list[str]:
    """The ``terms`` that ``graph`` names and no well-known vocabulary owns, sorted.

    Read over a unit's ontology context: the catalog's own terms in it.
    """
    if graph is None:
        return []
    present = {
        str(term) for triple in graph for term in triple if isinstance(term, URIRef)
    }
    return sorted(
        term
        for term in terms
        if term in present and well_known_vocabulary(term) is None
    )


class ValueComponent(BaseModel):
    """One non-annotation triple of a value node.

    Attributes:
        predicate: The property it is written on.
        properties: ``predicate`` and its catalog super-properties: two
            components compare equal on a property when these meet.
        value: The literal or the IRI.
        number: The canonical decimal form of a numeric literal, else ``None``.
        closed_range: For an individual, the classes of the property's range
            whose members are known; empty for a number or an open range.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    predicate: str
    properties: frozenset[str]
    value: Literal | URIRef
    number: str | None = None
    closed_range: frozenset[str] = frozenset()

    def on(self, properties: Collection[str]) -> bool:
        """Whether the component is written on one of ``properties`` or under one."""
        return not self.properties.isdisjoint(properties)


class ValueComponents(BaseModel):
    """The numbers and the individuals of a value node, and its other literals.

    Attributes:
        numbers: Numeric literals, by predicate and value.
        individuals: IRIs it names, by predicate and value.
        texts: Its other non-annotation literals (text, dates), by predicate
            and value: no component, but a value stating another one on a
            property is another value.
    """

    model_config = ConfigDict(frozen=True)

    numbers: tuple[ValueComponent, ...] = ()
    individuals: tuple[ValueComponent, ...] = ()
    texts: tuple[ValueComponent, ...] = ()

    def numbers_on(self, properties: Collection[str]) -> tuple[ValueComponent, ...]:
        """The numbers written on ``properties`` or a sub-property of one."""
        return tuple(number for number in self.numbers if number.on(properties))


#: Predicates that type, alias or annotate a node: none states a component.
_NOT_COMPONENTS: frozenset[Node] = frozenset({RDF.type, OWL.sameAs})


def value_components(
    graph: Graph, node: Node, contract: StatementContract
) -> ValueComponents:
    """The numbers and individuals ``node`` states, blank or named, and its texts.

    A component is a triple of ``node`` that is not a type, an alias or an
    annotation (:func:`~ontocast.onto.annotation.is_annotation`): a number or
    an individual (an IRI). A literal is a number when it is typed numeric, or
    untyped with a numeric form on a property whose declared range is numeric
    (:attr:`StatementContract.numeric_properties`): a digit string elsewhere
    is text, such as an identifier. Text and dates are kept apart as
    ``texts``; nested blank nodes are none of these. Ordered by predicate and
    value.
    """
    numbers: list[ValueComponent] = []
    individuals: list[ValueComponent] = []
    texts: list[ValueComponent] = []
    for predicate, obj in graph.predicate_objects(node):
        if predicate in _NOT_COMPONENTS or is_annotation(predicate):
            continue
        properties = contract.properties_of(str(predicate))
        if isinstance(obj, Literal):
            if not _is_number(obj, properties, contract):
                texts.append(
                    ValueComponent(
                        predicate=str(predicate), properties=properties, value=obj
                    )
                )
                continue
            canonical = canonical_literal(obj)
            assert canonical is not None
            numbers.append(
                ValueComponent(
                    predicate=str(predicate),
                    properties=properties,
                    value=obj,
                    number=canonical[0],
                )
            )
        elif isinstance(obj, URIRef):
            closed: set[str] = set()
            for prop in properties:
                closed |= contract.closed_ranges.get(prop, frozenset())
            individuals.append(
                ValueComponent(
                    predicate=str(predicate),
                    properties=properties,
                    value=obj,
                    closed_range=frozenset(closed),
                )
            )
    return ValueComponents(
        numbers=tuple(sorted(numbers, key=_order)),
        individuals=tuple(sorted(individuals, key=_order)),
        texts=tuple(sorted(texts, key=_order)),
    )


def _is_number(
    obj: Literal, properties: frozenset[str], contract: StatementContract
) -> bool:
    """Whether ``obj`` is a number on a property among ``properties``."""
    if obj.language is not None:
        return False
    canonical = canonical_literal(obj)
    if canonical is None or canonical[1] != "numeric":
        return False
    return obj.datatype is not None or not properties.isdisjoint(
        contract.numeric_properties
    )


def _order(component: ValueComponent) -> tuple[str, str]:
    return component.predicate, str(component.value)


__all__ = [
    "RoleLinks",
    "StatementContract",
    "ValueComponent",
    "ValueComponents",
    "own_terms",
    "value_components",
]
