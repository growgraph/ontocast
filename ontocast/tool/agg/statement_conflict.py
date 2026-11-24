"""Identity read from statements: when two things are distinct, when two values are one.

A statement reports a property of a subject with a result, possibly in a
context (a time, a condition, a procedure). The same property of one thing,
in one context, has one result; so two subjects whose statements report one
property with conflicting results and no differing context link are two
things. :func:`statement_conflict` is that test. The aggregator asks it about
a candidate pair; the facts gate asks it about one node and itself, where a
conflict means two things were merged into one.

A statement is also indexed under itself, so two statements reporting one
property with conflicting results stay two statements.

Two property nodes are distinct when the results reported for them disagree
in dimension: :func:`dimension_conflict`. A result's dimension is that of the
measurement units it names (:func:`value_dimensions`, from the well-known
registry); a result with no unit, or a unit of no known dimension, has none
and decides nothing. :func:`dimension_conflicts` reads the same signal within
one graph: results reported for one property node, or by one statement, that
disagree in dimension. Dimensionless quantities share one dimension, so a
ratio, an angle and a percentage never disagree.

:func:`values_conflict` says when two values state something differently (another
number on one slot, another measurement unit); :func:`values_agree` when they
are one value: they share a number and nothing both state differs.

Roles come from the :class:`~ontocast.tool.facts_validation.statements.StatementContract`;
no vocabulary term is named here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pydantic import BaseModel, ConfigDict
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.annotation import is_annotation
from ontocast.onto.model import FactsValidationFinding, FactsValidationFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import FactScope
from ontocast.onto.well_known import well_known_dimension, well_known_value_classes
from ontocast.tool.facts_validation.statements import (
    StatementContract,
    ValueComponent,
    ValueComponents,
    value_components,
)

from .signatures import canonical_literal

#: Whether two nodes denote one thing; the aggregator compares by cluster.
Same = Callable[[Node, Node], bool]


def _identical(left: Node, right: Node) -> bool:
    return left == right


class StatementRecord(BaseModel):
    """One statement, as the identity checks read it.

    Attributes:
        statement: The statement node.
        properties: The property nodes it reports.
        results: Its results that hold a number, each as its components; a
            literal result is one number that fits any slot.
        context: Its context links, as ``(predicate, value)``; a blank-node
            value is read as the sorted pairs it states.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    statement: Node
    properties: frozenset[Node]
    results: tuple[ValueComponents, ...]
    context: frozenset[tuple[str, Node | tuple[tuple[str, str], ...]]]


def value_dimensions(components: ValueComponents) -> frozenset[str]:
    """The dimensions of the measurement units a value names, as IRIs.

    Empty when it names no individual of a known dimension.
    """
    return frozenset(
        dimension
        for component in components.individuals
        if (dimension := well_known_dimension(str(component.value))) is not None
    )


class StatementIndex:
    """The statements of some graphs, by the subjects they are about and by the
    property nodes they report.

    Build with :meth:`of`; the graphs must not change while it is in use.
    """

    def __init__(
        self,
        by_subject: dict[Node, list[StatementRecord]],
        by_property: dict[Node, list[StatementRecord]] | None = None,
        statements: list[StatementRecord] | None = None,
    ) -> None:
        self._by_subject = by_subject
        self._by_property = by_property or {}
        self._statements = statements or []

    @classmethod
    def of(cls, graphs: Iterable[Graph], contract: StatementContract) -> StatementIndex:
        """Index every statement of ``graphs`` under its subjects and itself,
        and under each property node it reports.

        Each graph is read on its own: a blank result belongs to the graph
        that wrote it.
        """
        subject_links = contract.links("subject")
        property_links = contract.links("property")
        by_subject: dict[Node, list[StatementRecord]] = {}
        by_property: dict[Node, list[StatementRecord]] = {}
        statements: list[StatementRecord] = []
        for graph in graphs:
            about: dict[Node, set[Node]] = {}
            for prop in subject_links.forward:
                for statement, subject in graph.subject_objects(URIRef(prop)):
                    about.setdefault(statement, set()).add(subject)
            for prop in subject_links.inverse:
                for subject, statement in graph.subject_objects(URIRef(prop)):
                    about.setdefault(statement, set()).add(subject)
            reporting = set(about)
            for prop in property_links.forward:
                reporting.update(graph.subjects(URIRef(prop)))
            for prop in property_links.inverse:
                reporting.update(graph.objects(None, URIRef(prop)))
            for statement in sorted(reporting, key=str):
                record = _record(graph, statement, contract)
                statements.append(record)
                if not record.properties:
                    continue
                for prop in sorted(record.properties, key=str):
                    by_property.setdefault(prop, []).append(record)
                if statement not in about:
                    continue
                for subject in sorted(about[statement] | {statement}, key=str):
                    by_subject.setdefault(subject, []).append(record)
        return cls(by_subject, by_property, statements)

    def about(self, node: Node) -> list[StatementRecord]:
        """The statements about ``node``, its own record included."""
        return self._by_subject.get(node, [])

    def subjects(self) -> list[Node]:
        """Every node some statement is about, in IRI order."""
        return sorted(self._by_subject, key=str)

    def reporting(self, prop: Node) -> list[StatementRecord]:
        """The statements reporting the property node ``prop``."""
        return self._by_property.get(prop, [])

    def properties(self) -> list[Node]:
        """Every property node some statement reports, in IRI order."""
        return sorted(self._by_property, key=str)

    def statements(self) -> list[StatementRecord]:
        """Every statement indexed, those reporting no property included."""
        return list(self._statements)

    def dimensions_of(self, prop: Node) -> frozenset[str]:
        """The dimensions of the results reported for ``prop``
        (:func:`value_dimensions`); empty when none has a known one."""
        return frozenset(
            dimension
            for record in self.reporting(prop)
            for result in record.results
            for dimension in value_dimensions(result)
        )


def _linked(graph: Graph, node: Node, forward: frozenset[str], inverse: frozenset[str]):
    for prop in sorted(forward):
        for obj in graph.objects(node, URIRef(prop)):
            yield prop, obj
    for prop in sorted(inverse):
        for subj in graph.subjects(URIRef(prop), node):
            yield prop, subj


def _record(
    graph: Graph, statement: Node, contract: StatementContract
) -> StatementRecord:
    prop_links = contract.links("property")
    result_links = contract.links("result")
    context_links = contract.links("context")
    properties = frozenset(
        obj
        for _, obj in _linked(graph, statement, prop_links.forward, prop_links.inverse)
        if not isinstance(obj, Literal)
    )
    results: list[ValueComponents] = []
    for link, obj in _linked(
        graph, statement, result_links.forward, result_links.inverse
    ):
        view = _value_view(graph, link, obj, contract)
        if view.numbers:
            results.append(view)
    context = frozenset(
        (link, _context_value(graph, obj))
        for link, obj in _linked(
            graph, statement, context_links.forward, context_links.inverse
        )
    )
    return StatementRecord(
        statement=statement,
        properties=properties,
        results=tuple(results),
        context=context,
    )


def _value_view(
    graph: Graph, link: str, node: Node, contract: StatementContract
) -> ValueComponents:
    if isinstance(node, Literal):
        canonical = canonical_literal(node)
        if canonical is None or canonical[1] != "numeric":
            return ValueComponents()
        # A bare number names no slot of its own: it may meet any number.
        return ValueComponents(
            numbers=(
                ValueComponent(
                    predicate=link,
                    properties=frozenset(),
                    value=node,
                    number=canonical[0],
                ),
            )
        )
    return value_components(graph, node, contract)


def _context_value(graph: Graph, node: Node) -> Node | tuple[tuple[str, str], ...]:
    if not isinstance(node, BNode):
        return node
    return tuple(
        sorted(
            (str(predicate), _literal_key(obj))
            for predicate, obj in graph.predicate_objects(node)
            if not is_annotation(predicate) and not isinstance(obj, BNode)
        )
    )


def _literal_key(term: Node) -> str:
    if isinstance(term, Literal):
        canonical = canonical_literal(term)
        if canonical is not None:
            return canonical[0]
    return str(term)


def _context_equal(left: Node | tuple, right: Node | tuple, same: Same) -> bool:
    if isinstance(left, tuple) or isinstance(right, tuple):
        return left == right
    if isinstance(left, Literal) and isinstance(right, Literal):
        return _literal_key(left) == _literal_key(right)
    if isinstance(left, Literal) or isinstance(right, Literal):
        return False
    return same(left, right)


def _contexts_differ(left: StatementRecord, right: StatementRecord, same: Same) -> bool:
    """Whether a context link of one is absent from, or other in, the other."""
    for ours, theirs in ((left.context, right.context), (right.context, left.context)):
        for link, value in ours:
            if not any(
                other_link == link and _context_equal(value, other, same)
                for other_link, other in theirs
            ):
                return True
    return False


def _peers(
    component: ValueComponent, ours: ValueComponents, theirs: ValueComponents
) -> list[ValueComponent]:
    """The components of ``theirs`` that state what ``component`` states.

    Two components on one slot (their properties meet) state one thing. A bare
    number (a literal result, which names no slot) states what a value's one
    number states, and nothing of a value stating several.
    """
    pool = theirs.numbers if component.number is not None else theirs.individuals
    if not component.properties:
        return list(pool) if len(pool) == 1 else [c for c in pool if not c.properties]
    return [
        other
        for other in pool
        if (other.properties and component.on(other.properties))
        or (not other.properties and len(ours.numbers) == 1)
    ]


def _component_equal(left: ValueComponent, right: ValueComponent, same: Same) -> bool:
    if left.number is not None or right.number is not None:
        return left.number == right.number
    return same(left.value, right.value)


def values_conflict(
    left: ValueComponents, right: ValueComponents, same: Same = _identical
) -> bool:
    """Whether a component both values state differs: another number on one
    slot, another individual (a measurement unit) on one property.

    An estimate and its interval state nothing in common and do not conflict.
    """
    for ours, theirs in ((left, right), (right, left)):
        for component in (*ours.numbers, *ours.individuals):
            peers = _peers(component, ours, theirs)
            if peers and not any(
                _component_equal(component, peer, same) for peer in peers
            ):
                return True
    return False


def _texts_differ(left: ValueComponents, right: ValueComponents) -> bool:
    """Whether a property both values state text on holds other text in one."""
    for ours, theirs in ((left, right), (right, left)):
        for text in ours.texts:
            peers = [other.value for other in theirs.texts if text.on(other.properties)]
            if peers and text.value not in peers:
                return True
    return False


def values_agree(
    left: ValueComponents, right: ValueComponents, same: Same = _identical
) -> bool:
    """Whether two values are one: they share a number, no component both
    state differs (:func:`values_conflict`), and no property both state text
    or a date on holds other text in one.

    An estimate and its interval share no number and are two values; one
    number with two different measurement units, or in two currencies
    written as text, is two values.
    """
    shared = any(
        component.number == peer.number
        for component in left.numbers
        for peer in _peers(component, left, right)
    )
    return (
        shared
        and not values_conflict(left, right, same)
        and not _texts_differ(left, right)
    )


def _results_conflict(
    left: StatementRecord, right: StatementRecord, same: Same
) -> bool:
    """Whether every result of one conflicts with every result of the other."""
    if not left.results or not right.results:
        return False
    return all(
        values_conflict(ours, theirs, same)
        for ours in left.results
        for theirs in right.results
    )


def statement_conflict(
    index: StatementIndex, left: Node, right: Node, same: Same = _identical
) -> Node | None:
    """The property on which statements show ``left`` and ``right`` distinct.

    Two statements, one about each (or two about one node when ``left`` is
    ``right``), conflict when they report one property, every result of one
    conflicts with every result of the other (:func:`values_conflict`), and no
    context link of one is absent from or other in the other: a time series
    is not a conflict. Statements that are one (``same``) are never compared.

    Args:
        index: The statements to read.
        left: One node.
        right: The other node, or ``left`` itself.
        same: Whether two nodes denote one thing.

    Returns:
        The reported property of the first conflict found, or ``None``.
    """
    ours = index.about(left)
    theirs = ours if left == right else index.about(right)
    for a in ours:
        for b in theirs:
            if a.statement == b.statement or same(a.statement, b.statement):
                continue
            shared = next(
                (
                    prop
                    for prop in sorted(a.properties, key=str)
                    for other in b.properties
                    if same(prop, other)
                ),
                None,
            )
            if shared is None or _contexts_differ(a, b, same):
                continue
            if _results_conflict(a, b, same):
                return shared
    return None


def dimension_conflict(
    index: StatementIndex, left: Node, right: Node
) -> tuple[frozenset[str], frozenset[str]] | None:
    """Whether the results reported for the property nodes ``left`` and
    ``right`` disagree in dimension.

    They disagree when both have results of a known dimension
    (:meth:`StatementIndex.dimensions_of`) and no dimension is common to the
    two. Inert when either has none: a value without a measurement unit, or
    with a unit of no known dimension, decides nothing.

    Returns:
        The dimensions of ``left`` and of ``right``, or ``None``.
    """
    if left == right:
        return None
    ours = index.dimensions_of(left)
    theirs = index.dimensions_of(right)
    if ours and theirs and ours.isdisjoint(theirs):
        return ours, theirs
    return None


class DimensionConflict(BaseModel):
    """Results that disagree in dimension, within one graph.

    Attributes:
        node: The property node the results are reported for, or the statement
            whose results they are.
        by_property: Whether ``node`` is a property node.
        dimensions: The dimensions of the results, as IRIs.
        anchored: Whether the catalog declares what the property measures (a
            quantity kind or an applicable measurement unit,
            :func:`property_anchor`).
        anchor: The dimensions that declaration names; empty when it names
            none of a known dimension.
        odd: The statements whose results disagree with ``anchor``; empty when
            the anchor names no known dimension, so no single result is the
            odd one.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    node: Node
    by_property: bool
    dimensions: frozenset[str]
    anchored: bool
    anchor: frozenset[str] = frozenset()
    odd: tuple[Node, ...] = ()


def property_anchor(
    catalog: Graph | None, contract: StatementContract, prop: Node
) -> tuple[bool, frozenset[str]]:
    """Whether the catalog declares what ``prop`` measures, and its dimensions.

    The declaration is a link of the contract's ``anchor_links`` from
    ``prop``: a measurement unit, whose dimension is the registry's, or a
    quantity kind, whose dimensions are those of the measurement units the
    catalog links from it the same way.

    Returns:
        Whether ``prop`` is anchored, and the dimensions of its anchor (empty
        when none is of a known dimension).
    """
    if catalog is None or not isinstance(prop, URIRef):
        return False, frozenset()
    graph = catalog

    def targets(node: Node) -> list[Node]:
        return [
            obj
            for link in sorted(contract.anchor_links)
            for obj in graph.objects(node, URIRef(link))
        ]

    declared = targets(prop)
    dimensions: set[str] = set()
    for target in declared:
        dimension = well_known_dimension(str(target))
        if dimension is not None:
            dimensions.add(dimension)
            continue
        for unit in targets(target):
            dimension = well_known_dimension(str(unit))
            if dimension is not None:
                dimensions.add(dimension)
    return bool(declared), frozenset(dimensions)


def _disagree(dimensions: list[frozenset[str]]) -> bool:
    """Whether two of the known ``dimensions`` share none."""
    known = [found for found in dimensions if found]
    return any(
        left.isdisjoint(right)
        for index, left in enumerate(known)
        for right in known[index + 1 :]
    )


def dimension_conflicts(
    index: StatementIndex, catalog: Graph | None, contract: StatementContract
) -> list[DimensionConflict]:
    """Results that disagree in dimension: those reported for one property
    node, then those of one statement not already counted.

    Results under an anchored property (:func:`property_anchor`) whose anchor
    has known dimensions disagree when one shares none of them; that
    statement is the odd one. Otherwise they disagree when two results share
    no dimension. A result without a measurement unit, or with one of no known
    dimension, decides nothing, and so does a dimensionless one against
    another dimensionless one.
    """
    conflicts: list[DimensionConflict] = []
    covered: set[Node] = set()
    for prop in index.properties():
        records = index.reporting(prop)
        results = [
            (record.statement, value_dimensions(result))
            for record in records
            for result in record.results
        ]
        stated = frozenset(d for _, dims in results for d in dims)
        anchored, anchor = property_anchor(catalog, contract, prop)
        if anchor:
            odd = tuple(
                sorted(
                    {s for s, dims in results if dims and dims.isdisjoint(anchor)},
                    key=str,
                )
            )
            if not odd:
                continue
        elif _disagree([dims for _, dims in results]):
            odd = ()
        else:
            continue
        covered |= {record.statement for record in records}
        conflicts.append(
            DimensionConflict(
                node=prop,
                by_property=True,
                dimensions=stated,
                anchored=anchored,
                anchor=anchor,
                odd=odd,
            )
        )
    for record in index.statements():
        if record.statement in covered:
            continue
        dims = [value_dimensions(result) for result in record.results]
        if not _disagree(dims):
            continue
        covered.add(record.statement)
        anchors = [
            property_anchor(catalog, contract, prop)
            for prop in sorted(record.properties, key=str)
        ]
        conflicts.append(
            DimensionConflict(
                node=record.statement,
                by_property=False,
                dimensions=frozenset(d for found in dims for d in found),
                anchored=any(anchored for anchored, _ in anchors),
                anchor=frozenset(d for _, anchor in anchors for d in anchor),
            )
        )
    return conflicts


def dimension_conflict_message(conflict: DimensionConflict) -> str:
    """What a :class:`DimensionConflict` says, for a finding."""
    if not conflict.by_property:
        return (
            f"The results of <{conflict.node}> are in measurement units of "
            "different dimensions, so they measure different kinds of quantity: "
            "state each result with the property that names exactly the "
            "quantity it measures."
        )
    if conflict.odd:
        listed = ", ".join(f"<{node}>" for node in conflict.odd)
        return (
            f"The ontology declares what <{conflict.node}> measures, and the "
            f"results of {listed} are in a measurement unit of another "
            "dimension: give each of those statements the property that names "
            "exactly the quantity it states, or the measurement unit the text "
            "states."
        )
    return (
        f"The results reported for <{conflict.node}> are in measurement units of "
        "different dimensions, so they measure different kinds of quantity: "
        "give each statement the property that names exactly the quantity it "
        "states."
    )


def dimension_conflict_findings(
    graph: Graph,
    ontology_graph: RDFGraph | None,
    fact_namespaces: list[str],
    document: Node | None = None,
) -> list[FactsValidationFinding]:
    """One warning per :class:`DimensionConflict` of the aggregated graph.

    Anchored or not: a conflict a unit could not resolve, and one between
    results that several units wrote, are both served.

    Args:
        graph: The aggregated facts graph.
        ontology_graph: The ontology context, for the catalog's role links and
            anchors.
        fact_namespaces: Namespaces of the fact instances.
        document: The document node, never read as a statement's subject.
    """
    from ontocast.tool.facts_validation.terms import CatalogRoles

    contract = CatalogRoles.of(
        [ontology_graph] if ontology_graph is not None else []
    ).statement_contract(frozenset(well_known_value_classes()))
    scope = FactScope.of(graph, [ns for ns in fact_namespaces if ns], document)
    index = StatementIndex.of([scope.facts()], contract)
    return [
        FactsValidationFinding(
            kind=FactsValidationFindingKind.DIMENSION_CONFLICT,
            severity="warning",
            message=dimension_conflict_message(conflict),
            subject=str(conflict.node),
            values=sorted(conflict.dimensions),
        )
        for conflict in dimension_conflicts(index, ontology_graph, contract)
    ]


def statement_conflict_findings(
    graph: Graph,
    ontology_graph: RDFGraph | None,
    fact_namespaces: list[str],
    document: Node | None = None,
) -> list[FactsValidationFinding]:
    """One warning per fact node that :func:`statement_conflict` shows distinct
    from itself.

    Aggregation vetoes every merge that would produce one, so on a served
    graph the list is empty; a finding is a merge that slipped through, or one
    content unit stating one property twice with conflicting results.

    Args:
        graph: The aggregated facts graph.
        ontology_graph: The ontology context, for the catalog's role links.
        fact_namespaces: Namespaces of the fact instances reported.
        document: The document node, never reported.
    """
    from ontocast.tool.facts_validation.terms import CatalogRoles

    contract = CatalogRoles.of(
        [ontology_graph] if ontology_graph is not None else []
    ).statement_contract(frozenset(well_known_value_classes()))
    scope = FactScope.of(graph, [ns for ns in fact_namespaces if ns], document)
    index = StatementIndex.of([scope.facts()], contract)
    findings: list[FactsValidationFinding] = []
    for node in index.subjects():
        if not isinstance(node, URIRef) or node not in scope.instances:
            continue
        prop = statement_conflict(index, node, node)
        if prop is None:
            continue
        findings.append(
            FactsValidationFinding(
                kind=FactsValidationFindingKind.STATEMENT_CONFLICT,
                severity="warning",
                message=(
                    f"Statements about <{node}> report <{prop}> with conflicting "
                    "results and no differing time, condition or procedure: two "
                    "things in one node, or one property stated twice."
                ),
                subject=str(node),
                predicate=str(prop),
            )
        )
    return findings


__all__ = [
    "DimensionConflict",
    "Same",
    "StatementIndex",
    "StatementRecord",
    "dimension_conflict",
    "dimension_conflict_findings",
    "dimension_conflict_message",
    "dimension_conflicts",
    "property_anchor",
    "statement_conflict",
    "statement_conflict_findings",
    "value_dimensions",
    "values_agree",
    "values_conflict",
]
