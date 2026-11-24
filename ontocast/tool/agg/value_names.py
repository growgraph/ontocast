"""Identity and stable names for structured values.

A rendered value (a qualified quantity, a range, an identifier) is a blank
node: it has no name of its own and belongs to the statement that carries it.
Its identity is its content: two values (nodes of a value class) on one
carrier and property are one value when they share a number and neither a
component nor a text both state differs (:func:`merge_equal_values`), so one
number written by two units, or twice by one, is served once. Served graphs need stable IRIs (downloads, SPARQL,
references to one fact), so after merging each value is named under its
carrier: ``<carrier>.<property local name>``, then a number when one carrier
holds several values on one property.
"""

from __future__ import annotations

from collections import deque
from typing import cast

from rdflib import RDF, BNode, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.rdfgraph import RDFGraph, retarget_reifiers
from ontocast.onto.scaffolding import FactScope
from ontocast.tool.facts_validation.statements import (
    StatementContract,
    ValueComponents,
    value_components,
)
from ontocast.tool.facts_validation.terms import _local_name

from .statement_conflict import values_agree


def _content_key(graph: RDFGraph, node: BNode) -> tuple[tuple[str, str], ...]:
    """What a value says, for ordering values that share a carrier and property
    (blank-node ids are arbitrary)."""
    return tuple(
        sorted(
            (str(predicate), str(obj) if isinstance(obj, (URIRef, Literal)) else "")
            for predicate, obj in graph.predicate_objects(node)
        )
    )


def merge_equal_values(graph: RDFGraph, contract: StatementContract) -> int:
    """Make values on one carrier and property that say one thing one value, in place.

    Only blank nodes typed with a value class (:meth:`StatementContract.is_value`)
    are values here. Two of them on one carrier and property are one value
    when they share a number, no component both state differs, and no
    property both state text on holds other text in one
    (:func:`~ontocast.tool.agg.statement_conflict.values_agree`). Values are
    taken in content order; each joins the first group whose union it agrees
    with, and the group's first value survives with the union of their
    triples, keeping one lexical form of each number on a property (the
    shortest, then the least). Reifiers quoting a rewritten statement are
    repointed. An estimate and its interval share no number and stay two.

    Returns:
        The number of values merged away.
    """
    scope = FactScope.of(graph, None)
    replaced: dict[BNode, BNode] = {}
    for carrier in sorted(set(graph.subjects()), key=str):
        if isinstance(carrier, BNode) and not scope.holds(carrier):
            continue
        by_predicate: dict[str, list[BNode]] = {}
        for predicate, obj in graph.predicate_objects(carrier):
            if (
                isinstance(obj, BNode)
                and scope.holds(obj)
                and contract.is_value(graph.objects(obj, RDF.type))
            ):
                by_predicate.setdefault(str(predicate), []).append(obj)
        for values in by_predicate.values():
            if len(values) < 2:
                continue
            groups: list[tuple[BNode, ValueComponents]] = []
            for value in sorted(set(values), key=lambda n: _content_key(graph, n)):
                view = value_components(graph, value, contract)
                for index, (survivor, union) in enumerate(groups):
                    if values_agree(view, union):
                        replaced[value] = survivor
                        groups[index] = (survivor, _union(union, view))
                        break
                else:
                    groups.append((value, view))
    if not replaced:
        return 0

    def rename(term: Node) -> Node:
        # A survivor merged away on another carrier passes its values on.
        seen: set[Node] = set()
        while isinstance(term, BNode) and term in replaced and term not in seen:
            seen.add(term)
            term = replaced[term]
        return term

    replacements: dict[tuple, tuple] = {}
    for triple in list(graph):
        subject, predicate, obj = triple
        if subject in replaced or obj in replaced:
            replacements[triple] = (rename(subject), predicate, rename(obj))
    for old in replacements:
        graph.remove(old)
    for new in replacements.values():
        graph.add(new)
    survivors = {rename(value) for value in replaced}
    replacements.update(_one_form_per_number(graph, survivors, contract))
    retarget_reifiers(graph, replacements)
    return len(replaced)


def _union(left: ValueComponents, right: ValueComponents) -> ValueComponents:
    def joined(ours, theirs):
        return tuple(sorted(set(ours) | set(theirs), key=_component_order))

    return ValueComponents(
        numbers=joined(left.numbers, right.numbers),
        individuals=joined(left.individuals, right.individuals),
        texts=joined(left.texts, right.texts),
    )


def _one_form_per_number(
    graph: RDFGraph, values: set[Node], contract: StatementContract
) -> dict[tuple, tuple]:
    """Keep one lexical form of each number a merged value states on a property.

    ``12.5`` and ``12.50`` are one number; the shortest form stays, then the
    least. Returns each removed triple with the one kept in its place.
    """
    moved: dict[tuple, tuple] = {}
    for value in sorted(values, key=str):
        forms: dict[tuple[str, str], list[Literal]] = {}
        for number in value_components(graph, value, contract).numbers:
            forms.setdefault((number.predicate, number.number or ""), []).append(
                cast(Literal, number.value)
            )
        for (predicate, _), literals in sorted(forms.items()):
            if len(literals) < 2:
                continue
            kept, *rest = sorted(
                literals,
                key=lambda lit: (len(str(lit)), str(lit), str(lit.datatype or "")),
            )
            for literal in rest:
                triple = (value, URIRef(predicate), literal)
                graph.remove(triple)
                moved[triple] = (value, URIRef(predicate), kept)
    return moved


def _component_order(component) -> tuple[str, str]:
    return component.predicate, str(component.value)


def carrierless_values(graph: RDFGraph) -> set[BNode]:
    """Blank nodes no statement carries: values with no carrier."""
    scope = FactScope.of(graph, None)
    return {
        node
        for node in graph.subjects()
        if isinstance(node, BNode)
        and scope.holds(node)
        and (None, None, node) not in graph
    }


def name_values(graph: RDFGraph) -> int:
    """Give every carried blank-node value an IRI under its carrier, in place.

    Walks from named nodes through blank-node objects, so a nested value is
    named under its parent value. Reifiers quoting a renamed statement are
    repointed. Blank nodes no named node reaches (and reifiers) stay as they are.

    Returns:
        The number of values named.
    """
    scope = FactScope.of(graph, None)
    taken = {str(node) for node in graph.all_nodes() if isinstance(node, URIRef)}
    names: dict[BNode, URIRef] = {}
    queue: deque[Node] = deque(
        sorted(
            (node for node in set(graph.subjects()) if isinstance(node, URIRef)),
            key=str,
        )
    )
    while queue:
        carrier = queue.popleft()
        carrier_iri = (
            names.get(carrier, carrier) if isinstance(carrier, BNode) else carrier
        )
        values: dict[str, list[BNode]] = {}
        for predicate, obj in graph.predicate_objects(carrier):
            if isinstance(obj, BNode) and obj not in names and scope.holds(obj):
                values.setdefault(str(predicate), []).append(obj)
        for predicate in sorted(values):
            base = f"{carrier_iri}.{_local_name(predicate)}"
            for value in sorted(
                values[predicate], key=lambda n: _content_key(graph, n)
            ):
                if value in names:
                    continue
                iri, number = base, 2
                while iri in taken:
                    iri, number = f"{base}{number}", number + 1
                taken.add(iri)
                names[value] = URIRef(iri)
                queue.append(value)
    if not names:
        return 0

    def rename(term: Node) -> Node:
        return names.get(term, term) if isinstance(term, BNode) else term

    replacements: dict[tuple, tuple] = {}
    for triple in list(graph):
        subject, predicate, obj = triple
        if subject in names or obj in names:
            replacements[triple] = (rename(subject), predicate, rename(obj))
    for old, new in replacements.items():
        graph.remove(old)
        graph.add(new)
    retarget_reifiers(graph, replacements)
    return len(names)


__all__ = ["carrierless_values", "merge_equal_values", "name_values"]
