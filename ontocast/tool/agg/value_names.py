"""Stable names for structured values.

A rendered value (a qualified quantity, a range, an identifier) is a blank
node: it has no identity of its own and belongs to the statement that carries
it, so aggregation never merges or twins it. Served graphs need stable IRIs
(downloads, SPARQL, references to one fact), so after merging each value is
named under its carrier: ``<carrier>.<property local name>``, then a number
when one carrier holds several values on one property.
"""

from __future__ import annotations

from collections import deque

from rdflib import BNode, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.rdfgraph import RDFGraph, is_rdflib_triple, retarget_reifiers
from ontocast.onto.scaffolding import scaffolding_nodes


def _local_name(iri: str) -> str:
    for separator in ("#", "/"):
        _, sep, local = iri.rpartition(separator)
        if sep and local:
            return local
    return iri


def _content_key(graph: RDFGraph, node: BNode) -> tuple[tuple[str, str], ...]:
    """What a value says, for ordering values that share a carrier and property
    (blank-node ids are arbitrary)."""
    return tuple(
        sorted(
            (str(predicate), str(obj) if isinstance(obj, (URIRef, Literal)) else "")
            for predicate, obj in graph.predicate_objects(node)
        )
    )


def carrierless_values(graph: RDFGraph) -> set[BNode]:
    """Blank nodes no statement carries: values with no carrier."""
    scaffolding = scaffolding_nodes(graph)
    return {
        node
        for node in graph.subjects()
        if isinstance(node, BNode)
        and node not in scaffolding
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
    scaffolding = scaffolding_nodes(graph)
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
            if isinstance(obj, BNode) and obj not in names and obj not in scaffolding:
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
    for triple in [t for t in graph if is_rdflib_triple(t)]:
        subject, predicate, obj = triple
        if subject in names or obj in names:
            replacements[triple] = (rename(subject), predicate, rename(obj))
    for old, new in replacements.items():
        graph.remove(old)
        graph.add(new)
    retarget_reifiers(graph, replacements)
    return len(names)


__all__ = ["carrierless_values", "name_values"]
