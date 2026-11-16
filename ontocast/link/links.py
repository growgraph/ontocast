"""The link pass: one document's facts graph to its identity links graph.

A pure function of the document graph and the catalog's declarations. It
reads nothing else, so a document's links never depend on which documents
were linked before it, and the resolution graph built one document at a time
equals the one built from all of them at once.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from itertools import product

from rdflib import RDF, RDFS, SKOS, Literal, URIRef
from rdflib.namespace import DCTERMS, FOAF, XSD
from rdflib.term import Node

from ontocast.onto.constants import (
    IDENTITY_LINKS_SUFFIX,
    ONTOCAST,
    auxiliary_graph_uri,
)
from ontocast.onto.rdfgraph import RDFGraph, canonical_literal
from ontocast.tool.agg.signatures import normalize_string_value

from .declarations import IdentityDeclarations

#: Class of a match node: "instances of a class carrying this name or key".
MATCH = ONTOCAST.Match
#: How a match node is keyed: ``key`` | ``ifp`` | ``identifier`` | ``label``.
MATCH_KIND = ONTOCAST.matchKind
#: The class a ``key``, ``identifier`` or ``label`` match node is scoped to.
MATCH_CLASS = ONTOCAST.matchClass
#: The property (or properties) a ``key`` or ``ifp`` match node is keyed on.
MATCH_PROPERTY = ONTOCAST.matchProperty
#: The normalised value a match node is keyed on.
MATCH_VALUE = ONTOCAST.matchValue
#: On a link's reifier: the text in the document that was matched.
MATCHED_TEXT = ONTOCAST.matchedText

#: Link kinds, strongest evidence first; clusters join in this order.
LINK_KINDS: tuple[str, ...] = ("key", "ifp", "identifier", "label")

_STRING_DATATYPES = frozenset({None, XSD.string})


def identity_links_graph_uri(graph_uri: str) -> str:
    """The identity links graph of the facts graph ``graph_uri``."""
    return auxiliary_graph_uri(graph_uri, IDENTITY_LINKS_SUFFIX)


def match_node_iri(domain: str, payload: dict[str, object]) -> URIRef:
    """The match node for ``payload`` under ``domain``: the same IRI everywhere."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return URIRef(f"{domain.rstrip('/')}/match/{digest}")


def key_value(term: Node, catalog_terms: frozenset[str]) -> str | None:
    """The normalised form of ``term`` as a key value, or None when it cannot be one.

    A literal counts: strings trimmed and case-folded, typed values in the
    normal form triple stores converge on. An IRI counts only when the catalog
    holds it, because a document-local IRI means nothing in another document.
    """
    if isinstance(term, URIRef):
        return str(term) if str(term) in catalog_terms else None
    if not isinstance(term, Literal):
        return None
    literal = canonical_literal(term)
    assert isinstance(literal, Literal)
    lexical = str(literal).strip()
    if not lexical:
        return None
    if literal.datatype in _STRING_DATATYPES:
        return lexical.casefold()
    return f"{lexical}^^{literal.datatype}"


class _Builder:
    """Accumulates one document's links in a deterministic graph."""

    def __init__(self, domain: str) -> None:
        self.domain = domain
        self.graph = RDFGraph()
        self.graph.bind("skos", SKOS)
        self.graph.bind("ontocast", ONTOCAST)

    def link(
        self,
        entity: URIRef,
        payload: dict[str, object],
        *,
        kind: str,
        scope: Iterable[tuple[URIRef, str]],
        value: str,
        matched_text: str,
    ) -> None:
        node = match_node_iri(self.domain, payload)
        graph = self.graph
        graph.add((node, RDF.type, MATCH))
        graph.add((node, MATCH_KIND, Literal(kind)))
        for predicate, iri in scope:
            graph.add((node, predicate, URIRef(iri)))
        graph.add((node, MATCH_VALUE, Literal(value)))
        triple = (entity, SKOS.closeMatch, node)
        graph.add(triple)
        reifier = URIRef(
            f"{node}/r/{hashlib.sha256(str(entity).encode('utf-8')).hexdigest()[:16]}"
        )
        graph.reifications.add(reifier, triple)
        graph.add((reifier, MATCHED_TEXT, Literal(matched_text)))


def _types(graph: RDFGraph, node: URIRef) -> list[str]:
    return sorted(
        {str(t) for t in graph.objects(node, RDF.type) if isinstance(t, URIRef)}
    )


def linkable_instances(
    graph: RDFGraph,
    declarations: IdentityDeclarations,
    value_classes: frozenset[str],
) -> list[URIRef]:
    """The entities of a document graph that can carry links.

    A typed IRI subject that is not a catalog term, not a value (R1: values are
    reached through their entity, never resolved), and not the document node.
    """
    found: set[URIRef] = set()
    for subject in graph.subjects(RDF.type, None):
        if not isinstance(subject, URIRef) or subject in found:
            continue
        if str(subject) in declarations.catalog_terms:
            continue
        types = _types(graph, subject)
        if not types or FOAF.Document in map(URIRef, types):
            continue
        if any(t in value_classes for t in types):
            continue
        found.add(subject)
    return sorted(found, key=str)


def links_for(
    graph: RDFGraph,
    declarations: IdentityDeclarations,
    *,
    domain: str,
    value_classes: frozenset[str] = frozenset(),
) -> RDFGraph:
    """The identity links of one document's facts graph.

    Each entity gets one ``skos:closeMatch`` link per:

    - ``owl:hasKey`` that applies to its types and whose properties all have
      key values (one link per combination of values);
    - value of an ``owl:InverseFunctionalProperty``;
    - ``dcterms:identifier`` on an entity the document node links to (the
      caller's document metadata), scoped to each of its types;
    - normalised ``rdfs:label`` / ``skos:altLabel``, when one of its types is
      under a globally named class.

    Args:
        graph: One document's facts graph, as written to the store.
        declarations: The catalog's identity declarations.
        domain: The deployment domain; match nodes are minted under it.
        value_classes: Classes whose instances are values, never linked.

    Returns:
        RDFGraph: The links graph; empty when nothing is declared or matched.
    """
    builder = _Builder(domain)
    if not declarations.declares_anything and not _has_metadata_identifiers(graph):
        return builder.graph
    terms = declarations.catalog_terms
    metadata_entities = _metadata_entities(graph)
    for entity in linkable_instances(graph, declarations, value_classes):
        types = _types(graph, entity)
        _key_links(builder, graph, entity, types, declarations, terms)
        _ifp_links(builder, graph, entity, declarations, terms)
        if entity in metadata_entities:
            _identifier_links(builder, graph, entity, types, terms)
        _label_links(builder, graph, entity, types, declarations)
    return builder.graph


def _key_links(
    builder: _Builder,
    graph: RDFGraph,
    entity: URIRef,
    types: list[str],
    declarations: IdentityDeclarations,
    terms: frozenset[str],
) -> None:
    for cls_iri, props in declarations.keys_for(types):
        per_prop: list[list[tuple[str, str]]] = []
        for prop in props:
            values = sorted(
                {
                    (normalised, str(obj))
                    for obj in graph.objects(entity, URIRef(prop))
                    if (normalised := key_value(obj, terms)) is not None
                }
            )
            if not values:
                break
            per_prop.append(values)
        else:
            for combination in product(*per_prop):
                normalised = [value for value, _ in combination]
                builder.link(
                    entity,
                    {
                        "kind": "key",
                        "class": cls_iri,
                        "props": list(props),
                        "values": normalised,
                    },
                    kind="key",
                    scope=[(MATCH_CLASS, cls_iri)]
                    + [(MATCH_PROPERTY, p) for p in props],
                    value=json.dumps(normalised, separators=(",", ":")),
                    matched_text=" | ".join(raw for _, raw in combination),
                )


def _ifp_links(
    builder: _Builder,
    graph: RDFGraph,
    entity: URIRef,
    declarations: IdentityDeclarations,
    terms: frozenset[str],
) -> None:
    for prop in sorted(declarations.inverse_functional):
        for obj in graph.objects(entity, URIRef(prop)):
            normalised = key_value(obj, terms)
            if normalised is None:
                continue
            builder.link(
                entity,
                {"kind": "ifp", "property": prop, "value": normalised},
                kind="ifp",
                scope=[(MATCH_PROPERTY, prop)],
                value=normalised,
                matched_text=str(obj),
            )


def _identifier_links(
    builder: _Builder,
    graph: RDFGraph,
    entity: URIRef,
    types: list[str],
    terms: frozenset[str],
) -> None:
    for obj in graph.objects(entity, DCTERMS.identifier):
        normalised = key_value(obj, terms)
        if normalised is None:
            continue
        for type_iri in types:
            builder.link(
                entity,
                {"kind": "identifier", "class": type_iri, "value": normalised},
                kind="identifier",
                scope=[(MATCH_CLASS, type_iri)],
                value=normalised,
                matched_text=str(obj),
            )


def _label_links(
    builder: _Builder,
    graph: RDFGraph,
    entity: URIRef,
    types: list[str],
    declarations: IdentityDeclarations,
) -> None:
    declaring = sorted(declarations.declaring_classes(types))
    if not declaring:
        return
    names: dict[str, str] = {}
    for predicate in (RDFS.label, SKOS.altLabel):
        for label in graph.objects(entity, predicate):
            if not isinstance(label, Literal):
                continue
            normalised = normalize_string_value(str(label))
            if not normalised:
                continue
            raw = str(label)
            if normalised not in names or raw < names[normalised]:
                names[normalised] = raw
    for cls_iri in declaring:
        for normalised, raw in sorted(names.items()):
            builder.link(
                entity,
                {"kind": "label", "class": cls_iri, "value": normalised},
                kind="label",
                scope=[(MATCH_CLASS, cls_iri)],
                value=normalised,
                matched_text=raw,
            )


def _document_nodes(graph: RDFGraph) -> list[URIRef]:
    return sorted(
        (s for s in graph.subjects(RDF.type, FOAF.Document) if isinstance(s, URIRef)),
        key=str,
    )


def _metadata_entities(graph: RDFGraph) -> set[URIRef]:
    """Entities the document node links to: the caller's metadata entities."""
    found: set[URIRef] = set()
    for doc in _document_nodes(graph):
        for obj in graph.objects(doc, None):
            if isinstance(obj, URIRef) and obj != doc:
                found.add(obj)
    return found


def _has_metadata_identifiers(graph: RDFGraph) -> bool:
    return any(
        (entity, DCTERMS.identifier, None) in graph
        for entity in _metadata_entities(graph)
    )
