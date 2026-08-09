"""Shared text normalization and deterministic triple rendering helpers."""

from __future__ import annotations

import re
import unicodedata

from rdflib import BNode, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.iri_policy import split_namespace_local

ROLE_RESOURCE = "resource"
ROLE_PREDICATE = "predicate"


def normalize_text(text: str) -> str:
    """Normalize free text for embedding and matching."""
    text_no_diacritics = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    normalized = text_no_diacritics.replace("_", " ").replace("-", " ").strip().lower()
    return re.sub(r"\s+", " ", normalized)


def normalize_identifier(text: str) -> str:
    """Normalize identifier-like text with camel/snake/kebab splitting."""
    with_boundaries = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    with_boundaries = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", with_boundaries)
    return normalize_text(with_boundaries)


def normalize_uri_local_name(uri: URIRef) -> str:
    """Normalize the local part of a URI."""
    _, local = split_namespace_local(str(uri))
    return normalize_identifier(local)


def render_term_for_text(term: Node) -> str:
    """Render a graph term into deterministic text."""
    if isinstance(term, URIRef):
        return normalize_uri_local_name(term)
    if isinstance(term, Literal):
        return normalize_text(str(term))
    if isinstance(term, BNode):
        return "blank node"
    return normalize_text(str(term))


def stable_sorted_triples(
    triples: list[tuple[Node, Node, Node]],
) -> list[tuple[Node, Node, Node]]:
    """Return a deterministic ordering of triples."""
    return sorted(triples, key=lambda triple: str(triple))


def role_from_declaration(*, is_declared_property: bool, is_predicate: bool) -> str:
    """Map property declaration *or* predicate-position usage to role vocabulary.

    Predicate-position usage alone under-reports badly: a TBox-only ontology
    asserts its properties as subjects (``ex:hasResult a owl:ObjectProperty ;
    rdfs:domain … ; rdfs:range …``) and never uses them as predicates, so a
    catalog of pure schema modules classifies nearly every property as a
    resource. Those atoms then get a resource-shaped neighborhood
    representation — empty, for a term with no outgoing domain assertions —
    and the neighborhood channel goes blind to the predicates that carry the
    graph structure.

    Args:
        is_declared_property: Entity is declared an OWL/RDF property.
        is_predicate: Entity occurs in predicate position in the source graph.

    Returns:
        str: ``ROLE_PREDICATE`` when either signal holds, else ``ROLE_RESOURCE``.
    """
    return ROLE_PREDICATE if (is_declared_property or is_predicate) else ROLE_RESOURCE
