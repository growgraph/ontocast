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


def role_from_predicate_usage(*, is_predicate: bool) -> str:
    """Map predicate-position usage to vector-store role vocabulary."""
    return ROLE_PREDICATE if is_predicate else ROLE_RESOURCE
