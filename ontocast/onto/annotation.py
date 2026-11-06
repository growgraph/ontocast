"""Annotation predicates: literals that describe a node rather than assert a value."""

from rdflib import RDFS, SKOS
from rdflib.namespace import DCTERMS
from rdflib.term import Node

ANNOTATION_PREDICATES = frozenset({RDFS.label, RDFS.comment, DCTERMS.description})
_SKOS_NAMESPACE = str(SKOS)


def is_annotation(predicate: Node) -> bool:
    """True for a label, comment, description or SKOS term: a number inside one
    is prose, not data, and the triple states nothing about the node."""
    return predicate in ANNOTATION_PREDICATES or str(predicate).startswith(
        _SKOS_NAMESPACE
    )
