"""Helpers for deriving prompt-ready domain ontology namespace context."""

from ontocast.onto.constants import COMMON_PREFIXES, DEFAULT_IRI
from ontocast.onto.ontology import Ontology

_STANDARD_NAMESPACES: frozenset[str] = frozenset(
    uri.strip("<>") for uri in COMMON_PREFIXES.values()
) | {"https://schema.org/", DEFAULT_IRI}


def extract_domain_prefix_pairs(ontology: Ontology) -> list[tuple[str, str]]:
    """Return domain prefix/namespace pairs present in ontology graph."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for prefix, namespace_uri in ontology.graph.namespaces():
        if not prefix:
            continue
        namespace = str(namespace_uri)
        if namespace in _STANDARD_NAMESPACES:
            continue
        pair = (prefix, namespace)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)

    if pairs:
        return pairs

    if ontology.prefix and ontology.namespace:
        return [(ontology.prefix, ontology.namespace)]
    return []


def format_ontologies_clause(pairs: list[tuple[str, str]]) -> str:
    """Format a human-readable ontology namespace clause for prompts."""
    namespaces = [f"<{namespace}>" for _, namespace in pairs]
    if not namespaces:
        return "domain ontology namespaces declared in the provided ontology graph"
    if len(namespaces) == 1:
        return f"domain ontology {namespaces[0]}"
    return f"domain ontologies {', '.join(namespaces)}"


def format_prefix_clause(pairs: list[tuple[str, str]]) -> str:
    """Format a human-readable prefix clause for prompts."""
    prefixes = [f"`{prefix}:`" for prefix, _ in pairs]
    if not prefixes:
        return "the declared domain ontology prefixes in the provided ontology graph"
    if len(prefixes) == 1:
        return f"the prefix {prefixes[0]}"
    return f"their respective prefixes {', '.join(prefixes)}"
