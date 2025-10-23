import hashlib

from rdflib import Graph
from rdflib.namespace import NamespaceManager


def iri2namespace(iri: str, ontology: bool = False) -> str:
    """Convert an IRI to a namespace string.

    Args:
        iri: The IRI to convert.
        ontology: If True, append '#' for ontology namespace, otherwise '/'.

    Returns:
        str: The converted namespace string.
    """
    iri = iri.rstrip("#")
    return f"{iri}#" if ontology else f"{iri}/"


def get_rdflib_namespace_mappings() -> dict:
    g = Graph()
    ns_manager = NamespaceManager(g)
    return {str(uri): prefix for prefix, uri in ns_manager.namespaces()}


CONVENTIONAL_MAPPINGS = get_rdflib_namespace_mappings()


def render_text_hash(text: str, digits=12) -> str:
    """
    Generate a hash for the given text.

    Args:
        text: The text to hash
        digits: Number of digits in the hash (default: 12)

    Returns:
        A string hash of the text
    """
    return hashlib.sha256(text.encode()).hexdigest()[:digits]
