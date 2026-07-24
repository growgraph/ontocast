import re
from urllib.parse import urlparse

from rdflib import Graph
from rdflib.namespace import NamespaceManager

from ontocast.onto.iri_policy import strip_iri_brackets


def normalize_ontology_iri(iri: str) -> str:
    """Normalize an ontology IRI for catalog lookup and alias resolution.

    Strips angle brackets and trailing ``/`` or ``#`` so that
    ``http://example.org/ont``, ``…/ont/``, and ``…/ont#`` compare equal as
    lookup keys. Author namespace delimiters for prefix bindings remain the
    responsibility of :func:`normalize_namespace_iri`.
    """
    text = strip_iri_brackets(iri)
    return text.rstrip("/#")


def _conventional_prefix_for(iri: str) -> str | None:
    """Return rdflib's conventional prefix for *iri*, trying delimiter variants."""
    candidates = [
        iri,
        iri.rstrip("/#"),
        f"{iri.rstrip('/#')}#",
        f"{iri.rstrip('/#')}/",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in CONVENTIONAL_MAPPINGS:
            return CONVENTIONAL_MAPPINGS[candidate]
    return None


def derive_ontology_id(iri: str) -> str | None:
    """Derive a short ontology handle from an absolute IRI.

    Prefers rdflib conventional namespace prefixes (SKOS → ``skos``). Falls back
    to the last non-opaque path segment. Returns ``None`` instead of inventing
    garbage ids (e.g. FOAF ``0.1`` → would-be ``01``).
    """
    if not isinstance(iri, str) or not iri.strip():
        return None

    raw = strip_iri_brackets(iri.strip())
    conventional = _conventional_prefix_for(raw)
    if conventional:
        return conventional

    normalized_iri = normalize_ontology_iri(raw)
    parsed = urlparse(normalized_iri)

    candidate = (
        parsed.path.rsplit("/", 1)[-1]
        if parsed.path and "/" in parsed.path
        else parsed.netloc.split(".")[0]
        if parsed.netloc
        else normalized_iri
    )

    return _clean_derived_id(candidate)


def _clean_derived_id(value: str) -> str | None:
    value = re.sub(r"\.(owl|ttl|rdf|xml)$", "", value, flags=re.IGNORECASE)
    match = re.match(r"^(.*?)\.(org|com|net|io|edu|gov|int|mil)$", value, re.IGNORECASE)
    if match:
        value = match.group(1)
    result = re.sub(r"[^a-zA-Z0-9_-]", "", value).lower()
    if not result:
        return None
    # Refuse pure-numeric or version-like tails (e.g. FOAF ``0.1`` → ``01``).
    if result.isdigit():
        return None
    if len(result) < 2:
        return None
    return result


def get_rdflib_namespace_mappings() -> dict[str, str]:
    g = Graph()
    ns_manager = NamespaceManager(g)
    return {str(uri): prefix for prefix, uri in ns_manager.namespaces()}


CONVENTIONAL_MAPPINGS = get_rdflib_namespace_mappings()
# Namespace URIs that rdflib binds by default (xml, brick, csvw, …).
RDFLIB_DEFAULT_NAMESPACE_URIS: frozenset[str] = frozenset(CONVENTIONAL_MAPPINGS)


def is_rdflib_default_namespace(namespace: str) -> bool:
    """Return True when *namespace* is one of rdflib's built-in bindings."""
    return namespace in RDFLIB_DEFAULT_NAMESPACE_URIS
