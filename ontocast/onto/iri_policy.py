from __future__ import annotations

import re
from typing import Literal

URIContext = Literal["ontology", "facts", "auto"]

_ONTOLOGY_HINT_PATTERN = re.compile(
    r"(?:^|[/#:_-])(onto|ontology|owl|rdfs?|skos)(?:$|[/#:_-])",
    re.IGNORECASE,
)


def strip_iri_brackets(value: str) -> str:
    """Strip optional Turtle-style ``<...>`` wrappers from an IRI string."""
    text = value.strip()
    if text.startswith("<") and text.endswith(">") and len(text) > 2:
        return text[1:-1].strip()
    return text


# Codepoints the SPARQL ``IRIREF`` production forbids inside ``<...>``.
_IRIREF_FORBIDDEN = frozenset('<>"{}|^`\\')


def as_sparql_iriref(value: str) -> str | None:
    """Render an IRI for a SPARQL ``IRIREF`` slot, or ``None`` if it cannot be one.

    Callers should log-and-skip on ``None`` rather than raising: a single
    malformed IRI must not fail a whole query.

    Args:
        value: IRI text, optionally wrapped in Turtle-style angle brackets.

    Returns:
        str | None: ``<iri>`` ready for interpolation, or ``None`` when ``value``
        is empty or contains a codepoint ``IRIREF`` forbids.
    """
    text = strip_iri_brackets(value).strip()
    if not text:
        return None
    if any(char in _IRIREF_FORBIDDEN or ord(char) <= 0x20 for char in text):
        return None
    return f"<{text}>"


def _resolve_context(namespace: str, context: URIContext) -> URIContext:
    if context != "auto":
        return context
    lowered = namespace.lower()
    if _ONTOLOGY_HINT_PATTERN.search(namespace) or lowered.endswith("ont"):
        return "ontology"
    return "facts"


# Known aliases of one canonical namespace. Both spellings appear in the wild
# (LLM output, upstream ontologies); leaving them split scatters the same terms
# across two IRIs and silently breaks dedup and SPARQL.
_NAMESPACE_ALIASES: dict[str, str] = {
    "http://schema.org/": "https://schema.org/",
    "http://schema.org#": "https://schema.org/",
    "https://schema.org#": "https://schema.org/",
}


def canonicalize_namespace_iri(namespace: str) -> str:
    """Map a delimiter-normalized namespace IRI to its canonical alias."""
    return _NAMESPACE_ALIASES.get(namespace, namespace)


def normalize_namespace_iri(namespace: str, *, context: URIContext = "auto") -> str:
    """Return a namespace IRI with a deterministic terminal delimiter.

    Existing trailing ``#`` or ``/`` are preserved. When absent, we append ``#``
    for ontology contexts and ``/`` for facts/default contexts. Known namespace
    aliases (e.g. ``http://schema.org/``) are mapped to their canonical form.
    """
    text = strip_iri_brackets(namespace)
    if text.endswith("#") or text.endswith("/"):
        return canonicalize_namespace_iri(text)
    resolved_context = _resolve_context(text, context)
    suffix = "#" if resolved_context == "ontology" else "/"
    return canonicalize_namespace_iri(f"{text}{suffix}")


def join_namespace_local(
    namespace: str,
    local: str,
    *,
    context: URIContext = "auto",
) -> str:
    return f"{normalize_namespace_iri(namespace, context=context)}{local}"


def split_namespace_local(value: str) -> tuple[str | None, str]:
    text = value.strip()
    if not text:
        return None, ""
    if "#" in text:
        namespace, local = text.rsplit("#", 1)
        return f"{namespace}#", local
    trimmed = text.rstrip("/")
    if "/" in trimmed:
        namespace, local = trimmed.rsplit("/", 1)
        return f"{namespace}/", local
    return None, text


def is_in_namespace(uri: str, namespace: str, *, context: URIContext = "auto") -> bool:
    normalized_namespace = normalize_namespace_iri(namespace, context=context)
    return uri.startswith(normalized_namespace)


def sanitize_prefix_map(
    prefix_map: dict[str, str],
    *,
    context: URIContext = "auto",
) -> dict[str, str]:
    return {
        prefix: normalize_namespace_iri(namespace, context=context)
        for prefix, namespace in prefix_map.items()
    }
