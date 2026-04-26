from __future__ import annotations

import re
from typing import Literal

URIContext = Literal["ontology", "facts", "auto"]

_ONTOLOGY_HINT_PATTERN = re.compile(
    r"(?:^|[/#:_-])(onto|ontology|owl|rdfs?|skos)(?:$|[/#:_-])",
    re.IGNORECASE,
)


def _strip_brackets(value: str) -> str:
    text = value.strip()
    if text.startswith("<") and text.endswith(">") and len(text) > 2:
        return text[1:-1].strip()
    return text


def _resolve_context(namespace: str, context: URIContext) -> URIContext:
    if context != "auto":
        return context
    lowered = namespace.lower()
    if _ONTOLOGY_HINT_PATTERN.search(namespace) or lowered.endswith("ont"):
        return "ontology"
    return "facts"


def normalize_namespace_iri(namespace: str, *, context: URIContext = "auto") -> str:
    """Return a namespace IRI with a deterministic terminal delimiter.

    Existing trailing ``#`` or ``/`` are preserved. When absent, we append ``#``
    for ontology contexts and ``/`` for facts/default contexts.
    """
    text = _strip_brackets(namespace)
    if text.endswith("#") or text.endswith("/"):
        return text
    resolved_context = _resolve_context(text, context)
    suffix = "#" if resolved_context == "ontology" else "/"
    return f"{text}{suffix}"


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
