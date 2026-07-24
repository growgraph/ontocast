"""Helpers for merging RDF namespace prefix bindings without silent loss."""

from __future__ import annotations


def merge_namespace_bindings(
    existing: dict[str, str],
    incoming: dict[str, str],
) -> dict[str, str]:
    """Merge prefix→namespace maps with conflict rename (never silent override).

    Rules:
    - Same prefix, same namespace → keep.
    - Same prefix, different namespace → keep existing; rename incoming to
      ``{prefix}_{n}`` until unique.
    - New prefix → add.
    """
    result: dict[str, str] = {
        prefix: str(namespace) for prefix, namespace in existing.items() if prefix
    }
    for prefix, namespace in incoming.items():
        if not prefix:
            continue
        ns = str(namespace)
        if prefix not in result:
            result[prefix] = ns
            continue
        if result[prefix] == ns:
            continue
        n = len(result)
        new_prefix = f"{prefix}_{n}"
        while new_prefix in result:
            n += 1
            new_prefix = f"{prefix}_{n}"
        result[new_prefix] = ns
    return result


def choose_best_prefix(
    namespace: str,
    prefixes: list[str],
    preferred_namespace_prefixes: dict[str, str] | None = None,
) -> str:
    """Pick one prefix for *namespace*, preferring a catalog author prefix."""
    if not prefixes:
        raise ValueError("prefixes must be non-empty")
    preferred = None
    if preferred_namespace_prefixes:
        preferred = preferred_namespace_prefixes.get(namespace)
        if preferred is None:
            # Also try without trailing delimiter variance.
            preferred = preferred_namespace_prefixes.get(namespace.rstrip("/#"))
            if preferred is None:
                for key, value in preferred_namespace_prefixes.items():
                    if key.rstrip("/#") == namespace.rstrip("/#"):
                        preferred = value
                        break
        if preferred is not None and preferred in prefixes:
            return preferred
    return sorted(prefixes, key=lambda p: (len(p), p))[0]
