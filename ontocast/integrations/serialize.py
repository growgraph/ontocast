"""Render OntoCast return values as text an LLM can read.

Most of the interesting return types -- :class:`~ontocast.onto.rdfgraph.RDFGraph`,
:class:`~ontocast.onto.ontology.Ontology`, ``DoclingDocument`` -- are not JSON
serializable, so every wrapped tool needs an explicit projection to text. These
helpers provide it, and enforce one rule the naive version gets wrong:
truncation must be announced. Cutting Turtle at a byte offset produces text that
*looks* like a complete graph but parses as a syntax error, and a model given
silently-truncated context will confidently reason about triples that were
dropped.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from ontocast.onto.rdfgraph import RDFGraph

#: Appended whenever output is cut, so the model knows the view is partial.
_TRUNCATION_NOTE = (
    "\n# ... TRUNCATED at {limit} chars ({total} triples total). "
    "Narrow your query, add a filter, or lower top_k to see the rest."
)


def graph_to_llm_text(
    graph: RDFGraph,
    *,
    max_chars: int,
    sources: Sequence[str] | None = None,
) -> str:
    """Serialize a graph to Turtle for an LLM, with an explicit truncation marker.

    Args:
        graph: The graph to render.
        max_chars: Maximum characters to emit before truncating.
        sources: Optional source IRIs, emitted as a leading comment so the model
            can cite where the triples came from.

    Returns:
        Canonical Turtle, prefixed by a ``# sources:`` header when sources are
        given and suffixed by a truncation note when the budget is exceeded.
    """
    header = f"# sources: {', '.join(sources)}\n" if sources else ""
    if len(graph) == 0:
        return header + "# (no triples matched)"
    ttl = graph.serialize_canonical_turtle()
    if len(ttl) <= max_chars:
        return header + ttl
    return (
        header
        + ttl[:max_chars]
        + _TRUNCATION_NOTE.format(limit=max_chars, total=len(graph))
    )


def json_to_llm_text(payload: Any, *, max_chars: int) -> str:
    """Render a JSON-serializable payload, truncating with a marker if needed."""
    text = json.dumps(payload, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... TRUNCATED at {max_chars} chars."


def models_to_llm_text(
    items: Sequence[BaseModel],
    *,
    max_chars: int,
    fields: Sequence[str] | None = None,
) -> str:
    """Render pydantic results as a JSON list, dropping whole items when over budget.

    Truncating by item rather than by character keeps the output valid JSON,
    which matters more here than showing a partial final record.

    Args:
        items: The models to render.
        max_chars: Character budget for the rendered list.
        fields: Optional field allowlist; other fields are omitted. Use this to
            keep embedding vectors and other model-illegible payloads out.

    Returns:
        A JSON array, with a trailing note when items were dropped.
    """
    rows = [
        item.model_dump(mode="json", include=set(fields) if fields else None)
        for item in items
    ]
    text = json.dumps(rows, indent=2, default=str)
    if len(text) <= max_chars:
        return text

    kept: list[dict[str, Any]] = []
    for row in rows:
        candidate = kept + [row]
        if len(json.dumps(candidate, indent=2, default=str)) > max_chars:
            break
        kept.append(row)
    return (
        json.dumps(kept, indent=2, default=str)
        + f"\n// showing {len(kept)} of {len(rows)} results ({max_chars}-char budget)."
    )


def truncate(text: str, *, max_chars: int) -> str:
    """Cut plain text to a budget, announcing the cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... TRUNCATED at {max_chars} of {len(text)} chars."
