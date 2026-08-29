"""Domain-agnostic numeric-mention inventory for coverage checking.

Compares numbers stated in a source text against numeric literals present
in the extracted graph. The comparison is deliberately verbatim-oriented:
extraction is expected to transcribe source values exactly (units are
normalized downstream in code, never by the LLM), so a text number missing
from the graph is a candidate extraction gap.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation

from rdflib import Literal

from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)

# Numbers incl. decimals; exponents like 1.4e3; range/uncertainty separators
# are handled by extracting each side separately.
_NUMBER_PATTERN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![\w])")

# Bare 4-digit integers in this span are treated as publication years and left
# out of coverage checking, because citation/date noise otherwise dominates the
# advisory findings. The cost is real and accepted: a genuine bare quantity in
# this range (a 2000 K temperature, 1950 rpm) is exempted too. Only bare
# integers are affected -- anything with a decimal point, exponent, or unit
# attachment still counts. Disable per call with ``ignore_year_like=False``.
_YEAR_RANGE = (1900, 2100)


def canonical_number(text: str) -> str | None:
    """Return the canonical decimal form of a numeric string, or None."""
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return format(value.normalize(), "f")


#: Characters that join a digit group to the rest of an identifier rather than
#: to a magnitude. Deliberately narrow. A decimal point is excluded because the
#: number pattern already consumes "3.14" as one match, and a hyphen is
#: excluded because "10-15 meV" is a range whose sides are both real values --
#: the module reads each side separately by design.
_IDENTIFIER_ADJACENT = frozenset("/:")


def _is_identifier_fragment(text: str, start: int, end: int) -> bool:
    """Whether the digit group at ``text[start:end]`` belongs to an identifier.

    The signal is adjacency to an identifier separator: "600/92" is one file
    number, "10.1234/example" one DOI. A magnitude with its unit ("8.5 nm") is
    untouched, and so is a range.

    Args:
        text: The text the match came from.
        start: Match start offset.
        end: Match end offset.

    Returns:
        True when the digit group sits against an identifier separator.
    """
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return before in _IDENTIFIER_ADJACENT or after in _IDENTIFIER_ADJACENT


def extract_numeric_tokens(
    text: str,
    *,
    ignore_year_like: bool = True,
    ignore_identifier_fragments: bool = False,
) -> set[str]:
    """Extract canonical numeric tokens from free text.

    Args:
        text: Source text.
        ignore_year_like: Drop bare integers in the 1900-2100 range (years,
            citation artifacts). Values that also occur with a decimal point
            are kept.
        ignore_identifier_fragments: Drop digit groups sitting against an
            identifier separator -- see :func:`_is_identifier_fragment`. A
            digit group standing alone as its own token is *not* covered:
            nothing around it distinguishes a file-number component from a
            small quantity, and guessing there would cost real values.

    Returns:
        Set of canonical decimal strings.
    """
    tokens: set[str] = set()
    for match in _NUMBER_PATTERN.finditer(text):
        raw = match.group(1)
        if ignore_identifier_fragments and _is_identifier_fragment(
            text, match.start(1), match.end(1)
        ):
            continue
        canonical = canonical_number(raw)
        if canonical is None:
            continue
        if (
            ignore_year_like
            and "." not in raw
            and "e" not in raw.lower()
            and _YEAR_RANGE[0] <= int(canonical.split(".")[0] or 0) <= _YEAR_RANGE[1]
            and canonical.isdigit()
        ):
            continue
        tokens.add(canonical)
    return tokens


def numeric_literals_in_graph(graph: RDFGraph) -> set[str]:
    """Collect canonical numeric values appearing anywhere in graph literals.

    Numbers inside label/comment strings count as present: coverage findings
    target numbers missing from the graph entirely, while structuring
    label-only numbers is the critic's judgement call.
    """
    values: set[str] = set()
    for _, _, obj in graph:
        if not isinstance(obj, Literal):
            continue
        text = str(obj)
        canonical = canonical_number(text.strip())
        if canonical is not None:
            values.add(canonical)
            continue
        for match in _NUMBER_PATTERN.finditer(text):
            canonical = canonical_number(match.group(1))
            if canonical is not None:
                values.add(canonical)
    return values


def missing_numeric_mentions(
    text: str,
    graph: RDFGraph,
    *,
    ignore_year_like: bool = True,
    ignore_identifier_fragments: bool = False,
    limit: int = 30,
) -> list[str]:
    """Return canonical numbers stated in text but absent from the graph.

    The result is capped at ``limit``; when it truncates, a warning records how
    many mentions were dropped. Ordering is shortest-first, which is a stable
    presentation order, not a relevance ranking -- the caller is advisory
    telemetry, so the cap bounds prompt size rather than selecting the most
    important gaps.

    Args:
        text: Source text for the unit.
        graph: Graph extracted from that text.
        ignore_year_like: Drop bare integers in the publication-year span.
        ignore_identifier_fragments: Drop digit groups that are parts of an
            identifier. Offering them invites the repair render to structure a
            file number or a citation into numeric properties, which the
            downstream multi-value check then flags.
        limit: Maximum mentions returned.

    Returns:
        Canonical decimal strings, shortest-first, capped at ``limit``.
    """
    missing = extract_numeric_tokens(
        text,
        ignore_year_like=ignore_year_like,
        ignore_identifier_fragments=ignore_identifier_fragments,
    ) - numeric_literals_in_graph(graph)
    ordered = sorted(missing, key=lambda value: (len(value), value))
    if len(ordered) > limit:
        logger.warning(
            "Numeric coverage: %d missing mention(s) truncated to %d for the "
            "repair prompt",
            len(ordered),
            limit,
        )
    return ordered[:limit]
