"""Which multi-valued facts of a merged graph an identity merge produced.

A source subject is a subject as one unit wrote it. Values or links that a
single source subject already held were together before any identity decision,
so they cannot be the residue of a bad merge, and the un-merge repair, which can
only undo identity decisions, must not act on them.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from rdflib import URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.model import MergeProvenance
from ontocast.tool.agg.signatures import object_value_key

#: A source subject: the index of the unit that wrote it, and its IRI there.
_Source = tuple[int, str]


def _joined(held: Collection[set[str]]) -> bool:
    """True when the sources hold two or more keys and no one source holds them all."""
    union = set().union(*held)
    return len(union) >= 2 and not any(keys == union for keys in held)


def build_merge_provenance(
    units: Sequence[ContentUnit],
    final_mapping: dict[URIRef, URIRef],
) -> MergeProvenance:
    """Record which multi-valued facts of the merged graph a merge produced.

    Values are compared by :func:`object_value_key`, as the validation gate
    compares them. Keys are canonicalized through ``final_mapping``, because
    the gate sees the post-merge graph and a pre-merge IRI would never match.

    Args:
        units: The content units that contributed to the merged graph.
        final_mapping: Source entity -> final URI, as used for the rewrite.

    Returns:
        The (subject, predicate) pairs with at least two distinct values, and
        the (subject, object) pairs linked by at least two predicates, that no
        single source subject accounts for.
    """
    values: dict[tuple[str, str], dict[_Source, set[str]]] = {}
    links: dict[tuple[str, str], dict[tuple[int, str, str], set[str]]] = {}
    for index, unit in enumerate(units):
        if unit.graph is None:
            continue
        for subject, predicate, obj in unit.graph:
            if not isinstance(subject, URIRef) or not isinstance(predicate, URIRef):
                continue
            final_subject = str(final_mapping.get(subject, subject))
            source = (index, str(subject))
            value = final_mapping.get(obj, obj) if isinstance(obj, URIRef) else obj
            values.setdefault((final_subject, str(predicate)), {}).setdefault(
                source, set()
            ).add(object_value_key(value))
            if isinstance(obj, URIRef):
                link = (final_subject, str(final_mapping.get(obj, obj)))
                links.setdefault(link, {}).setdefault(
                    (index, str(subject), str(obj)), set()
                ).add(str(predicate))

    return MergeProvenance(
        joined_values=sorted(
            key for key, held in values.items() if _joined(held.values())
        ),
        joined_links=sorted(
            key for key, held in links.items() if _joined(held.values())
        ),
    )
