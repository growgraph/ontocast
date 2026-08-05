"""Literal/object signatures and schema harvesting for merge guards.

Merge guards need cheap, canonical views of what an entity *asserts*:
which literal values it holds per predicate, and which IRI objects it
points at per predicate. Two entities asserting conflicting values for
the same predicate are distinct individuals no matter how similar their
labels are — a false merge silently corrupts data, while a false split
leaves visible, recoverable redundancy.

Everything here is domain-agnostic: functionality is harvested from the
ontology context (``owl:FunctionalProperty`` and OWL max-cardinality-1
restrictions) or inferred empirically from the corpus, never hardcoded
per vocabulary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from rdflib import OWL, RDF, Literal, URIRef
from rdflib.namespace import XSD

from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)

# Bounds the quadratic sibling-pair term per object group.
SIBLING_GROUP_CAP = 32

_NUMERIC_DATATYPES = {
    XSD.decimal,
    XSD.integer,
    XSD.int,
    XSD.long,
    XSD.short,
    XSD.byte,
    XSD.float,
    XSD.double,
    XSD.nonNegativeInteger,
    XSD.positiveInteger,
    XSD.nonPositiveInteger,
    XSD.negativeInteger,
    XSD.unsignedInt,
    XSD.unsignedLong,
    XSD.unsignedShort,
    XSD.unsignedByte,
}

_TEMPORAL_DATATYPES = {
    XSD.date,
    XSD.dateTime,
    XSD.dateTimeStamp,
    XSD.time,
    XSD.gYear,
    XSD.gYearMonth,
}

_MAX_ONE_CARDINALITY_PREDICATES = (
    OWL.maxCardinality,
    OWL.cardinality,
    OWL.maxQualifiedCardinality,
    OWL.qualifiedCardinality,
)


def canonical_literal(literal: Literal) -> tuple[str, str] | None:
    """Return a canonical ``(value, kind)`` pair for a guard-relevant literal.

    Numeric literals (typed with an XSD numeric datatype, or untyped with a
    numeric lexical form) canonicalize through :class:`~decimal.Decimal` so
    ``230``, ``"230"^^xsd:decimal`` and ``"230.0"^^xsd:double`` compare equal.
    Temporal literals canonicalize to their lexical form. Strings return
    ``None`` — string conflicts are handled by the strict lexical bar, not by
    value comparison.

    Args:
        literal: Literal to canonicalize.

    Returns:
        ``(canonical_value, kind)`` with kind in ``{"numeric", "temporal"}``,
        or ``None`` when the literal is not guard-relevant.
    """
    datatype = literal.datatype
    if datatype in _TEMPORAL_DATATYPES:
        return (str(literal).strip(), "temporal")
    if datatype in _NUMERIC_DATATYPES or datatype is None:
        try:
            value = Decimal(str(literal).strip())
        except (InvalidOperation, ValueError):
            return None
        normalized = format(value.normalize(), "f")
        return (normalized, "numeric")
    return None


def harvest_max_one_predicates(ontology_graph: RDFGraph | None) -> set[URIRef]:
    """Harvest predicates the schema constrains to at most one value.

    Sources, both fully generic:

    - subjects typed ``owl:FunctionalProperty``;
    - ``owl:onProperty`` targets of OWL restrictions carrying
      ``owl:maxCardinality`` / ``owl:cardinality`` (or their qualified
      variants) equal to 1.

    Args:
        ontology_graph: Merged ontology context, or None.

    Returns:
        Set of predicate IRIs with a schema-asserted max-1 constraint.
    """
    functional: set[URIRef] = set()
    if ontology_graph is None:
        return functional
    for subject in ontology_graph.subjects(RDF.type, OWL.FunctionalProperty):
        if isinstance(subject, URIRef):
            functional.add(subject)
    for cardinality_predicate in _MAX_ONE_CARDINALITY_PREDICATES:
        for restriction, cardinality in ontology_graph.subject_objects(
            cardinality_predicate
        ):
            if not isinstance(cardinality, Literal):
                continue
            try:
                if int(cardinality) != 1:
                    continue
            except (TypeError, ValueError):
                continue
            for on_property in ontology_graph.objects(restriction, OWL.onProperty):
                if isinstance(on_property, URIRef):
                    functional.add(on_property)
    return functional


@dataclass
class MergeGuardContext:
    """Corpus- and schema-derived context consulted by merge guards.

    Attributes:
        sibling_pairs: Pairs of entities that co-occur as objects of one
            subject (scope-dependent) and therefore denote distinct
            individuals.
        functional_predicates: Predicates with a schema-asserted or
            empirically observed max-1 object constraint.
    """

    sibling_pairs: set[frozenset[URIRef]] = field(default_factory=set)
    functional_predicates: set[URIRef] = field(default_factory=set)


def build_sibling_pairs(
    object_groups: dict[tuple[URIRef, URIRef], set[URIRef]],
    *,
    scope: str,
) -> set[frozenset[URIRef]]:
    """Build never-merge pairs from co-object groups.

    Two URIs listed as objects on one subject describe distinct things —
    merging the endpoints of a range, the samples of areas 1–3, or the
    grants of one acknowledgement destroys exactly the distinction the
    author asserted.

    Args:
        object_groups: ``(subject, predicate) -> URIRef objects`` observed
            across the unit corpus.
        scope: ``"subject"`` pairs all objects of one subject regardless of
            predicate; ``"predicate"`` restricts to objects sharing the same
            predicate.

    Returns:
        Set of unordered entity pairs that must never merge.
    """
    pairs: set[frozenset[URIRef]] = set()
    if scope == "predicate":
        grouped: dict[tuple[URIRef, URIRef], set[URIRef]] = object_groups
    else:
        merged: dict[URIRef, set[URIRef]] = {}
        for (subject, _), objects in object_groups.items():
            merged.setdefault(subject, set()).update(objects)
        grouped = {(subject, subject): objects for subject, objects in merged.items()}
    truncated_groups = 0
    dropped_objects = 0
    for objects in grouped.values():
        if len(objects) < 2:
            continue
        if len(objects) > SIBLING_GROUP_CAP:
            truncated_groups += 1
            dropped_objects += len(objects) - SIBLING_GROUP_CAP
        bounded = sorted(objects, key=str)[:SIBLING_GROUP_CAP]
        for index, left in enumerate(bounded):
            for right in bounded[index + 1 :]:
                pairs.add(frozenset((left, right)))
    if truncated_groups:
        # Past the cap the guard silently stops protecting the tail, and which
        # members survive is decided by IRI byte order -- arbitrary with respect
        # to merge risk. Bounded work, but never a silent bound.
        logger.warning(
            "Sibling merge guard truncated %d group(s) at SIBLING_GROUP_CAP=%d, "
            "leaving %d co-object(s) unguarded (selection is by IRI order)",
            truncated_groups,
            SIBLING_GROUP_CAP,
            dropped_objects,
        )
    return pairs


def empirically_functional_predicates(
    object_groups: dict[tuple[URIRef, URIRef], set[URIRef]],
    *,
    min_support: int,
) -> set[URIRef]:
    """Infer predicates that behave single-valued across the corpus.

    A predicate qualifies when it is observed on at least ``min_support``
    subjects and no subject anywhere holds two distinct IRI objects for it
    (e.g. ``qudt:unit``: every quantity node carries exactly one unit, even
    though no catalog declares the property functional). Multi-valued domain
    predicates are exempt by construction.

    Args:
        object_groups: ``(subject, predicate) -> URIRef objects``.
        min_support: Minimum distinct subjects required before the inference
            is trusted.

    Returns:
        Set of empirically single-valued predicate IRIs.
    """
    subject_counts: dict[URIRef, int] = {}
    multi_valued: set[URIRef] = set()
    for (_, predicate), objects in object_groups.items():
        subject_counts[predicate] = subject_counts.get(predicate, 0) + 1
        if len(objects) > 1:
            multi_valued.add(predicate)
    return {
        predicate
        for predicate, count in subject_counts.items()
        if count >= min_support and predicate not in multi_valued
    }
