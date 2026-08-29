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
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from rdflib import OWL, RDF, Literal, URIRef
from rdflib.namespace import XSD

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.representation_text import normalize_text

logger = logging.getLogger(__name__)

_CAMEL_SPLIT_RE = re.compile(r"(?=[A-Z][a-z])")


def normalize_string_value(text: str) -> str:
    """Normalize a string for identity comparison.

    Lowercase, diacritics removed, special characters cleaned, CamelCase
    split so it yields the same tokens as snake_case. Single source for the
    normalizer and the validation gate, which must agree on what counts as
    "the same string".
    """
    return normalize_text(_CAMEL_SPLIT_RE.sub(" ", text))


_TOKEN_EDGE_PUNCTUATION_RE = re.compile(r"^\W+|\W+$")


def clean_label_token(token: str) -> str:
    """Strip punctuation from token edges: "baranov," / "d." -> "baranov" / "d".

    ``normalize_text`` deliberately keeps punctuation, so token-level
    comparisons must shed it themselves — an initial written "D." is two
    characters lexically and one character semantically.
    """
    return _TOKEN_EDGE_PUNCTUATION_RE.sub("", token)


def label_tokens(label: str) -> list[str]:
    """Split a normalized label into punctuation-cleaned tokens."""
    return [cleaned for token in label.split() if (cleaned := clean_label_token(token))]


def tokens_alias_compatible(left: str, right: str) -> bool:
    """Exact token match, or a (possibly dotted) single-char initial of it."""
    if left == right:
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    return len(shorter) == 1 and longer.startswith(shorter)


def labels_alias_with_initials(
    left_labels: set[str],
    right_labels: set[str],
) -> bool:
    """True when a label pair matches token-injectively allowing initials.

    Every token of the shorter label must match a distinct token of the
    longer one (exactly, or as a single-character initial), and at least
    one matched token must be a full word (len > 2). Generic abbreviation
    structure — nothing person-specific.
    """
    for left_label in left_labels:
        left_tokens = label_tokens(left_label)
        for right_label in right_labels:
            right_tokens = label_tokens(right_label)
            if not left_tokens or not right_tokens:
                continue
            shorter, longer = (
                (left_tokens, right_tokens)
                if len(left_tokens) <= len(right_tokens)
                else (right_tokens, left_tokens)
            )
            available = list(longer)
            shared_full_token = False
            matched_all = True
            for token in shorter:
                match_index = next(
                    (
                        index
                        for index, candidate in enumerate(available)
                        if tokens_alias_compatible(token, candidate)
                    ),
                    None,
                )
                if match_index is None:
                    matched_all = False
                    break
                if token == available[match_index] and len(token) > 2:
                    shared_full_token = True
                del available[match_index]
            if matched_all and shared_full_token:
                return True
    return False


def string_values_compatible(left: str, right: str) -> bool:
    """Compatible when equal, prefix-related, or initial-abbreviations."""
    if left == right:
        return True
    if left.startswith(right) or right.startswith(left):
        return True
    return labels_alias_with_initials({left}, {right})


def labels_differ_only_by_initials(
    left_labels: set[str],
    right_labels: set[str],
) -> bool:
    """True when some label pair is identical except for conflicting initials.

    "french company s" vs "french company t" — the full-word token sets are
    identical and each side carries its own short token (an initial or
    single-letter identifier) absent from the other. Authors write exactly
    this shape to *distinguish* entities, so it is evidence of distinctness,
    not of identity — the inverse of :func:`labels_alias_with_initials`,
    where the initial expands a full word on the other side.
    """
    for left_label in left_labels:
        left_tokens = label_tokens(left_label)
        left_long = {token for token in left_tokens if len(token) > 2}
        left_short = {token for token in left_tokens if len(token) <= 2}
        if not left_long or not left_short:
            continue
        for right_label in right_labels:
            right_tokens = label_tokens(right_label)
            right_long = {token for token in right_tokens if len(token) > 2}
            right_short = {token for token in right_tokens if len(token) <= 2}
            if not right_long or not right_short:
                continue
            if left_long != right_long:
                continue
            # Alias-compatible short tokens ("u" vs "us", shared "j") are
            # spelling variance, not a distinguishing mark.
            if not any(
                tokens_alias_compatible(left_token, right_token)
                for left_token in left_short
                for right_token in right_short
            ):
                return True
    return False


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
