"""Reduce-time reconciliation of minted terms against the whole catalog.

The per-unit lane sees a snapshot; under vector-retrieval context that is a
retrieved subset, so a unit that re-mints a concept the catalog already
defines is -- from inside the unit -- inventing a new term. The reduce step
checks the merged document delta against every catalog ontology, through the
manager's :class:`~ontocast.tool.ontology_validation.shape.CatalogSurfaceIndex`.

A minted term matches a catalog term when one of its labels, preferred labels
or notations equals a catalog one case-insensitively, or else when its local
name does (``plRedShiftProperty`` and ``PLRedShiftProperty``). The match must
resolve to exactly **one** catalog IRI -- an ambiguous name is refused -- and
the roles must be compatible: a minted property never reconciles onto a
catalog class, nor the reverse.
"""

from __future__ import annotations

import logging
from typing import Literal as TypingLiteral

from pydantic import BaseModel
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation.terms import _PROPERTY_TYPES, _local_name
from ontocast.tool.ontology_validation.shape import (
    SURFACE_PREDICATES,
    CatalogSurfaceIndex,
    resolve_unique,
    role_compatible,
)

logger = logging.getLogger(__name__)

#: Delta evidence that a minted term acts as a property / as a class.
_PROPERTY_EVIDENCE_PREDICATES = (RDFS.domain, RDFS.range, RDFS.subPropertyOf)
_CLASS_EVIDENCE_PREDICATES = (RDFS.subClassOf,)


class MintedDuplicate(BaseModel):
    """One minted term that names an existing catalog term."""

    minted_iri: str
    catalog_iri: str
    #: The name both terms share: a label, preferred label or notation, or
    #: the local name.
    surface: str
    #: ``"property"`` / ``"class"`` / ``"unknown"`` — the minted term's role
    #: as evidenced by the delta itself.
    role: str
    #: Which name matched.
    key: TypingLiteral["label", "local_name"] = "label"


def minted_roles(inserts: RDFGraph) -> dict[URIRef, str]:
    """Role of each minted subject, from the delta's own declarations."""
    roles: dict[URIRef, str] = {}

    def note(subject: object, role: str) -> None:
        if not isinstance(subject, URIRef):
            return
        current = roles.get(subject)
        if current is None:
            roles[subject] = role
        elif current != role:
            # Contradictory evidence inside one delta — the role-confusion
            # finding owns that defect; reconciliation must not guess.
            roles[subject] = "unknown"

    for type_iri in _PROPERTY_TYPES + (RDF.Property,):
        for subject in inserts.subjects(RDF.type, type_iri):
            note(subject, "property")
    for predicate in _PROPERTY_EVIDENCE_PREDICATES:
        for subject in inserts.subjects(predicate, None):
            note(subject, "property")
    for type_iri in (OWL.Class, RDFS.Class):
        for subject in inserts.subjects(RDF.type, type_iri):
            note(subject, "class")
    for predicate in _CLASS_EVIDENCE_PREDICATES:
        for subject in inserts.subjects(predicate, None):
            note(subject, "class")
    return roles


def surfaces_of(subject: URIRef, graph: RDFGraph) -> list[str]:
    """The labels, preferred labels and notations ``graph`` gives ``subject``."""
    return sorted(
        {
            str(value).strip()
            for predicate in SURFACE_PREDICATES
            for value in graph.objects(subject, predicate)
            if isinstance(value, Literal) and str(value).strip()
        }
    )


def match_catalog_term(
    subject: URIRef,
    inserts: RDFGraph,
    catalog_index: CatalogSurfaceIndex,
    role: str,
) -> MintedDuplicate | None:
    """The catalog term a minted ``subject`` names, if exactly one does.

    Labels are tried before the local name; the role must be compatible. A
    subject the catalog already holds is not minted and never matches.
    """
    text = str(subject)
    if text in catalog_index.terms:
        return None
    candidates: list[tuple[str, str, TypingLiteral["label", "local_name"]]] = []
    for surface in surfaces_of(subject, inserts):
        match = resolve_unique(catalog_index, label=surface)
        if match is not None:
            candidates.append((match[0], surface, match[1]))
            break
    local = _local_name(text)
    if not candidates:
        match = resolve_unique(catalog_index, local=local)
        if match is not None:
            candidates.append((match[0], local, match[1]))
    for catalog_iri, surface, key in candidates:
        if catalog_iri == text or not role_compatible(role, catalog_iri, catalog_index):
            continue
        return MintedDuplicate(
            minted_iri=text,
            catalog_iri=catalog_iri,
            surface=surface,
            role=role,
            key=key,
        )
    return None


def detect_minted_duplicates(
    merged_inserts: RDFGraph,
    catalog_index: CatalogSurfaceIndex,
) -> list[MintedDuplicate]:
    """Find minted terms that name an existing catalog term.

    Args:
        merged_inserts: Document-level merged insert delta.
        catalog_index: Names of every catalog term.

    Returns:
        One record per minted term with a unique, role-compatible match,
        ordered by minted IRI. Detection only — the caller decides whether to
        rewrite.
    """
    if catalog_index.is_empty() or len(merged_inserts) == 0:
        return []
    roles = minted_roles(merged_inserts)
    duplicates: list[MintedDuplicate] = []
    for subject in sorted(
        {s for s in merged_inserts.subjects() if isinstance(s, URIRef)}, key=str
    ):
        match = match_catalog_term(
            subject, merged_inserts, catalog_index, roles.get(subject, "unknown")
        )
        if match is not None:
            duplicates.append(match)
    return duplicates


def apply_minted_duplicate_rewrites(
    merged_inserts: RDFGraph,
    duplicates: list[MintedDuplicate],
) -> int:
    """Rewrite minted IRIs to their catalog IRIs, in place.

    Substitutes in subject **and** object position — a second minted term
    referencing the duplicate must end up pointing at the catalog term, or the
    rewrite would strand it. Predicate position is included for completeness
    (a minted property duplicate used as a predicate elsewhere in the delta).

    Returns:
        Number of triples rewritten.
    """
    if not duplicates:
        return 0
    mapping = {
        URIRef(duplicate.minted_iri): URIRef(duplicate.catalog_iri)
        for duplicate in duplicates
    }
    rewritten = 0
    for triple in list(merged_inserts):
        subject, predicate, obj = triple
        replaced = (
            mapping.get(subject, subject) if isinstance(subject, URIRef) else subject,
            mapping.get(predicate, predicate)
            if isinstance(predicate, URIRef)
            else predicate,
            mapping.get(obj, obj) if isinstance(obj, URIRef) else obj,
        )
        if replaced == triple:
            continue
        merged_inserts.remove(triple)
        merged_inserts.add(replaced)
        rewritten += 1
    return rewritten
