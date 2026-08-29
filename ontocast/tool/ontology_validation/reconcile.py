"""Reduce-time reconciliation of minted terms against full catalog terminals.

The per-unit lane cannot do this job: under vector-retrieval context the
snapshot is a retrieved subset, so a unit that re-mints a concept the catalog
already defines is — from inside the unit — inventing a genuinely new term.
The duplicate lives precisely in the part of the catalog the snapshot does not
contain, which is why ``_label_collision_findings`` (indexed on the snapshot)
is structurally blind to it. The reduce step is the first place the full
terminals are in hand — they are already fetched there for the namespace owner
map — so that is where minted terms are checked.

Matching is deliberately the strictest rule that exists in this codebase:
exact surface form (``build_surface_index`` — case-sensitive labels,
prefLabels, notations), resolving to exactly **one** catalog IRI
(``resolve_unique_surface`` refuses ambiguous surfaces), with a compatible
role (a minted property never reconciles onto a catalog class or vice versa).
Everything looser is a judgment call and stays out of the deterministic lane.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel
from rdflib import OWL, RDF, RDFS, SKOS, Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation.terms import (
    _PROPERTY_TYPES,
    _catalog_term_roles,
    build_surface_index,
    resolve_unique_surface,
)

logger = logging.getLogger(__name__)

#: Predicates whose literal object names a term for reconciliation purposes.
_SURFACE_PREDICATES = (RDFS.label, SKOS.prefLabel, SKOS.notation)

#: Delta evidence that a minted term acts as a property / as a class.
_PROPERTY_EVIDENCE_PREDICATES = (RDFS.domain, RDFS.range, RDFS.subPropertyOf)
_CLASS_EVIDENCE_PREDICATES = (RDFS.subClassOf,)


class MintedDuplicate(BaseModel):
    """One minted term that exactly matches an existing catalog term."""

    minted_iri: str
    catalog_iri: str
    #: The exact surface form (label/prefLabel/notation) both terms share.
    surface: str
    #: ``"property"`` / ``"class"`` / ``"unknown"`` — the minted term's role
    #: as evidenced by the delta itself.
    role: str


def _minted_roles(inserts: RDFGraph) -> dict[URIRef, str]:
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


def _role_compatible(minted_role: str, catalog_iri: str, terminal: RDFGraph) -> bool:
    """True when reconciling would not cross the class/property divide."""
    properties, classes = _catalog_term_roles(terminal)
    catalog = str(catalog_iri)
    if minted_role == "property":
        return catalog in properties or catalog not in classes
    if minted_role == "class":
        return catalog in classes or catalog not in properties
    # Unknown minted role: only reconcile onto a term the catalog itself
    # leaves role-free — anything stronger is a guess.
    return catalog not in properties and catalog not in classes


def detect_minted_duplicates(
    merged_inserts: RDFGraph,
    terminal_graphs: dict[str, RDFGraph],
) -> list[MintedDuplicate]:
    """Find minted terms whose surface form the full catalog already declares.

    Args:
        merged_inserts: Document-level merged insert delta.
        terminal_graphs: Writable IRI -> the freshest full terminal graph
            (the same graphs the apply step writes onto).

    Returns:
        One record per (minted term, catalog term) unique-surface match with a
        compatible role, ordered by minted IRI. Detection only — the caller
        decides whether to rewrite.
    """
    if not terminal_graphs or len(merged_inserts) == 0:
        return []

    terminal_subjects: set[str] = set()
    for terminal in terminal_graphs.values():
        for subject in terminal.subjects():
            if isinstance(subject, URIRef):
                terminal_subjects.add(str(subject))

    roles = _minted_roles(merged_inserts)
    indexed_terminals = [
        (terminal, build_surface_index(terminal))
        for terminal in terminal_graphs.values()
    ]

    def first_match(subject: URIRef, role: str) -> MintedDuplicate | None:
        for predicate in _SURFACE_PREDICATES:
            for value in merged_inserts.objects(subject, predicate):
                if not isinstance(value, Literal):
                    continue
                surface = str(value).strip()
                if not surface:
                    continue
                for terminal, index in indexed_terminals:
                    catalog_iri = resolve_unique_surface(index, surface)
                    if catalog_iri is None or str(catalog_iri) == str(subject):
                        continue
                    if not _role_compatible(role, str(catalog_iri), terminal):
                        continue
                    return MintedDuplicate(
                        minted_iri=str(subject),
                        catalog_iri=str(catalog_iri),
                        surface=surface,
                        role=role,
                    )
        return None

    duplicates: list[MintedDuplicate] = []
    for subject in sorted(
        {s for s in merged_inserts.subjects() if isinstance(s, URIRef)}, key=str
    ):
        if str(subject) in terminal_subjects:
            continue
        match = first_match(subject, roles.get(subject, "unknown"))
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
