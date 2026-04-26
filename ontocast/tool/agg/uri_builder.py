"""URI construction with naming convention normalization.

Builds final URIs for entity representatives following
RDF/Semantic Web naming conventions (see README.md):
- Classes (entities / types): PascalCase (e.g., JudicialDecision)
- Properties (predicates): lowerCamelCase (e.g., hasDecision)
- Instances with natural names: lowerCamelCase (e.g., frenchCourtOfCassation)
- Instances with structured/external IDs: preserve structure (e.g., case_2023_456)

Underscores are avoided in ontology terms (classes, properties).
Underscores are acceptable for instances derived from external IDs.
"""

import logging
import re
from enum import StrEnum

from rdflib import OWL, RDF, RDFS, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.iri_policy import (
    is_in_namespace,
    join_namespace_local,
    normalize_namespace_iri,
    split_namespace_local,
)
from ontocast.onto.rdfgraph import RDFGraph

from .normalizer import EntityRepresentation

logger = logging.getLogger(__name__)

# Types that mark an entity as a class
_CLASS_TYPES = frozenset({RDFS.Class, OWL.Class})

# Types that mark an entity as a property
_PROPERTY_TYPES = frozenset(
    {RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty}
)


class EntityRole(StrEnum):
    """Role of an entity in an RDF graph."""

    CLASS = "class"
    PROPERTY = "property"
    INSTANCE = "instance"


def detect_role(entity: URIRef, graph: RDFGraph) -> EntityRole:
    """Detect the role of an entity: class, property, or instance.

    Args:
        entity: The entity URI.
        graph: The RDF graph containing the entity.

    Returns:
        The detected :class:`EntityRole`.
    """
    entity_types: set[URIRef] = set()
    is_predicate = False

    for s, p, o in graph:
        if s == entity and p == RDF.type and isinstance(o, URIRef):
            entity_types.add(o)
        if p == entity:
            is_predicate = True

    if entity_types & _CLASS_TYPES:
        return EntityRole.CLASS
    if entity_types & _PROPERTY_TYPES or is_predicate:
        return EntityRole.PROPERTY
    return EntityRole.INSTANCE


def detect_role_from_context(
    types: list[URIRef],
    is_predicate: bool = False,
) -> EntityRole:
    """Detect entity role from pre-extracted context (no graph scan needed).

    This is the preferred entry point when the caller has already extracted
    types and predicate usage via
    :meth:`EntityNormalizer.extract_entity_context`, avoiding a redundant
    full-graph iteration.

    Args:
        types: ``rdf:type`` values of the entity.
        is_predicate: Whether the entity appears in the predicate position
            of at least one triple.

    Returns:
        The detected :class:`EntityRole`.
    """
    type_set = frozenset(types)

    if type_set & _CLASS_TYPES:
        return EntityRole.CLASS
    if type_set & _PROPERTY_TYPES or is_predicate:
        return EntityRole.PROPERTY
    return EntityRole.INSTANCE


def to_pascal_case(normalized: str) -> str:
    """Convert a space-separated lowercase string to PascalCase.

    Args:
        normalized: Space-separated lowercase string.

    Returns:
        PascalCase string.

    Examples:
        >>> to_pascal_case('judicial decision')
        'JudicialDecision'
        >>> to_pascal_case('french court of cassation')
        'FrenchCourtOfCassation'
    """
    words = normalized.split()
    return "".join(w.capitalize() for w in words if w)


def to_lower_camel_case(normalized: str) -> str:
    """Convert a space-separated lowercase string to lowerCamelCase.

    Args:
        normalized: Space-separated lowercase string.

    Returns:
        lowerCamelCase string.

    Examples:
        >>> to_lower_camel_case('has decision')
        'hasDecision'
        >>> to_lower_camel_case('date published')
        'datePublished'
    """
    words = normalized.split()
    if not words:
        return ""
    return words[0] + "".join(w.capitalize() for w in words[1:])


def has_structured_id(entity: URIRef) -> bool:
    """Detect if an entity represents a structured/external identifier.

    Structured IDs contain digits together with underscores, e.g.
    ``Case_2023_456`` or ``Decision_2021_09_15``.

    Args:
        entity: Original entity URI.

    Returns:
        True if the entity appears to have a structured ID.
    """
    local = str(entity).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return bool(re.search(r"\d", local) and "_" in local)


def format_structured_id(entity: URIRef) -> str:
    """Format a structured identifier preserving underscores and digits.

    Structured IDs stay lowercase/snake-like after sanitization.

    Args:
        entity: Original entity URI.

    Returns:
        Cleaned identifier string.
    """
    local = str(entity).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    cleaned = re.sub(r"[^\w]", "_", local)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "Entity"
    return cleaned


def normalize_local_name(
    representation: EntityRepresentation,
    role: EntityRole | str,
) -> str:
    """Produce a properly-cased local name following RDF conventions.

    Args:
        representation: Entity representation with metadata.
        role: Entity role (an :class:`EntityRole` value).

    Returns:
        Properly cased local name.
    """
    if role == EntityRole.PROPERTY:
        return to_lower_camel_case(representation.normal_form)

    if role == EntityRole.CLASS:
        return to_pascal_case(representation.normal_form)

    if role == EntityRole.INSTANCE and has_structured_id(representation.iri):
        return format_structured_id(representation.iri)

    # Instances with natural names → lowerCamelCase
    return to_lower_camel_case(representation.normal_form)


class URIBuilder:
    """Build normalized URIs for all entities following RDF naming conventions.

    - **Fact entities** (under *base_iri*) get new URIs under *base_iri*.
    - **Ontology entities** (everything else) are preserved as-is.
    """

    def __init__(
        self,
        base_iri: str = DEFAULT_IRI,
    ):
        """Initialise the builder.

        Args:
            base_iri: Base IRI for fact entities (default ``DEFAULT_IRI``).
                Entities under this namespace are facts; everything else is
                treated as an ontology entity.
        """
        self.base_iri = normalize_namespace_iri(base_iri, context="facts")
        self._used_uris: set[URIRef] = set()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def is_ontology_entity(self, entity: URIRef) -> bool:
        """Return True if *entity* does **not** belong to the facts namespace."""
        return not is_in_namespace(str(entity), self.base_iri, context="facts")

    @staticmethod
    def _extract_namespace(entity: URIRef) -> str:
        """Extract the namespace part of a URI (everything before the local name).

        For ``http://example.org/ns#Foo`` returns ``http://example.org/ns#``.
        For ``http://example.org/ns/Foo`` returns ``http://example.org/ns/``.
        """
        namespace, _ = split_namespace_local(str(entity))
        return namespace or str(entity)

    def _ensure_unique_uri(self, base: str, local_name: str) -> URIRef:
        """Return a unique URI under *base* for *local_name*."""
        candidate = URIRef(join_namespace_local(base, local_name, context="auto"))
        if candidate not in self._used_uris:
            self._used_uris.add(candidate)
            return candidate

        counter = 1
        while True:
            candidate = URIRef(
                join_namespace_local(base, f"{local_name}_{counter}", context="auto")
            )
            if candidate not in self._used_uris:
                self._used_uris.add(candidate)
                return candidate
            counter += 1

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build_uri(
        self,
        entity: URIRef,
        representation: EntityRepresentation,
        role: EntityRole | str,
        target_iri: URIRef | str | None = None,
        is_ontology_entity: bool | None = None,
    ) -> URIRef:
        """Build a normalised URI for a single entity.

        Fact entities are normalised and placed under *target_iri* (falling
        back to *base_iri*). Ontology entities are preserved as-is.

        Args:
            entity: Original entity URI.
            representation: Entity representation with metadata.
            role: Entity role (an :class:`EntityRole` value).
            target_iri: Optional document IRI to use as namespace for fact
                entities instead of the default *base_iri*.  When chunks carry
                different ``doc_iri`` values the caller passes the appropriate
                one here so that each fact is placed under its document
                namespace.
            is_ontology_entity: Explicit ontology/fact classification.  When
                provided this takes precedence over namespace-based inference.

        Returns:
            Normalised URI.
        """
        is_ontology = (
            self.is_ontology_entity(entity)
            if is_ontology_entity is None
            else is_ontology_entity
        )

        if is_ontology:
            return entity

        local_name = normalize_local_name(representation, role)
        base = (
            normalize_namespace_iri(str(target_iri), context="facts")
            if target_iri
            else self.base_iri
        )
        return self._ensure_unique_uri(base=base, local_name=local_name)

    def create_entity_uri_mapping(
        self,
        identity_mapping: dict[URIRef, URIRef],
        representations: dict[URIRef, EntityRepresentation],
        entity_doc_iris: dict[URIRef, URIRef],
        entity_is_ontology: dict[URIRef, bool],
    ) -> dict[URIRef, URIRef]:
        """Create final URI mapping from identity mapping + namespace policy.

        This method decouples canonical identity choice from URI surface choice:
        identity mapping decides *what* is the same entity, while this method
        decides *how* each source entity should be rendered as a final URI.
        Fact entities are always rendered in their source ``doc_iri`` namespace.
        Ontology entities are preserved as their canonical URI.

        Args:
            identity_mapping: Mapping ``entity -> canonical_entity``.
            representations: All entity representations.
            entity_doc_iris: Mapping from source entity to source ``doc_iri``.
            entity_is_ontology: Classification map where ``True`` means the
                canonical entity should stay in ontology space.

        Returns:
            Mapping ``entity -> final_uri``.
        """
        self._used_uris.clear()
        mapping: dict[URIRef, URIRef] = {}
        canonical_cache: dict[tuple[URIRef, str], URIRef] = {}

        for entity, canonical in identity_mapping.items():
            rep = representations.get(canonical)
            if rep is None:
                mapping[entity] = entity
                continue

            role = rep.role if rep.role is not None else EntityRole.INSTANCE
            is_ontology = entity_is_ontology.get(
                canonical, self.is_ontology_entity(canonical)
            )
            if is_ontology:
                mapping[entity] = canonical
                continue

            doc_iri = entity_doc_iris.get(entity)
            base = (
                normalize_namespace_iri(str(doc_iri), context="facts")
                if doc_iri
                else self.base_iri
            )
            cache_key = (canonical, base)
            if cache_key in canonical_cache:
                mapping[entity] = canonical_cache[cache_key]
                continue

            canonical_uri = self.build_uri(
                canonical,
                rep,
                role,
                target_iri=doc_iri,
                is_ontology_entity=False,
            )
            canonical_cache[cache_key] = canonical_uri
            mapping[entity] = canonical_uri

        normalised = sum(1 for e, u in mapping.items() if e != u)
        logger.info(
            f"Built URI mapping: {len(mapping)} entities, {normalised} normalised"
        )
        return mapping

    @staticmethod
    def compose_mappings(
        clustering_mapping: dict[URIRef, URIRef],
        uri_mapping: dict[URIRef, URIRef],
    ) -> dict[URIRef, URIRef]:
        """Compose clustering and URI mappings.

        ``e → representative(e) → normalised_uri(representative(e))``

        Args:
            clustering_mapping: ``e → e_rep``.
            uri_mapping: ``e_rep → final_uri``.

        Returns:
            Composed mapping ``e → final_uri``.
        """
        composed = {
            original: uri_mapping.get(representative, representative)
            for original, representative in clustering_mapping.items()
        }
        logger.info(
            f"Composed mapping: {len(composed)} entities → "
            f"{len(set(composed.values()))} final URIs"
        )
        return composed
