"""URI construction with naming convention normalization.

Builds final URIs for entity representatives following
RDF/Semantic Web naming conventions (see README.md):
- Classes (entities / types): PascalCase (e.g., JudicialDecision)
- Properties (predicates): lowerCamelCase (e.g., hasDecision)
- Instances with natural names: PascalCase (e.g., FrenchCourtOfCassation)
- Instances with structured/external IDs: preserve structure (e.g., Case_2023_456)

Underscores are avoided in ontology terms (classes, properties).
Underscores are acceptable for instances derived from external IDs.
"""

import logging
import re
from enum import StrEnum

from rdflib import OWL, RDF, RDFS, URIRef

from ontocast.onto.constants import DEFAULT_IRI
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

    The leading word segment is capitalised so that the result starts
    with an uppercase letter (e.g. ``Case_2023_456``).

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
    # Capitalise first segment for readability
    parts = cleaned.split("_", 1)
    parts[0] = parts[0].capitalize()
    return "_".join(parts)


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

    if role == EntityRole.INSTANCE and has_structured_id(representation.entity):
        return format_structured_id(representation.entity)

    # Classes and instances with natural names → PascalCase
    return to_pascal_case(representation.normal_form)


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
        self.base_iri = base_iri.rstrip("/") + "/"
        self._used_names: set[str] = set()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def is_ontology_entity(self, entity: URIRef) -> bool:
        """Return True if *entity* does **not** belong to the facts namespace."""
        return not str(entity).startswith(self.base_iri)

    @staticmethod
    def _extract_namespace(entity: URIRef) -> str:
        """Extract the namespace part of a URI (everything before the local name).

        For ``http://example.org/ns#Foo`` returns ``http://example.org/ns#``.
        For ``http://example.org/ns/Foo`` returns ``http://example.org/ns/``.
        """
        uri_str = str(entity)
        if "#" in uri_str:
            return uri_str.rsplit("#", 1)[0] + "#"
        trimmed = uri_str.rstrip("/")
        if "/" in trimmed:
            return trimmed.rsplit("/", 1)[0] + "/"
        return uri_str

    def _ensure_unique(self, name: str) -> str:
        """Return *name* or ``name_N`` if *name* was already used."""
        if name not in self._used_names:
            self._used_names.add(name)
            return name

        counter = 1
        while f"{name}_{counter}" in self._used_names:
            counter += 1
        unique = f"{name}_{counter}"
        self._used_names.add(unique)
        return unique

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
        unique_name = self._ensure_unique(local_name)

        # Use target_iri (doc_iri) when provided, otherwise fall back to base_iri
        base = (str(target_iri).rstrip("/") + "/") if target_iri else self.base_iri
        return URIRef(f"{base}{unique_name}")

    def create_uri_mapping(
        self,
        representatives: list[URIRef],
        representations: dict[URIRef, EntityRepresentation],
        entity_doc_iris: dict[URIRef, URIRef] | None = None,
        entity_is_ontology: dict[URIRef, bool] | None = None,
    ) -> dict[URIRef, URIRef]:
        """Create a mapping from representative URIs to normalised URIs.

        The role of each entity is read from its
        :class:`EntityRepresentation` (computed during normalisation),
        so no graph lookup is needed here.

        When *entity_doc_iris* is supplied, fact entities are placed under
        their document IRI instead of the default ``base_iri``.  This allows
        chunks from different documents to retain distinct namespaces even
        after clustering.

        Args:
            representatives: Representative entity URIs (one per cluster).
            representations: All entity representations.
            entity_doc_iris: Optional mapping from entity to its source
                document IRI.  When provided the ``doc_iri`` of the
                representative is used as the target namespace for all fact
                entities in that cluster.
            entity_is_ontology: Optional explicit ontology/fact classification.
                When provided, this classification is used instead of
                inferring entity type from namespace shape.

        Returns:
            Mapping ``e_rep → final_uri``.
        """
        mapping: dict[URIRef, URIRef] = {}

        for entity in representatives:
            rep = representations.get(entity)
            if rep is None:
                mapping[entity] = entity
                continue

            role = rep.role if rep.role is not None else EntityRole.INSTANCE
            doc_iri = entity_doc_iris.get(entity) if entity_doc_iris else None
            is_ontology = entity_is_ontology.get(entity) if entity_is_ontology else None
            mapping[entity] = self.build_uri(
                entity,
                rep,
                role,
                target_iri=doc_iri,
                is_ontology_entity=is_ontology,
            )

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
