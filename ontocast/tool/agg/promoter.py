"""URI promotion from chunk-level to document-level.

This module handles the promotion of chunk-level entity URIs to document-level URIs,
while preserving ontology entity URIs unchanged.
"""

import logging
import re

from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.iri_policy import (
    is_in_namespace,
    join_namespace_local,
    normalize_namespace_iri,
    split_namespace_local,
)

from .normalizer import EntityRepresentation

logger = logging.getLogger(__name__)


class URIPromoter:
    """Promotes chunk-level URIs to document-level URIs.

    This class is responsible for converting entities from chunk namespaces
    to the document namespace, while keeping ontology entities unchanged.
    """

    def __init__(
        self,
        doc_namespace: str,
        chunk_namespaces: set[str],
        facts_iri: str = DEFAULT_IRI,
    ):
        """Initialize the URI promoter.

        Args:
            doc_namespace: Document namespace for promoted URIs
            chunk_namespaces: Set of chunk namespace URIs
            facts_iri: Base IRI for fact entities. Entities under this
                namespace are facts; all other entities are ontology
                entities and preserved as-is.
        """
        self.doc_namespace = normalize_namespace_iri(doc_namespace, context="facts")
        self.chunk_namespaces = {
            normalize_namespace_iri(namespace, context="facts")
            for namespace in chunk_namespaces
        }
        self.facts_iri = normalize_namespace_iri(facts_iri, context="facts")
        self._used_uris: set[str] = set()

    def _clean_local_name(self, name: str) -> str:
        """Clean a name for use as URI local part.

        Args:
            name: Name to clean

        Returns:
            Cleaned name suitable for URI
        """
        # Replace invalid URI characters with underscores
        cleaned = re.sub(r"[^\w\-.]", "_", name)
        # Remove consecutive underscores
        cleaned = re.sub(r"_+", "_", cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip("_")
        return cleaned or "entity"

    def _ensure_unique_uri(self, uri: str) -> str:
        """Ensure URI is unique by appending counter if needed.

        Args:
            uri: Proposed URI

        Returns:
            Unique URI
        """
        if uri not in self._used_uris:
            self._used_uris.add(uri)
            return uri

        # URI already used, append counter
        base_uri = uri
        counter = 1

        while uri in self._used_uris:
            # Extract local name and add counter
            namespace, local = split_namespace_local(base_uri)
            if namespace is None:
                namespace = normalize_namespace_iri(base_uri, context="facts")
                local = ""
            uri = join_namespace_local(namespace, f"{local}_{counter}", context="auto")
            counter += 1

        self._used_uris.add(uri)
        return uri

    def should_promote(self, entity: URIRef) -> bool:
        """Check if an entity should be promoted to document namespace.

        Args:
            entity: Entity URI to check

        Returns:
            True if entity should be promoted (is from chunk namespace)
        """
        entity_str = str(entity)

        # Don't promote ontology entities (anything not under facts_iri)
        if not is_in_namespace(entity_str, self.facts_iri, context="facts"):
            return False

        # Promote chunk entities
        if any(
            is_in_namespace(entity_str, namespace, context="facts")
            for namespace in self.chunk_namespaces
        ):
            return True

        # Unknown namespace - be conservative, don't promote
        return False

    def promote_entity(
        self, entity: URIRef, representation: EntityRepresentation
    ) -> URIRef:
        """Promote a chunk entity to document namespace.

        Args:
            entity: Original entity URI
            representation: Entity representation with metadata

        Returns:
            Promoted entity URI
        """
        if not self.should_promote(entity):
            # Keep ontology entities unchanged
            return entity

        # Create new URI in document namespace
        # Use normalized form as local name
        local_name = self._clean_local_name(representation.normal_form)

        # Construct promoted URI
        promoted_uri_str = join_namespace_local(
            self.doc_namespace,
            local_name,
            context="facts",
        )

        # Ensure uniqueness
        unique_uri_str = self._ensure_unique_uri(promoted_uri_str)

        return URIRef(unique_uri_str)

    def create_promotion_mapping(
        self,
        entities: list[URIRef],
        representations: dict[URIRef, EntityRepresentation],
    ) -> dict[URIRef, URIRef]:
        """Create mapping from original to promoted URIs.

        Args:
            entities: List of entity URIs to promote
            representations: Dictionary mapping entities to their representations

        Returns:
            Dictionary mapping original URIs to promoted URIs (e -> promoted(e))
        """
        mapping = {}
        promoted_count = 0
        preserved_count = 0

        for entity in entities:
            representation = representations.get(entity)
            if representation is None:
                logger.warning(f"No representation found for entity {entity}")
                mapping[entity] = entity
                continue

            promoted = self.promote_entity(entity, representation)
            mapping[entity] = promoted

            if promoted != entity:
                promoted_count += 1
            else:
                preserved_count += 1

        logger.info(
            f"Created promotion mapping: "
            f"{promoted_count} promoted, "
            f"{preserved_count} preserved (ontology entities)"
        )

        return mapping

    def compose_mappings(
        self,
        clustering_mapping: dict[URIRef, URIRef],
        promotion_mapping: dict[URIRef, URIRef],
    ) -> dict[URIRef, URIRef]:
        """Compose clustering and promotion mappings.

        First, entities are mapped to their cluster representatives (clustering_mapping).
        Then, representatives are promoted to document namespace (promotion_mapping).

        The composed mapping is: e -> promoted(representative(e))

        Args:
            clustering_mapping: Map from entity to cluster representative (e -> e_rep)
            promotion_mapping: Map from representative to promoted URI (e_rep -> e')

        Returns:
            Composed mapping (e -> e')
        """
        composed = {}

        for original, representative in clustering_mapping.items():
            # Look up the promoted version of the representative
            promoted = promotion_mapping.get(representative, representative)
            composed[original] = promoted

        logger.info(
            f"Composed mapping: {len(composed)} entities mapped to "
            f"{len(set(composed.values()))} final URIs"
        )

        return composed
