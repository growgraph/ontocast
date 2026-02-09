"""Entity normalization for disambiguation.

This module handles the preparation of entities for embedding-based disambiguation.
It creates normalized string representations r(e) that include:
- Normalized form of the entity URI
- Semantic neighbors (types, properties)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rdflib import RDF, RDFS, Literal, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.rdfgraph import RDFGraph

if TYPE_CHECKING:
    from ontocast.tool.agg.uri_builder import EntityRole


@dataclass
class EntityRepresentation:
    """Normalized representation of an entity for embedding.

    Attributes:
        entity: Original entity URI
        normal_form: Normalized string (lowercase, no diacritics, etc.)
        types: List of type URIs for this entity
        properties: List of property URIs used with this entity
        labels: List of labels found for this entity
        representation: Combined string representation r(e) for embedding
        is_ontology_entity: Whether this entity is from an ontology namespace
        role: Detected entity role (class / property / instance)
    """

    entity: URIRef
    normal_form: str
    types: list[URIRef]
    properties: list[URIRef]
    labels: list[str]
    representation: str
    is_ontology_entity: bool
    role: EntityRole | None = field(default=None)


class EntityNormalizer:
    """Normalizes entities and creates string representations for embedding.

    This class is responsible for transforming entity URIs into normalized
    string representations that can be embedded and compared.
    """

    def __init__(self, facts_iri: str = DEFAULT_IRI):
        """Initialize the entity normalizer.

        Args:
            facts_iri: Base IRI for fact entities. Entities under this namespace
                are facts; all other entities are considered ontology entities.
        """
        self.facts_iri = facts_iri.rstrip("/") + "/"

    def normalize_string(self, text: str) -> str:
        """Normalize a string: lowercase, remove diacritics, clean special chars.

        CamelCase is split so that it yields the same logical tokens as snake_case
        (e.g. 'PLRedShift' -> 'pl red shift').

        Args:
            text: Input string to normalize

        Returns:
            Normalized string suitable for comparison

        Examples:
            'PLRedShift' -> 'pl red shift'
            'PL_red_shift_value' -> 'pl red shift value'
            'Café' -> 'cafe'
        """
        # Remove diacritics
        text = "".join(
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

        # Insert space before capitals that start a word (followed by lowercase)
        # so e.g. PLRedShift -> PL Red Shift -> pl red shift (like snake_case)
        text = re.sub(r"(?=[A-Z][a-z])", " ", text)

        # Convert to lowercase
        text = text.lower()

        # Replace underscores and hyphens with spaces
        text = text.replace("_", " ").replace("-", " ")

        # Collapse multiple spaces and strip
        return re.sub(r"\s+", " ", text).strip()

    def normalize_uri(self, uri: URIRef) -> str:
        """Extract and normalize the local part of a URI.

        Args:
            uri: URI to normalize

        Returns:
            Normalized local name

        Examples:
            'http://example.org/PLRedShift' -> 'pl red shift'
            'http://example.org/PL_red_shift_value' -> 'pl red shift value'
        """
        uri_str = str(uri)

        # Extract local name from fragment or path
        if "#" in uri_str:
            local = uri_str.rsplit("#", 1)[-1]
        else:
            trimmed = uri_str.rstrip("/")
            local = trimmed.rsplit("/", 1)[-1] if "/" in trimmed else trimmed

        # Handle camelCase before normalization
        # Insert spaces before uppercase letters
        local = re.sub(r"([a-z])([A-Z])", r"\1 \2", local)
        local = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", local)

        return self.normalize_string(local)

    def is_ontology_entity(self, entity: URIRef) -> bool:
        """Check if an entity belongs to an ontology namespace.

        Facts live under ``facts_iri``; everything else is an ontology entity.

        Args:
            entity: Entity URI to check

        Returns:
            True if entity is **not** from the facts namespace
        """
        return not str(entity).startswith(self.facts_iri)

    def extract_entity_context(
        self, entity: URIRef, graph: RDFGraph
    ) -> tuple[list[URIRef], list[URIRef], list[str], bool]:
        """Extract semantic context for an entity from the graph.

        Args:
            entity: Entity to extract context for
            graph: RDF graph containing the entity

        Returns:
            Tuple of (types, properties, labels, is_predicate).
            *is_predicate* is ``True`` when the entity appears in the
            predicate position of at least one triple.
        """
        types = []
        properties = set()
        labels = []
        is_predicate = False

        # Extract information from triples
        for s, p, o in graph:
            # When entity is subject
            if s == entity:
                properties.add(p)

                # Collect types
                if p == RDF.type and isinstance(o, URIRef):
                    types.append(o)

                # Collect labels
                if p == RDFS.label and isinstance(o, Literal):
                    labels.append(str(o))

            # When entity is object
            elif o == entity:
                properties.add(p)

            # When entity is used as predicate
            if p == entity:
                is_predicate = True

        return types, list(properties), labels, is_predicate

    def create_representation(
        self, entity: URIRef, graph: RDFGraph
    ) -> EntityRepresentation:
        """Create a normalized representation r(e) for an entity.

        This combines the normalized form with semantic neighbors to create
        a rich representation suitable for embedding.  The entity role
        (class / property / instance) is detected from the already-extracted
        context so no additional graph scan is needed downstream.

        Args:
            entity: Entity URI
            graph: RDF graph containing the entity

        Returns:
            EntityRepresentation containing r(e) and metadata
        """
        from ontocast.tool.agg.uri_builder import detect_role_from_context

        # Get normalized form
        normal_form = self.normalize_uri(entity)

        # Extract semantic context
        types, properties, labels, is_predicate = self.extract_entity_context(
            entity, graph
        )

        # Detect role from the already-extracted context (no extra graph scan)
        role = detect_role_from_context(types, is_predicate)

        # Build representation string r(e)
        parts = [normal_form]

        # Add labels if available (most informative)
        if labels:
            parts.extend(
                self.normalize_string(label) for label in labels[:3]
            )  # Max 3 labels

        # Add type information (very important semantic signal)
        if types:
            type_names = [self.normalize_uri(t) for t in types[:3]]  # Max 3 types
            parts.extend(f"type {tn}" for tn in type_names)

        # Add property information (additional semantic signal)
        if properties:
            # Filter out very common properties
            filtered_props = [
                p for p in properties if p not in {RDF.type, RDFS.label, RDFS.comment}
            ]
            prop_names = [
                self.normalize_uri(p) for p in filtered_props[:5]
            ]  # Max 5 properties
            parts.extend(f"has {pn}" for pn in prop_names)

        # Combine into representation
        representation = " ".join(parts)

        # Check if ontology entity
        is_ontology = self.is_ontology_entity(entity)

        return EntityRepresentation(
            entity=entity,
            normal_form=normal_form,
            types=types,
            properties=properties,
            labels=labels,
            representation=representation,
            is_ontology_entity=is_ontology,
            role=role,
        )

    def create_representations_batch(
        self, entities: list[URIRef], graphs: dict[URIRef, RDFGraph]
    ) -> dict[URIRef, EntityRepresentation]:
        """Create representations for multiple entities.

        Args:
            entities: List of entity URIs
            graphs: Mapping from entity to its source graph

        Returns:
            Dictionary mapping entity URIs to their representations
        """
        representations = {}

        for entity in entities:
            graph = graphs.get(entity)
            if graph is not None:
                representations[entity] = self.create_representation(entity, graph)

        return representations
