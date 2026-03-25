"""Entity normalization for disambiguation.

This module handles the preparation of entities for embedding-based disambiguation.
It creates normalized string representations r(e) that include:
- Normalized form of the entity URI
- Semantic neighbors (types, properties)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rdflib import RDF, RDFS, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.representation_contract import combine_embedding_text
from ontocast.tool.representation_text import (
    normalize_text,
    normalize_uri_local_name,
    render_term_for_text,
    stable_sorted_triples,
)

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

    iri: URIRef
    normal_form: str
    types: list[URIRef]
    properties: list[URIRef]
    labels: list[str]
    is_ontology_entity: bool
    role: EntityRole | None = field(default=None)
    core_representation: str = ""
    neighborhood_representation: str = ""
    representation: str = ""

    def __post_init__(self) -> None:
        if not self.core_representation:
            self.core_representation = self.representation or self.normal_form
        if not self.representation:
            self.representation = combine_embedding_text(self)

    ontology_iri: str | None = None


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
        # Keep legacy behavior for this method while sharing the same core utility.
        text = re.sub(r"(?=[A-Z][a-z])", " ", text)
        return normalize_text(text)

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
        return normalize_uri_local_name(uri)

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

        sorted_types = sorted(types, key=lambda entity: str(entity))
        sorted_properties = sorted(properties, key=lambda entity: str(entity))
        return sorted_types, sorted_properties, labels, is_predicate

    def _render_term(self, term: Node) -> str:
        return render_term_for_text(term)

    def _build_neighborhood_representation(
        self, entity: URIRef, graph: RDFGraph
    ) -> str:
        by_role: dict[str, list[str]] = {
            "as_subject": [],
            "as_object": [],
            "as_predicate": [],
        }
        seen_by_role: dict[str, set[str]] = {
            "as_subject": set(),
            "as_object": set(),
            "as_predicate": set(),
        }

        triples_sorted = stable_sorted_triples(list(graph))
        for subj, pred, obj in triples_sorted:
            if subj == entity:
                sentence = (
                    f"{self._render_term(subj)} has relation {self._render_term(pred)} "
                    f"to {self._render_term(obj)}"
                )
                if sentence not in seen_by_role["as_subject"]:
                    seen_by_role["as_subject"].add(sentence)
                    by_role["as_subject"].append(sentence)
            if obj == entity:
                sentence = (
                    f"{self._render_term(subj)} relates via {self._render_term(pred)} "
                    f"to this entity {self._render_term(obj)}"
                )
                if sentence not in seen_by_role["as_object"]:
                    seen_by_role["as_object"].add(sentence)
                    by_role["as_object"].append(sentence)
            if pred == entity:
                sentence = (
                    f"predicate {self._render_term(pred)} links {self._render_term(subj)} "
                    f"and {self._render_term(obj)}"
                )
                if sentence not in seen_by_role["as_predicate"]:
                    seen_by_role["as_predicate"].add(sentence)
                    by_role["as_predicate"].append(sentence)

        selected: list[str] = []
        cap_per_role = 3
        for role in ("as_subject", "as_object", "as_predicate"):
            selected.extend(by_role[role][:cap_per_role])
        if not selected:
            return "no neighborhood facts available"
        return ". ".join(selected)

    def _build_core_representation(
        self,
        *,
        normal_form: str,
        types: list[URIRef],
        properties: list[URIRef],
        labels: list[str],
    ) -> str:
        # Prefer grammatical text over schema-like field labels to improve
        # embedding alignment with natural-language queries.
        sentences: list[str] = [normal_form]
        if labels:
            normalized_labels = [self.normalize_string(label) for label in labels[:3]]
            sentences.append(f"It is labeled {', '.join(normalized_labels)}")
        if types:
            type_names = [self.normalize_uri(entity_type) for entity_type in types[:3]]
            # Keep the 'type' keyword to maintain compatibility with any
            # downstream keyword checks in unit tests.
            sentences.append(f"It has type {', '.join(type_names)}")
        if properties:
            filtered_props = [
                prop
                for prop in properties
                if prop not in {RDF.type, RDFS.label, RDFS.comment}
            ]
            prop_names = [self.normalize_uri(prop) for prop in filtered_props[:5]]
            sentences.append(f"It has properties {', '.join(prop_names)}")
        return ". ".join(sentences)

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

        core_representation = self._build_core_representation(
            normal_form=normal_form,
            types=types,
            properties=properties,
            labels=labels,
        )
        neighborhood_representation = self._build_neighborhood_representation(
            entity=entity,
            graph=graph,
        )

        # Check if ontology entity
        is_ontology = self.is_ontology_entity(entity)

        return EntityRepresentation(
            iri=entity,
            normal_form=normal_form,
            types=types,
            properties=properties,
            labels=labels,
            is_ontology_entity=is_ontology,
            role=role,
            core_representation=core_representation,
            neighborhood_representation=neighborhood_representation,
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
