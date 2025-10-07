from collections import defaultdict
from typing import Mapping

from rapidfuzz import fuzz
from rdflib import RDF, RDFS, Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import derive_ontology_id
from ontocast.tool.onto import EntityMetadata, PredicateMetadata


class EntityDisambiguator:
    """Disambiguate and aggregate entities across multiple chunk graphs.

    This class provides functionality for identifying and resolving similar
    entities across different chunks of text, using string similarity and
    semantic information.

    Attributes:
        similarity_threshold: Threshold for considering entities similar.
        semantic_threshold: Higher threshold for semantic similarity.
    """

    def __init__(
        self, similarity_threshold: float = 85.0, semantic_threshold: float = 90.0
    ):
        """Initialize the entity disambiguator.

        Args:
            similarity_threshold: Threshold for considering entities similar (default: 85.0).
            semantic_threshold: Higher threshold for semantic similarity (default: 90.0).
        """
        self.similarity_threshold = similarity_threshold
        self.semantic_threshold = semantic_threshold

    def normalize_uri(
        self, uri: URIRef, namespaces: Mapping[str, str | URIRef]
    ) -> tuple[str, str]:
        """Normalize a URI by expanding any prefixed names and extracting a proper local name.

        Args:
            uri: The URI to normalize.
            namespaces: Dictionary of namespace prefixes to URIs.

        Returns:
            tuple[str, str]: The full URI and local name.
        """
        uri_str = str(uri)

        # Expand prefixed names like ns:Thing to full URIs when we can
        for prefix, namespace in namespaces.items():
            if uri_str.startswith(f"{prefix}:"):
                # Handle both str and URIRef namespace values
                namespace_str = str(namespace)
                full_uri = uri_str.replace(f"{prefix}:", namespace_str)
                uri_str = full_uri
                break

        # Extract local name from fragment or last path segment
        if "#" in uri_str:
            local = uri_str.rsplit("#", 1)[-1]
        else:
            trimmed = uri_str.rstrip("/")
            local = trimmed.rsplit("/", 1)[-1] if "/" in trimmed else trimmed

        return uri_str, local

    def extract_entity_labels(self, graph: RDFGraph) -> dict[URIRef, EntityMetadata]:
        """Extract labels for entities from graph, including their local names.

        Args:
            graph: The RDF graph to process.

        Returns:
            dict[URIRef, EntityMetadata]: Dictionary mapping entity URIs to their metadata.
        """
        labels = {}
        namespaces = dict(graph.namespaces())

        # Collect all entities first
        all_entities = set()
        for subj, pred, obj in graph:
            if isinstance(subj, URIRef):
                all_entities.add(subj)
            if isinstance(obj, URIRef):
                all_entities.add(obj)

        # Initialize metadata for all entities
        for entity in all_entities:
            full_uri, local_name = self.normalize_uri(entity, namespaces)
            uri_ref = URIRef(full_uri)
            labels[uri_ref] = EntityMetadata(local_name=local_name)

        # Collect explicit labels and comments
        for subj, pred, obj in graph:
            if (
                pred in [RDFS.label, RDFS.comment]
                and isinstance(obj, Literal)
                and isinstance(subj, URIRef)
            ):
                full_uri, _ = self.normalize_uri(subj, namespaces)
                uri_ref = URIRef(full_uri)

                if uri_ref in labels:
                    if pred == RDFS.label:
                        labels[uri_ref].label = str(obj)
                    elif pred == RDFS.comment:
                        labels[uri_ref].comment = str(obj)

        return labels

    def find_similar_entities(
        self,
        entities_with_labels: dict[URIRef, EntityMetadata],
        entity_types: dict[URIRef, set[URIRef]] | None = None,
    ) -> list[list[URIRef]]:
        """Group similar entities based on string similarity, local names, and types.

        Args:
            entities_with_labels: Dictionary mapping entity URIs to their metadata.
            entity_types: Optional dictionary mapping entities to their types.

        Returns:
            list[list[URIRef]]: Groups of similar entities.
        """
        if entity_types is None:
            entity_types = {}

        # Create lookup structures for optimization
        local_name_groups = defaultdict(list)
        label_to_entities = defaultdict(list)

        for entity, metadata in entities_with_labels.items():
            # Group by normalized local name for exact matching
            normalized_local = metadata.local_name.lower().strip()
            if normalized_local:
                local_name_groups[normalized_local].append(entity)

            # Group by normalized label for similarity matching
            if metadata.label:
                normalized_label = metadata.label.lower().strip()
                label_to_entities[normalized_label].append(entity)

        entity_groups = []
        processed = set()

        # First pass: exact local name matches
        for local_name, entities in local_name_groups.items():
            if len(entities) > 1:
                # Check type compatibility within the group
                compatible_group = self._filter_by_type_compatibility(
                    entities, entity_types
                )
                if len(compatible_group) > 1:
                    entity_groups.append(compatible_group)
                    processed.update(compatible_group)

        # Second pass: exact label matches
        for label, entities in label_to_entities.items():
            unprocessed_entities = [e for e in entities if e not in processed]
            if len(unprocessed_entities) > 1:
                compatible_group = self._filter_by_type_compatibility(
                    unprocessed_entities, entity_types
                )
                if len(compatible_group) > 1:
                    entity_groups.append(compatible_group)
                    processed.update(compatible_group)

        # Third pass: fuzzy label matching for remaining entities
        remaining_entities = [
            e for e in entities_with_labels.keys() if e not in processed
        ]
        if len(remaining_entities) > 1:
            fuzzy_groups = self._find_fuzzy_similar_entities(
                remaining_entities, entities_with_labels, entity_types
            )
            entity_groups.extend(fuzzy_groups)

        return entity_groups

    def _filter_by_type_compatibility(
        self, entities: list[URIRef], entity_types: dict[URIRef, set[URIRef]]
    ) -> list[URIRef]:
        """Filter entities by type compatibility."""
        if not entity_types:
            return entities

        # Group entities by their types
        type_groups = defaultdict(list)
        typeless_entities = []

        for entity in entities:
            types = entity_types.get(entity, set())
            if not types:
                typeless_entities.append(entity)
            else:
                # Use frozenset for hashable type signature
                type_signature = frozenset(types)
                type_groups[type_signature].append(entity)

        # Combine compatible groups
        compatible_entities = []

        # Add the largest type group
        if type_groups:
            largest_group = max(type_groups.values(), key=len)
            compatible_entities.extend(largest_group)

        # Add typeless entities (they're compatible with everything)
        compatible_entities.extend(typeless_entities)

        return compatible_entities

    def _find_fuzzy_similar_entities(
        self,
        entities: list[URIRef],
        entities_with_labels: dict[URIRef, EntityMetadata],
        entity_types: dict[URIRef, set[URIRef]],
    ) -> list[list[URIRef]]:
        """Find fuzzy similar entities among the remaining entities."""
        entity_groups = []
        processed = set()

        for i, entity1 in enumerate(entities):
            if entity1 in processed:
                continue

            similar_group = [entity1]
            metadata1 = entities_with_labels[entity1]
            types1 = entity_types.get(entity1, set())
            processed.add(entity1)

            if not metadata1.label:  # Skip entities without labels
                continue

            label1 = metadata1.label.lower().strip()

            for entity2 in entities[i + 1 :]:
                if entity2 in processed:
                    continue

                metadata2 = entities_with_labels[entity2]
                types2 = entity_types.get(entity2, set())

                if not metadata2.label:  # Skip entities without labels
                    continue

                label2 = metadata2.label.lower().strip()

                # Check type compatibility
                if not self._are_types_compatible(types1, types2):
                    continue

                # Calculate label similarity
                similarity = fuzz.ratio(label1, label2)

                # Use higher threshold if entities share types
                threshold = (
                    self.semantic_threshold
                    if types1.intersection(types2)
                    else self.similarity_threshold
                )

                if similarity >= threshold:
                    similar_group.append(entity2)
                    processed.add(entity2)

            if len(similar_group) > 1:
                entity_groups.append(similar_group)

        return entity_groups

    def _are_types_compatible(self, types1: set[URIRef], types2: set[URIRef]) -> bool:
        """Check if two sets of types are compatible."""
        # Compatible if one has no types, or they share at least one type
        if not types1 or not types2:
            return True
        return bool(types1.intersection(types2))

    def create_canonical_iri(
        self,
        similar_entities: list[URIRef],
        doc_namespace: str,
        entity_labels: dict[URIRef, EntityMetadata],
        preferred_namespaces: set[str] | None = None,
    ) -> URIRef:
        """Create a canonical URI for a group of similar entities.

        Args:
            similar_entities: List of similar entity URIs.
            doc_namespace: The document namespace to use.
            entity_labels: Dictionary mapping entities to their metadata.
            preferred_namespaces: Optional set of preferred ontology namespaces.

        Returns:
            URIRef: The canonical URI for the group.
        """
        # Priority 1: Use entity from preferred ontology namespace
        if preferred_namespaces:
            for entity in similar_entities:
                entity_str = str(entity)
                if any(entity_str.startswith(ns) for ns in preferred_namespaces):
                    return entity

        # Priority 2: Choose entity with the best label (longest, most descriptive)
        best_entity = max(
            similar_entities,
            key=lambda e: len(
                entity_labels.get(e, EntityMetadata(local_name="")).label or ""
            ),
        )

        best_metadata = entity_labels.get(best_entity, EntityMetadata(local_name=""))
        local_name = best_metadata.local_name or derive_ontology_id(best_entity)

        # Create canonical URI in document namespace
        clean_local_name = self._clean_local_name(local_name)
        return URIRef(f"{doc_namespace}{clean_local_name}")

    def create_canonical_predicate(
        self,
        similar_predicates: list[URIRef],
        doc_namespace: str,
        predicate_info: dict[URIRef, PredicateMetadata],
    ) -> URIRef:
        """Create a canonical URI for a group of similar predicates.

        Args:
            similar_predicates: List of similar predicate URIs.
            doc_namespace: The document namespace to use.
            predicate_info: Dictionary mapping predicate URIs to their metadata.

        Returns:
            URIRef: The canonical URI for the group.
        """

        # Choose predicate with the most complete information
        def info_completeness(pred: URIRef) -> int:
            info = predicate_info.get(pred, PredicateMetadata(local_name=""))
            return sum(
                1
                for v in [info.label, info.comment, info.domain, info.range]
                if v is not None
            )

        best_pred = max(similar_predicates, key=info_completeness)
        best_info = predicate_info.get(best_pred, PredicateMetadata(local_name=""))
        local_name = best_info.local_name or derive_ontology_id(best_pred)

        # Create canonical URI in document namespace
        clean_local_name = self._clean_local_name(local_name)
        return URIRef(f"{doc_namespace}{clean_local_name}")

    def _clean_local_name(self, local_name: str) -> str:
        """Clean a local name for use in URIs."""
        import re

        # Replace spaces and special characters with underscores
        cleaned = re.sub(r"[^\w\-.]", "_", local_name)
        # Remove consecutive underscores
        cleaned = re.sub(r"_+", "_", cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip("_")
        return cleaned or "entity"

    def extract_predicate_info(
        self, graph: RDFGraph
    ) -> dict[URIRef, PredicateMetadata]:
        """Extract predicate information including labels, domains, and ranges.

        Args:
            graph: The RDF graph to process.

        Returns:
            dict[URIRef, PredicateMetadata]: Dictionary mapping predicate URIs to their metadata.
        """
        predicate_info = {}
        namespaces = dict(graph.namespaces())

        # Collect all predicates used in triples
        all_predicates = set()
        for _, pred, _ in graph:
            if isinstance(pred, URIRef):
                all_predicates.add(pred)

        # Initialize metadata for all predicates
        for pred in all_predicates:
            full_uri, local_name = self.normalize_uri(pred, namespaces)
            uri_ref = URIRef(full_uri)
            predicate_info[uri_ref] = PredicateMetadata(local_name=local_name)

        # Collect metadata for predicates
        for subj, pred, obj in graph:
            if not isinstance(subj, URIRef):
                continue

            full_subj_uri, _ = self.normalize_uri(subj, namespaces)
            norm_subj = URIRef(full_subj_uri)

            if norm_subj not in predicate_info:
                continue

            if pred == RDF.type and obj == RDF.Property:
                predicate_info[norm_subj].is_explicit_property = True
            elif pred == RDFS.label and isinstance(obj, Literal):
                predicate_info[norm_subj].label = str(obj)
            elif pred == RDFS.comment and isinstance(obj, Literal):
                predicate_info[norm_subj].comment = str(obj)
            elif pred == RDFS.domain:
                predicate_info[norm_subj].domain = obj
            elif pred == RDFS.range:
                predicate_info[norm_subj].range = obj

        return predicate_info

    def find_similar_predicates(
        self, predicates_with_info: dict[URIRef, PredicateMetadata]
    ) -> list[list[URIRef]]:
        """Group similar predicates based on string similarity and domain/range compatibility.

        Args:
            predicates_with_info: Dictionary mapping predicate URIs to their metadata.

        Returns:
            list[list[URIRef]]: Groups of similar predicates.
        """
        # Create lookup structures for optimization
        local_name_groups = defaultdict(list)
        label_groups = defaultdict(list)

        for predicate, info in predicates_with_info.items():
            # Group by normalized local name
            normalized_local = info.local_name.lower().strip()
            if normalized_local:
                local_name_groups[normalized_local].append(predicate)

            # Group by normalized label
            if info.label:
                normalized_label = info.label.lower().strip()
                label_groups[normalized_label].append(predicate)

        predicate_groups = []
        processed = set()

        # First pass: exact local name matches with domain/range compatibility
        for local_name, predicates in local_name_groups.items():
            if len(predicates) > 1:
                compatible_group = self._filter_by_domain_range_compatibility(
                    predicates, predicates_with_info
                )
                if len(compatible_group) > 1:
                    predicate_groups.append(compatible_group)
                    processed.update(compatible_group)

        # Second pass: exact label matches with domain/range compatibility
        for label, predicates in label_groups.items():
            unprocessed_predicates = [p for p in predicates if p not in processed]
            if len(unprocessed_predicates) > 1:
                compatible_group = self._filter_by_domain_range_compatibility(
                    unprocessed_predicates, predicates_with_info
                )
                if len(compatible_group) > 1:
                    predicate_groups.append(compatible_group)
                    processed.update(compatible_group)

        # Third pass: fuzzy label matching for remaining predicates
        remaining_predicates = [
            p for p in predicates_with_info.keys() if p not in processed
        ]
        if len(remaining_predicates) > 1:
            fuzzy_groups = self._find_fuzzy_similar_predicates(
                remaining_predicates, predicates_with_info
            )
            predicate_groups.extend(fuzzy_groups)

        return predicate_groups

    def _filter_by_domain_range_compatibility(
        self, predicates: list[URIRef], predicate_info: dict[URIRef, PredicateMetadata]
    ) -> list[URIRef]:
        """Filter predicates by domain/range compatibility."""
        if len(predicates) <= 1:
            return predicates

        # Group by domain/range signature
        signature_groups = defaultdict(list)

        for pred in predicates:
            info = predicate_info.get(pred, PredicateMetadata(local_name=""))
            # Create signature from domain and range (None values are compatible with anything)
            signature = (info.domain, info.range)
            signature_groups[signature].append(pred)

        # Find the largest compatible group
        # Predicates are compatible if their domains and ranges match or one is None
        all_compatible = []

        for signature, group in signature_groups.items():
            if len(group) > len(all_compatible):
                all_compatible = group
            elif len(group) == len(all_compatible):
                # If same size, prefer group with more specific domain/range info
                current_specificity = sum(1 for x in signature if x is not None)

                # Get the signature for the best group
                best_predicate = all_compatible[0]
                best_predicate_info = predicate_info.get(
                    best_predicate, PredicateMetadata(local_name="")
                )
                best_signature = (
                    best_predicate_info.domain or None,
                    best_predicate_info.range or None,
                )
                best_group_signature = signature_groups.get(
                    best_signature, (None, None)
                )
                best_specificity = sum(1 for x in best_group_signature if x is not None)
                if current_specificity > best_specificity:
                    all_compatible = group

        # Also include predicates with None domains/ranges (they're compatible with anything)
        for signature, group in signature_groups.items():
            if signature != (
                predicate_info.get(
                    all_compatible[0], PredicateMetadata(local_name="")
                ).domain,
                predicate_info.get(
                    all_compatible[0], PredicateMetadata(local_name="")
                ).range,
            ):
                # Check if compatible (None values are wildcards)
                if self._domains_ranges_compatible(
                    signature,
                    (
                        predicate_info.get(
                            all_compatible[0], PredicateMetadata(local_name="")
                        ).domain,
                        predicate_info.get(
                            all_compatible[0], PredicateMetadata(local_name="")
                        ).range,
                    ),
                ):
                    all_compatible.extend(group)

        return all_compatible

    def _domains_ranges_compatible(
        self,
        sig1: tuple[URIRef | None, URIRef | None],
        sig2: tuple[URIRef | None, URIRef | None],
    ) -> bool:
        """Check if two domain/range signatures are compatible."""
        domain1, range1 = sig1
        domain2, range2 = sig2

        domain_compatible = domain1 == domain2 or domain1 is None or domain2 is None
        range_compatible = range1 == range2 or range1 is None or range2 is None

        return domain_compatible and range_compatible

    def _find_fuzzy_similar_predicates(
        self, predicates: list[URIRef], predicate_info: dict[URIRef, PredicateMetadata]
    ) -> list[list[URIRef]]:
        """Find fuzzy similar predicates among the remaining predicates."""
        predicate_groups = []
        processed = set()

        for i, pred1 in enumerate(predicates):
            if pred1 in processed:
                continue

            similar_group = [pred1]
            info1 = predicate_info[pred1]
            processed.add(pred1)

            if not info1.label:  # Skip predicates without labels
                continue

            label1 = info1.label.lower().strip()

            for pred2 in predicates[i + 1 :]:
                if pred2 in processed:
                    continue

                info2 = predicate_info[pred2]

                if not info2.label:  # Skip predicates without labels
                    continue

                label2 = info2.label.lower().strip()

                # Check domain/range compatibility
                if not self._check_domain_range_compatibility(info1, info2):
                    continue

                # Calculate label similarity
                similarity = fuzz.ratio(label1, label2)

                if similarity >= self.similarity_threshold:
                    similar_group.append(pred2)
                    processed.add(pred2)

            if len(similar_group) > 1:
                predicate_groups.append(similar_group)

        return predicate_groups

    def _check_domain_range_compatibility(
        self, info1: PredicateMetadata, info2: PredicateMetadata
    ) -> bool:
        """Check if two predicates have compatible domains and ranges."""
        domain_compatible = (
            info1.domain == info2.domain or info1.domain is None or info2.domain is None
        )
        range_compatible = (
            info1.range == info2.range or info1.range is None or info2.range is None
        )
        return domain_compatible and range_compatible
