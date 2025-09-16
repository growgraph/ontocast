from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz
from rdflib import RDF, RDFS, Literal, URIRef

from ontocast.onto import RDFGraph, derive_ontology_id
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
            similarity_threshold: Threshold for considering entities similar
                (default: 85.0).
            semantic_threshold: Higher threshold for semantic similarity
                (default: 90.0).
        """
        self.similarity_threshold = similarity_threshold
        self.semantic_threshold = semantic_threshold

    def normalize_uri(self, uri: URIRef, namespaces: Dict[str, str]) -> Tuple[str, str]:
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
                full_uri = uri_str.replace(f"{prefix}:", str(namespace))
                uri_str = full_uri
                break

        # Extract local name from fragment or last path segment
        if "#" in uri_str:
            local = uri_str.rsplit("#", 1)[-1]
        else:
            trimmed = uri_str.rstrip("/")
            local = trimmed.rsplit("/", 1)[-1] if "/" in trimmed else trimmed

        return uri_str, local

    def extract_entity_labels(self, graph: RDFGraph) -> Dict[URIRef, EntityMetadata]:
        """Extract labels for entities from graph, including their local names.

        Args:
            graph: The RDF graph to process.

        Returns:
            Dict[URIRef, EntityMetadata]: Dictionary mapping entity URIs to their
                metadata.
        """
        labels = {}
        namespaces = dict(graph.namespaces())

        # First pass: collect explicit labels and comments
        for subj, pred, obj in graph:
            if (
                pred in [RDFS.label, RDFS.comment]
                and isinstance(obj, Literal)
                and isinstance(subj, URIRef)
            ):
                full_uri, local_name = self.normalize_uri(subj, namespaces)
                uri_ref = URIRef(full_uri)
                if uri_ref not in labels:
                    labels[uri_ref] = EntityMetadata(local_name=local_name)

                if pred == RDFS.label:
                    labels[uri_ref].label = str(obj)
                elif pred == RDFS.comment:
                    labels[uri_ref].comment = str(obj)

        # Second pass: collect all entities and use local name as fallback
        for subj, pred, obj in graph:
            for entity in [subj, obj]:
                if isinstance(entity, URIRef):
                    full_uri, local_name = self.normalize_uri(entity, namespaces)
                    uri_ref = URIRef(full_uri)
                    if uri_ref not in labels:
                        labels[uri_ref] = EntityMetadata(local_name=local_name)
        return labels

    def find_similar_entities(
        self,
        entities_with_labels: Dict[URIRef, EntityMetadata],
        entity_types: Dict[URIRef, Set[URIRef]] = None,
    ) -> List[List[URIRef]]:
        """Group similar entities based on string similarity, local names, and types.

        Args:
            entities_with_labels: Dictionary mapping entity URIs to their metadata.
            entity_types: Optional dictionary mapping entities to their types.

        Returns:
            List[List[URIRef]]: Groups of similar entities.
        """
        if entity_types is None:
            entity_types = {}

        entity_groups = []
        processed = set()
        entities_list = list(entities_with_labels.keys())

        for i, entity1 in enumerate(entities_list):
            if entity1 in processed:
                continue

            similar_group = [entity1]
            info1 = entities_with_labels[entity1]
            types1 = entity_types.get(entity1, set())
            processed.add(entity1)

            for j, entity2 in enumerate(entities_list[i + 1 :], i + 1):
                if entity2 in processed:
                    continue

                info2 = entities_with_labels[entity2]
                types2 = entity_types.get(entity2, set())

                # Check type compatibility - entities should share at least one type
                #                                   or have no conflicting types
                type_compatible = (
                    not types1
                    or not types2  # One has no type info
                    or bool(types1.intersection(types2))  # They share at least one type
                )

                if not type_compatible:
                    continue

                # Exact local name match (highest priority)
                if info1.local_name.lower() == info2.local_name.lower():
                    similar_group.append(entity2)
                    processed.add(entity2)
                    continue

                # Label similarity check
                label1 = info1.label.lower() if info1.label is not None else ""
                label2 = info2.label.lower() if info2.label is not None else ""

                if label1 and label2:
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

    def create_canonical_iri(
        self,
        similar_entities: List[URIRef],
        doc_namespace: str,
        entity_labels: Dict[URIRef, EntityMetadata],
        preferred_namespaces: Optional[Set[str]] = None,
    ) -> URIRef:
        """Create a canonical URI for a group of similar entities.

        Args:
            similar_entities: List of similar entity URIs.
            doc_namespace: The document namespace to use.
            entity_labels: Dictionary mapping entities to their metadata.

        Returns:
            URIRef: The canonical URI for the group.
        """
        # 1) If any entity belongs to a preferred ontology namespace, pick it as canonical

        if preferred_namespaces is not None:
            for ent in similar_entities:
                s = str(ent)
                if any(s.startswith(ns) for ns in preferred_namespaces):
                    return ent

        # 2) Otherwise, choose the entity with the best label (longest, most descriptive)
        best_entity = max(
            similar_entities,
            key=lambda e: len(
                entity_labels.get(e, EntityMetadata(local_name="")).label or ""
            ),
        )
        best_info = entity_labels.get(
            best_entity, EntityMetadata(local_name=derive_ontology_id(best_entity))
        )
        local_name = best_info.local_name

        # Clean the local name for use in URI
        clean_local_name = self._clean_local_name(local_name)
        return URIRef(f"{doc_namespace}{clean_local_name}")

    def create_canonical_predicate(
        self,
        similar_predicates: List[URIRef],
        doc_namespace: str,
        predicate_info: Dict[URIRef, PredicateMetadata],
    ) -> URIRef:
        """Create a canonical URI for a group of similar predicates.

        Args:
            similar_predicates: List of similar predicate URIs.
            doc_namespace: The document namespace to use.
            predicate_info: Dictionary mapping predicate URIs to their metadata.

        Returns:
            URIRef: The canonical URI for the group.
        """
        # Use the predicate with the most complete information
        best_pred = max(
            similar_predicates,
            key=lambda p: sum(
                1
                for v in [
                    predicate_info.get(p, PredicateMetadata(local_name="")).label,
                    predicate_info.get(p, PredicateMetadata(local_name="")).comment,
                    predicate_info.get(p, PredicateMetadata(local_name="")).domain,
                    predicate_info.get(p, PredicateMetadata(local_name="")).range,
                ]
                if v is not None
            ),
        )

        # Create new canonical URI in document namespace
        best_info = predicate_info.get(
            best_pred, PredicateMetadata(local_name=derive_ontology_id(best_pred))
        )
        local_name = best_info.local_name

        # Clean the local name for use in URI
        clean_local_name = self._clean_local_name(local_name)
        return URIRef(f"{doc_namespace}{clean_local_name}")

    def _clean_local_name(self, local_name: str) -> str:
        """Clean a local name for use in URIs."""
        # Remove or replace problematic characters
        import re

        # Replace spaces and special characters with underscores
        cleaned = re.sub(r"[^\w\-.]", "_", local_name)
        # Remove consecutive underscores
        cleaned = re.sub(r"_+", "_", cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip("_")
        return cleaned or "entity"  # Fallback if empty

    def extract_predicate_info(
        self, graph: RDFGraph
    ) -> Dict[URIRef, PredicateMetadata]:
        """Extract predicate information including labels, domains, and ranges.

        Args:
            graph: The RDF graph to process.

        Returns:
            Dict[URIRef, PredicateMetadata]: Dictionary mapping predicate URIs to
                their metadata.
        """
        predicate_info = {}
        namespaces = dict(graph.namespaces())

        # First pass: identify all predicates used in triples
        for _, pred, _ in graph:
            if isinstance(pred, URIRef):
                full_uri, local_name = self.normalize_uri(pred, namespaces)
                uri_ref = URIRef(full_uri)
                if uri_ref not in predicate_info:
                    predicate_info[uri_ref] = PredicateMetadata(local_name=local_name)

        # Second pass: collect metadata for predicates
        for subj, pred, obj in graph:
            if isinstance(subj, URIRef):
                full_subj_uri, _ = self.normalize_uri(subj, namespaces)
                norm_subj = URIRef(full_subj_uri)

                if pred == RDF.type and obj == RDF.Property:
                    if norm_subj in predicate_info:
                        predicate_info[norm_subj].is_explicit_property = True
                elif pred in [RDFS.label, RDFS.comment] and isinstance(obj, Literal):
                    if norm_subj in predicate_info:
                        if pred == RDFS.label:
                            predicate_info[norm_subj].label = str(obj)
                        else:
                            predicate_info[norm_subj].comment = str(obj)
                elif pred == RDFS.domain and norm_subj in predicate_info:
                    predicate_info[norm_subj].domain = obj
                elif pred == RDFS.range and norm_subj in predicate_info:
                    predicate_info[norm_subj].range = obj
        return predicate_info

    def find_similar_predicates(
        self, predicates_with_info: Dict[URIRef, PredicateMetadata]
    ) -> List[List[URIRef]]:
        """Group similar predicates based on string similarity and domain/range
        compatibility.

        Args:
            predicates_with_info: Dictionary mapping predicate URIs to their metadata.

        Returns:
            List[List[URIRef]]: Groups of similar predicates.
        """
        predicate_groups = []
        processed = set()
        predicates_list = list(predicates_with_info.keys())

        for i, pred_a in enumerate(predicates_list):
            if pred_a in processed:
                continue

            similar_group = [pred_a]
            info1 = predicates_with_info[pred_a]
            processed.add(pred_a)

            for j, pred_b in enumerate(predicates_list[i + 1 :], i + 1):
                if pred_b in processed:
                    continue

                info2 = predicates_with_info[pred_b]

                # Exact local name match
                if info1.local_name.lower() == info2.local_name.lower():
                    # Still check domain/range compatibility for exact matches
                    if self._check_domain_range_compatibility(info1, info2):
                        similar_group.append(pred_b)
                        processed.add(pred_b)
                    continue

                # Check label similarity
                if info1.label is not None and info2.label is not None:
                    label_similarity = fuzz.ratio(
                        info1.label.lower(), info2.label.lower()
                    )

                    # Check domain/range compatibility
                    domain_range_compatible = self._check_domain_range_compatibility(
                        info1, info2
                    )

                    if (
                        label_similarity >= self.similarity_threshold
                        and domain_range_compatible
                    ):
                        similar_group.append(pred_b)
                        processed.add(pred_b)

            if len(similar_group) > 1:
                predicate_groups.append(similar_group)

        return predicate_groups

    def _check_domain_range_compatibility(
        self, info1: PredicateMetadata, info2: PredicateMetadata
    ) -> bool:
        """Check if two predicates have compatible domains and ranges."""
        # Compatible if they match or one is None (not specified)
        domain_compatible = (
            info1.domain == info2.domain or info1.domain is None or info2.domain is None
        )
        range_compatible = (
            info1.range == info2.range or info1.range is None or info2.range is None
        )
        return domain_compatible and range_compatible
