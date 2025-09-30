"""Refactored Graph aggregation tools for OntoCast.

This module provides functionality for aggregating and disambiguating RDF graphs
from multiple chunks, handling entity and predicate disambiguation, and ensuring
consistent namespace usage across the aggregated graph.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

from rdflib import Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from ontocast.onto.chunk import Chunk
from ontocast.onto.constants import PROV
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.disambiguator import EntityDisambiguator
from ontocast.tool.onto import EntityMetadata, PredicateMetadata

logger = logging.getLogger(__name__)


class ChunkRDFGraphAggregator:
    """Main class for aggregating and disambiguating chunk graphs.

    This class provides functionality for combining RDF graphs from multiple chunks
    while handling entity and predicate disambiguation. It ensures consistent
    namespace usage and creates canonical URIs for similar entities and predicates.

    Attributes:
        disambiguator: Entity disambiguator instance for handling entity similarity.
        include_provenance: Whether to include detailed provenance triples.
    """

    def __init__(
        self,
        similarity_threshold: float = 85.0,
        semantic_threshold: float = 90.0,
        include_provenance: bool = False,
    ):
        """Initialize the chunk RDF graph aggregator.

        Args:
            similarity_threshold: Threshold for considering entities similar (default: 85.0).
            semantic_threshold: Higher threshold for semantic similarity (default: 90.0).
            include_provenance: Whether to include detailed provenance triples (default: False).
        """
        self.disambiguator = EntityDisambiguator(
            similarity_threshold, semantic_threshold
        )
        self.include_provenance = include_provenance

    def aggregate_graphs(self, chunks: List[Chunk], doc_namespace: str) -> RDFGraph:
        """Aggregate multiple chunk graphs with entity and predicate disambiguation.

        This method combines multiple chunk graphs into a single graph while
        handling entity and predicate disambiguation. It creates canonical URIs
        for similar entities and predicates, and ensures consistent namespace usage.

        Args:
            chunks: List of chunks to aggregate.
            doc_namespace: The document IRI to use as base for canonical URIs.

        Returns:
            RDFGraph: Aggregated graph with disambiguated entities and predicates.
        """
        logger.info(f"Aggregating {len(chunks)} chunks for document {doc_namespace}")

        if not chunks:
            return RDFGraph()

        # Initialize aggregated graph
        aggregated_graph = RDFGraph()
        doc_namespace = self._normalize_namespace(doc_namespace)

        # Setup namespaces and collect chunk info
        namespace_info = self._collect_namespace_info(
            chunks, doc_namespace, aggregated_graph
        )

        # Collect and disambiguate entities and predicates
        entity_mapping, predicate_mapping, metadata = self._create_mappings(
            chunks, doc_namespace, namespace_info
        )

        # Add metadata for canonical entities and predicates
        self._add_canonical_metadata(
            aggregated_graph, entity_mapping, predicate_mapping, metadata
        )

        # Process triples from all chunks
        self._process_chunk_triples(
            chunks,
            aggregated_graph,
            entity_mapping,
            predicate_mapping,
            doc_namespace,
            namespace_info["chunk_namespaces"],
        )

        logger.info(
            f"Aggregated {len(chunks)} chunks into graph with {len(aggregated_graph)} triples, "
            f"{len(entity_mapping)} entity mappings, {len(predicate_mapping)} predicate mappings"
        )
        return aggregated_graph

    def _normalize_namespace(self, doc_namespace: str) -> str:
        """Ensure doc_namespace ends with appropriate separator."""
        return (
            doc_namespace if doc_namespace.endswith(("/", "#")) else doc_namespace + "/"
        )

    def _collect_namespace_info(
        self, chunks: List[Chunk], doc_namespace: str, aggregated_graph: RDFGraph
    ) -> Dict[str, Union[Dict[str, str], Set[str]]]:
        """Collect and bind all namespaces from chunks."""
        all_namespaces = {}
        chunk_namespaces = set()
        preferred_namespaces = set()

        for chunk in chunks:
            if chunk.graph is None:
                continue

            chunk_namespaces.add(chunk.namespace)

            for prefix, uri in chunk.graph.namespaces():
                if prefix not in all_namespaces:
                    all_namespaces[prefix] = uri
                elif all_namespaces[prefix] != uri:
                    # Handle prefix conflicts
                    new_prefix = f"{prefix}_{len(all_namespaces)}"
                    all_namespaces[new_prefix] = uri

        # Identify preferred (ontology) namespaces
        for uri in all_namespaces.values():
            uri_str = str(uri)
            if uri_str != doc_namespace and not any(
                uri_str.startswith(ns) for ns in chunk_namespaces
            ):
                preferred_namespaces.add(uri_str)

        # Bind namespaces to aggregated graph
        for prefix, uri in all_namespaces.items():
            aggregated_graph.bind(prefix, uri)
        aggregated_graph.bind("prov", PROV)
        aggregated_graph.bind("cd", doc_namespace)

        return {
            "all_namespaces": all_namespaces,
            "chunk_namespaces": chunk_namespaces,
            "preferred_namespaces": preferred_namespaces,
        }

    def _create_mappings(
        self, chunks: List[Chunk], doc_namespace: str, namespace_info: Dict
    ) -> tuple[Dict[URIRef, URIRef], Dict[URIRef, URIRef], Dict]:
        """Create entity and predicate mappings with disambiguation."""

        # Collect all entities and predicates with metadata
        all_entities = {}
        all_predicates = {}
        entity_types = defaultdict(set)

        for chunk in chunks:
            if chunk.graph is None:
                continue

            logger.debug(
                f"Processing chunk {chunk.hid} with namespace {chunk.namespace}"
            )

            # Collect entities and types
            chunk_entities = self.disambiguator.extract_entity_labels(chunk.graph)
            all_entities.update(chunk_entities)

            # Collect type information
            for subj, pred, obj in chunk.graph:
                if (
                    pred == RDF.type
                    and isinstance(subj, URIRef)
                    and isinstance(obj, URIRef)
                ):
                    entity_types[subj].add(obj)

            # Collect predicates
            chunk_predicates = self.disambiguator.extract_predicate_info(chunk.graph)
            self._merge_predicate_info(all_predicates, chunk_predicates)

        # Create similarity groups
        entity_groups = self.disambiguator.find_similar_entities(
            all_entities, entity_types
        )
        predicate_groups = self.disambiguator.find_similar_predicates(all_predicates)

        # Create mappings
        entity_mapping = self._create_entity_mapping(
            entity_groups,
            all_entities,
            doc_namespace,
            namespace_info["preferred_namespaces"],
            namespace_info["chunk_namespaces"],
        )

        predicate_mapping = self._create_predicate_mapping(
            predicate_groups,
            all_predicates,
            doc_namespace,
            namespace_info["chunk_namespaces"],
        )

        metadata = {
            "all_entities": all_entities,
            "all_predicates": all_predicates,
            "entity_types": entity_types,
        }

        return entity_mapping, predicate_mapping, metadata

    def _create_entity_mapping(
        self,
        entity_groups: List[List[URIRef]],
        all_entities: Dict[URIRef, EntityMetadata],
        doc_namespace: str,
        preferred_namespaces: Set[str],
        chunk_namespaces: Set[str],
    ) -> Dict[URIRef, URIRef]:
        """Create mapping from original to canonical entity URIs."""
        entity_mapping = {}
        canonical_entities = set()

        # Process similar entity groups
        for group in entity_groups:
            canonical_uri = self.disambiguator.create_canonical_iri(
                group, doc_namespace, all_entities, preferred_namespaces
            )
            canonical_uri = self._ensure_unique_uri(
                canonical_uri, canonical_entities, doc_namespace
            )
            canonical_entities.add(canonical_uri)

            for entity in group:
                entity_mapping[entity] = canonical_uri

        # Process remaining individual entities from chunk namespaces
        for entity in all_entities:
            if entity not in entity_mapping:
                entity_str = str(entity)
                # Only map chunk-local entities to document namespace
                if any(entity_str.startswith(ns) for ns in chunk_namespaces):
                    local_name = self._clean_name(
                        all_entities[entity].local_name or "entity"
                    )
                    canonical_uri = URIRef(f"{doc_namespace}{local_name}")
                    canonical_uri = self._ensure_unique_uri(
                        canonical_uri, canonical_entities, doc_namespace
                    )
                    canonical_entities.add(canonical_uri)
                    entity_mapping[entity] = canonical_uri

        return entity_mapping

    def _create_predicate_mapping(
        self,
        predicate_groups: List[List[URIRef]],
        all_predicates: Dict[URIRef, PredicateMetadata],
        doc_namespace: str,
        chunk_namespaces: Set[str],
    ) -> Dict[URIRef, URIRef]:
        """Create mapping from original to canonical predicate URIs."""
        predicate_mapping = {}
        canonical_predicates = set()

        # Process similar predicate groups
        for group in predicate_groups:
            canonical_uri = self.disambiguator.create_canonical_predicate(
                group, doc_namespace, all_predicates
            )
            canonical_uri = self._ensure_unique_uri(
                canonical_uri, canonical_predicates, doc_namespace
            )
            canonical_predicates.add(canonical_uri)

            for predicate in group:
                predicate_mapping[predicate] = canonical_uri

        # Process remaining individual predicates from chunk namespaces
        for predicate in all_predicates:
            if predicate not in predicate_mapping:
                predicate_str = str(predicate)
                if any(predicate_str.startswith(ns) for ns in chunk_namespaces):
                    local_name = self._clean_name(
                        all_predicates[predicate].local_name or "predicate"
                    )
                    canonical_uri = URIRef(f"{doc_namespace}{local_name}")
                    canonical_uri = self._ensure_unique_uri(
                        canonical_uri, canonical_predicates, doc_namespace
                    )
                    canonical_predicates.add(canonical_uri)
                    predicate_mapping[predicate] = canonical_uri

        return predicate_mapping

    def _ensure_unique_uri(
        self, uri: URIRef, existing: Set[URIRef], namespace: str
    ) -> URIRef:
        """Ensure URI uniqueness by appending counter if needed."""
        base_uri = uri
        counter = 1
        while uri in existing:
            local_name = str(base_uri).split(namespace)[-1]
            uri = URIRef(f"{namespace}{local_name}_{counter}")
            counter += 1
        return uri

    def _clean_name(self, name: str) -> str:
        """Clean name for use in URIs."""
        import re

        cleaned = re.sub(r"[^\w\-.]", "_", name)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned or "entity"

    def _merge_predicate_info(
        self,
        target: Dict[URIRef, PredicateMetadata],
        source: Dict[URIRef, PredicateMetadata],
    ) -> None:
        """Merge predicate information, preferring more complete data."""
        for pred, info in source.items():
            if pred not in target:
                target[pred] = info
            else:
                existing = target[pred]
                for attr in ["label", "comment", "domain", "range"]:
                    existing_val = getattr(existing, attr)
                    new_val = getattr(info, attr)

                    if existing_val is None and new_val is not None:
                        setattr(existing, attr, new_val)
                    elif (
                        existing_val is not None
                        and new_val is not None
                        and isinstance(new_val, str)
                        and len(new_val) > len(str(existing_val))
                    ):
                        setattr(existing, attr, new_val)

                if info.is_explicit_property:
                    existing.is_explicit_property = True

    def _process_chunk_triples(
        self,
        chunks: List[Chunk],
        aggregated_graph: RDFGraph,
        entity_mapping: Dict[URIRef, URIRef],
        predicate_mapping: Dict[URIRef, URIRef],
        doc_namespace: str,
        chunk_namespaces: Set[str],
    ) -> None:
        """Process triples from all chunks with disambiguation."""
        for chunk in chunks:
            if chunk.graph is None:
                continue

            chunk_iri = URIRef(chunk.iri)

            # Add minimal provenance if requested
            if self.include_provenance:
                aggregated_graph.add((chunk_iri, RDF.type, PROV.Entity))
                aggregated_graph.add(
                    (chunk_iri, PROV.wasPartOf, URIRef(doc_namespace.rstrip("#/")))
                )

            # Process triples with mapping
            for subj, pred, obj in chunk.graph:
                # Skip chunk metadata triples
                if subj == chunk_iri:
                    continue

                # Apply mappings based on namespace
                new_subj = self._apply_mapping(subj, entity_mapping, chunk_namespaces)
                new_pred = self._apply_mapping(
                    pred, predicate_mapping, chunk_namespaces
                )

                # Special handling for rdf:type objects (preserve ontology classes)
                if new_pred == RDF.type and isinstance(obj, URIRef):
                    new_obj = obj  # Keep ontology classes unchanged
                else:
                    new_obj = self._apply_mapping(obj, entity_mapping, chunk_namespaces)

                aggregated_graph.add((new_subj, new_pred, new_obj))

                # Add selective provenance
                if (
                    self.include_provenance
                    and isinstance(new_subj, URIRef)
                    and str(new_subj).startswith(doc_namespace)
                ):
                    aggregated_graph.add((new_subj, PROV.wasGeneratedBy, chunk_iri))

    def _apply_mapping(
        self,
        uri: Union[URIRef, Literal],
        mapping: Dict[URIRef, URIRef],
        chunk_namespaces: Set[str],
    ) -> Union[URIRef, Literal]:
        """Apply mapping only if URI is from chunk namespace."""
        if not isinstance(uri, URIRef):
            return uri

        uri_str = str(uri)
        if any(uri_str.startswith(ns) for ns in chunk_namespaces):
            return mapping.get(uri, uri)
        return uri

    def _add_canonical_metadata(
        self,
        graph: RDFGraph,
        entity_mapping: Dict[URIRef, URIRef],
        predicate_mapping: Dict[URIRef, URIRef],
        metadata: Dict,
    ) -> None:
        """Add metadata for canonical entities and predicates."""
        all_entities = metadata["all_entities"]
        all_predicates = metadata["all_predicates"]
        entity_types = metadata["entity_types"]

        # Group entities by their canonical form
        canonical_to_originals = defaultdict(list)
        for original, canonical in entity_mapping.items():
            canonical_to_originals[canonical].append(original)

        # Add metadata for grouped entities
        for canonical, originals in canonical_to_originals.items():
            self._add_entity_metadata(
                graph, canonical, originals, all_entities, entity_types
            )

        # Add metadata for individual entities
        processed_entities = set(entity_mapping.keys())
        for entity in all_entities:
            if entity not in processed_entities:
                self._add_individual_entity_metadata(
                    graph, entity, all_entities, entity_types
                )

        # Similar process for predicates
        canonical_pred_to_originals = defaultdict(list)
        doc_namespace = graph.namespace_manager.store.namespace("cd")

        for original, canonical in predicate_mapping.items():
            if str(canonical).startswith(doc_namespace):
                canonical_pred_to_originals[canonical].append(original)

        for canonical, originals in canonical_pred_to_originals.items():
            merged_info = self._get_merged_predicate_info(originals, all_predicates)
            self._add_predicate_metadata(graph, canonical, merged_info)

        # Add metadata for individual predicates
        processed_predicates = set(predicate_mapping.keys())
        for predicate, info in all_predicates.items():
            if predicate not in processed_predicates and str(predicate).startswith(
                doc_namespace
            ):
                self._add_predicate_metadata(graph, predicate, info)

    def _add_entity_metadata(
        self,
        graph: RDFGraph,
        canonical: URIRef,
        originals: List[URIRef],
        all_entities: Dict[URIRef, EntityMetadata],
        entity_types: Dict[URIRef, Set[URIRef]],
    ) -> None:
        """Add metadata for a canonical entity."""
        # Best label from the group
        best_label = self._get_best_label(
            [all_entities.get(orig) for orig in originals]
        )
        if best_label:
            graph.add((canonical, RDFS.label, Literal(best_label)))

        # Collect all types
        all_types = set()
        for orig in originals:
            all_types.update(entity_types.get(orig, set()))

        for type_uri in all_types:
            graph.add((canonical, RDF.type, type_uri))

        # Link to ontology instances
        doc_namespace = graph.namespace_manager.store.namespace("cd")
        for orig in originals:
            if not str(orig).startswith(doc_namespace):
                graph.add((canonical, OWL.sameAs, orig))

    def _add_individual_entity_metadata(
        self,
        graph: RDFGraph,
        entity: URIRef,
        all_entities: Dict[URIRef, EntityMetadata],
        entity_types: Dict[URIRef, Set[URIRef]],
    ) -> None:
        """Add metadata for an individual entity."""
        if entity in all_entities and all_entities[entity].label:
            graph.add((entity, RDFS.label, Literal(all_entities[entity].label)))

        for type_uri in entity_types.get(entity, set()):
            graph.add((entity, RDF.type, type_uri))

    def _add_predicate_metadata(
        self, graph: RDFGraph, predicate: URIRef, info: PredicateMetadata
    ) -> None:
        """Add metadata for a predicate."""
        if info.label:
            graph.add((predicate, RDFS.label, Literal(info.label)))
        if info.comment:
            graph.add((predicate, RDFS.comment, Literal(info.comment)))
        if info.domain:
            graph.add((predicate, RDFS.domain, info.domain))
        if info.range:
            graph.add((predicate, RDFS.range, info.range))
        if info.is_explicit_property:
            graph.add((predicate, RDF.type, RDF.Property))

    def _get_best_label(
        self, metadata_list: List[Optional[EntityMetadata]]
    ) -> Optional[str]:
        """Get the best label from a list of entity metadata."""
        labels = [m.label for m in metadata_list if m and m.label]
        return max(labels, key=len) if labels else None

    def _get_merged_predicate_info(
        self, originals: List[URIRef], all_predicates: Dict[URIRef, PredicateMetadata]
    ) -> PredicateMetadata:
        """Merge predicate information from multiple sources."""
        merged = PredicateMetadata(local_name="", is_explicit_property=False)

        for pred in originals:
            info = all_predicates.get(pred)
            if not info:
                continue

            for attr in ["label", "comment", "domain", "range"]:
                current = getattr(merged, attr)
                new_val = getattr(info, attr)

                if current is None and new_val is not None:
                    setattr(merged, attr, new_val)
                elif (
                    current is not None
                    and new_val is not None
                    and isinstance(new_val, str)
                    and len(new_val) > len(str(current))
                ):
                    setattr(merged, attr, new_val)

            if info.is_explicit_property:
                merged.is_explicit_property = True

        return merged
