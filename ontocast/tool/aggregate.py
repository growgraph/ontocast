"""Graph aggregation tools for OntoCast.

This module provides functionality for aggregating and disambiguating RDF graphs
from multiple chunks, handling entity and predicate disambiguation, and ensuring
consistent namespace usage across the aggregated graph.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

from rdflib import Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from ontocast.onto import PROV, Chunk, RDFGraph
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
    """

    def __init__(
        self, similarity_threshold: float = 85.0, semantic_threshold: float = 90.0
    ):
        """Initialize the chunk RDF graph aggregator.

        Args:
            similarity_threshold: Threshold for considering entities similar
                (default: 85.0).
            semantic_threshold: Higher threshold for semantic similarity
                (default: 90.0).
        """
        self.disambiguator = EntityDisambiguator(
            similarity_threshold, semantic_threshold
        )

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

        aggregated_graph = RDFGraph()

        # Ensure doc_namespace ends with appropriate separator
        if not doc_namespace.endswith(("/", "#")):
            doc_namespace = doc_namespace + "/"

        # Collect all namespaces from all chunks
        all_namespaces = {}
        chunk_namespaces: Set[str] = set()
        for chunk in chunks:
            if chunk.graph is None:
                continue
            chunk_namespaces.add(chunk.namespace)
            for prefix, uri in chunk.graph.namespaces():
                if prefix not in all_namespaces:
                    all_namespaces[prefix] = uri
                elif all_namespaces[prefix] != uri:
                    # If same prefix but different URI, create a new prefix
                    new_prefix = f"{prefix}_{len(all_namespaces)}"
                    all_namespaces[new_prefix] = uri

        # Bind all namespaces to the aggregated graph
        for prefix, uri in all_namespaces.items():
            aggregated_graph.bind(prefix, uri)
        aggregated_graph.bind("prov", PROV)
        aggregated_graph.bind("cd", doc_namespace)

        # Create a mapping of URIs to their canonical form
        uri_mapping = {}
        for prefix, uri in all_namespaces.items():
            uri_mapping[uri] = uri  # Preserve external namespaces

        # Collect all entities and their labels across chunks
        all_entities_with_labels: Dict[URIRef, EntityMetadata] = {}
        chunk_entity_mapping = {}

        # Collect all predicates and their info across chunks
        all_predicates_with_info: Dict[URIRef, PredicateMetadata] = {}
        chunk_predicate_mapping = {}

        # Track entity-type relationships for better disambiguation
        entity_types = defaultdict(set)

        # First pass: collect all entities and predicates
        for chunk in chunks:
            chunk_id = chunk.hid
            logger.info(f"Processing chunk {chunk_id} with namespace {chunk.namespace}")

            # Entity disambiguation
            entities_labels = self.disambiguator.extract_entity_labels(chunk.graph)
            chunk_entity_mapping[chunk_id] = entities_labels
            all_entities_with_labels.update(entities_labels)

            # Collect type information for entities
            for subj, pred, obj in chunk.graph:
                if (
                    pred == RDF.type
                    and isinstance(subj, URIRef)
                    and isinstance(obj, URIRef)
                ):
                    entity_types[subj].add(obj)

            # Predicate disambiguation
            predicates_info = self.disambiguator.extract_predicate_info(chunk.graph)
            chunk_predicate_mapping[chunk_id] = predicates_info

            # Merge predicate info, preferring more complete information
            for pred, info in predicates_info.items():
                if pred not in all_predicates_with_info:
                    all_predicates_with_info[pred] = info
                else:
                    # Merge info, preferring non-None values and more complete data
                    existing_info = all_predicates_with_info[pred]
                    for key in ["label", "comment", "domain", "range"]:
                        if (
                            getattr(existing_info, key) is None
                            and getattr(info, key) is not None
                        ):
                            setattr(existing_info, key, getattr(info, key))
                        elif (
                            getattr(existing_info, key) is not None
                            and getattr(info, key) is not None
                            and isinstance(getattr(info, key), str)
                            and len(str(getattr(info, key)))
                            > len(str(getattr(existing_info, key)))
                        ):
                            # Prefer longer, more descriptive values
                            setattr(existing_info, key, getattr(info, key))

                    # If either source has explicit property declaration, keep it
                    if info.is_explicit_property:
                        existing_info.is_explicit_property = True

        # Enhanced similarity detection with type information
        similar_entity_groups = self.disambiguator.find_similar_entities(
            all_entities_with_labels, entity_types
        )

        # Optionally collect preferred ontology namespaces (e.g., from 'fca', etc.)
        # If you already know them (from selected ontology), pass them in externally.
        preferred_namespaces: Set[str] = set()
        for prefix, uri in all_namespaces.items():
            # Heuristic: treat non-document, non-chunk namespaces as ontology namespaces
            if str(uri) != doc_namespace and not any(
                str(uri).startswith(ns) for ns in chunk_namespaces
            ):
                preferred_namespaces.add(str(uri))

        # Find similar predicates across chunks
        similar_predicate_groups = self.disambiguator.find_similar_predicates(
            all_predicates_with_info
        )

        # Create entity mapping (original -> canonical) with document namespace
        entity_mapping = {}
        canonical_entities = set()

        for group in similar_entity_groups:
            canonical_uri = self.disambiguator.create_canonical_iri(
                group, doc_namespace, all_entities_with_labels, preferred_namespaces
            )
            # Ensure uniqueness of canonical URIs
            base_canonical = canonical_uri
            counter = 1
            while canonical_uri in canonical_entities:
                local_name = str(base_canonical).split(doc_namespace)[-1]
                canonical_uri = URIRef(f"{doc_namespace}{local_name}_{counter}")
                counter += 1

            canonical_entities.add(canonical_uri)
            for entity in group:
                entity_mapping[entity] = canonical_uri

        def _clean_name(name: str) -> str:
            import re

            s = re.sub(r"[^\w\-.]", "_", name)
            s = re.sub(r"_+", "_", s).strip("_")
            return s or "entity"

        entities_local = {}
        for ent, meta in all_entities_with_labels.items():
            entities_local[ent] = (
                meta.local_name or str(ent).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            )

        for ent in list(all_entities_with_labels.keys()):
            ent_str = str(ent)
            if ent not in entity_mapping:
                # Skip ontology entities entirely — keep their URIs intact
                if any(ent_str.startswith(ns) for ns in preferred_namespaces):
                    continue
                # Only map chunk-local entities into the document namespace
                if any(ent_str.startswith(ns) for ns in chunk_namespaces):
                    local = _clean_name(entities_local.get(ent, "entity"))
                    candidate = URIRef(f"{doc_namespace}{local}")
                    counter = 1
                    while candidate in canonical_entities:
                        candidate = URIRef(f"{doc_namespace}{local}_{counter}")
                        counter += 1
                    canonical_entities.add(candidate)
                    entity_mapping[ent] = candidate

        # Create predicate mapping (original -> canonical) with document namespace
        predicate_mapping = {}
        canonical_predicates = set()

        for group in similar_predicate_groups:
            canonical_uri = self.disambiguator.create_canonical_predicate(
                group, doc_namespace, all_predicates_with_info
            )
            # Ensure uniqueness of canonical URIs
            base_canonical = canonical_uri
            counter = 1
            while canonical_uri in canonical_predicates:
                local_name = str(base_canonical).split(doc_namespace)[-1]
                canonical_uri = URIRef(f"{doc_namespace}{local_name}_{counter}")
                counter += 1

            canonical_predicates.add(canonical_uri)
            for predicate in group:
                predicate_mapping[predicate] = canonical_uri

        predicates_local = {}
        for p, info in all_predicates_with_info.items():
            predicates_local[p] = (
                info.local_name or str(p).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            )

        for p, info in list(all_predicates_with_info.items()):
            p_str = str(p)
            if p not in predicate_mapping and any(
                p_str.startswith(ns) for ns in chunk_namespaces
            ):
                local = _clean_name(predicates_local.get(p, "predicate"))
                candidate = URIRef(f"{doc_namespace}{local}")
                counter = 1
                while candidate in canonical_predicates:
                    candidate = URIRef(f"{doc_namespace}{local}_{counter}")
                    counter += 1
                canonical_predicates.add(candidate)
                predicate_mapping[p] = candidate

        # Add canonical entity and predicate metadata to the graph
        self._add_canonical_metadata(
            aggregated_graph,
            entity_mapping,
            predicate_mapping,
            all_entities_with_labels,
            all_predicates_with_info,
            entity_types,
        )

        # Process each chunk graph
        for chunk in chunks:
            chunk_iri = URIRef(chunk.iri)
            chunk_ns = chunk.namespace  # e.g., https://example.com/doc/.../chunk/<hid>/
            logger.debug(f"Processing triples from chunk {chunk_iri}")

            # Add provenance information
            aggregated_graph.add((chunk_iri, RDF.type, PROV.Entity))
            aggregated_graph.add(
                (chunk_iri, PROV.wasPartOf, URIRef(doc_namespace.rstrip("#/")))
            )

            # Add triples with entity and predicate disambiguation
            for subj, pred, obj in chunk.graph:
                # Skip if the subject is the chunk IRI itself
                if subj == chunk_iri:
                    continue

                # Subject: only map if from the chunk namespace
                if isinstance(subj, URIRef) and str(subj).startswith(chunk_ns):
                    new_subj = entity_mapping.get(subj, subj)
                else:
                    new_subj = subj

                # Predicate: only map if from the chunk namespace
                if isinstance(pred, URIRef) and str(pred).startswith(chunk_ns):
                    new_pred = predicate_mapping.get(pred, pred)
                else:
                    new_pred = pred

                # Object:
                # - If rdf:type, never map the object (keep ontology class as-is)
                # - Else, only map if from the chunk namespace
                if new_pred == RDF.type and isinstance(obj, URIRef):
                    new_obj = obj
                else:
                    if isinstance(obj, URIRef) and str(obj).startswith(chunk_ns):
                        new_obj = entity_mapping.get(obj, obj)
                    else:
                        new_obj = obj

                # Add the triple
                aggregated_graph.add((new_subj, new_pred, new_obj))

                # Add provenance: which chunk this triple came from (only for doc entities)
                if isinstance(new_subj, URIRef) and str(new_subj).startswith(
                    doc_namespace
                ):
                    aggregated_graph.add((new_subj, PROV.wasGeneratedBy, chunk_iri))

        logger.info(
            f"Aggregated {len(chunks)} chunks into graph "
            f"with {len(aggregated_graph)} triples, "
            f"{len(entity_mapping)} entity mappings, "
            f"{len(predicate_mapping)} predicate mappings"
        )
        return aggregated_graph

    def _add_canonical_metadata(
        self,
        graph: RDFGraph,
        entity_mapping: Dict[URIRef, URIRef],
        predicate_mapping: Dict[URIRef, URIRef],
        entity_labels: Dict[URIRef, EntityMetadata],
        predicate_info: Dict[URIRef, PredicateMetadata],
        entity_types: Dict[URIRef, Set[URIRef]],
    ) -> None:
        """Add metadata for canonical entities and predicates."""
        # Process mapped entities (those that had similar counterparts)
        canonical_to_originals = defaultdict(list)
        for original, canonical in entity_mapping.items():
            canonical_to_originals[canonical].append(original)

        for canonical, originals in canonical_to_originals.items():
            # Use the best label from the group
            best_label = self._get_best_label(
                [entity_labels.get(orig) for orig in originals]
            )
            if best_label:
                graph.add((canonical, RDFS.label, Literal(best_label)))

            # Add type information
            all_types = set()
            for orig in originals:
                all_types.update(entity_types.get(orig, set()))
            for type_uri in all_types:
                graph.add((canonical, RDF.type, type_uri))

            # Link canonical to any ontology instance in the group
            for orig in originals:
                s_orig = str(orig)
                if not s_orig.startswith(graph.namespace_manager.store.namespace("cd")):
                    graph.add((canonical, OWL.sameAs, orig))

        # Process unique entities (those that didn't have similar counterparts)
        processed_entities = set(entity_mapping.keys())

        # Get all unique entities from both labels and types
        all_entities = set(entity_labels.keys()) | set(entity_types.keys())

        for entity in all_entities:
            if entity not in processed_entities:
                # Add label if available
                if entity in entity_labels and entity_labels[entity].label is not None:
                    graph.add(
                        (entity, RDFS.label, Literal(entity_labels[entity].label))
                    )
                # Add type information
                if entity in entity_types:
                    for type_uri in entity_types[entity]:
                        graph.add((entity, RDF.type, type_uri))

        # Process mapped predicates (those that had similar counterparts)
        canonical_pred_to_originals = defaultdict(list)
        for original, canonical in predicate_mapping.items():
            # Only process predicates that use our document namespace
            if str(canonical).startswith(graph.namespace_manager.store.namespace("cd")):
                canonical_pred_to_originals[canonical].append(original)

        for canonical, originals in canonical_pred_to_originals.items():
            # Merge the best information from all original predicates
            merged_info = self._merge_predicate_info(
                [predicate_info.get(orig) for orig in originals]
            )

            if merged_info.label:
                graph.add((canonical, RDFS.label, Literal(merged_info.label)))
            if merged_info.comment:
                graph.add((canonical, RDFS.comment, Literal(merged_info.comment)))
            if merged_info.domain:
                graph.add((canonical, RDFS.domain, merged_info.domain))
            if merged_info.range:
                graph.add((canonical, RDFS.range, merged_info.range))
            if merged_info.is_explicit_property:
                graph.add((canonical, RDF.type, RDF.Property))

        # Process unique predicates (those that didn't have similar counterparts)
        processed_predicates = set(predicate_mapping.keys())
        for predicate, info in predicate_info.items():
            # Only process predicates that use our document namespace
            if str(predicate).startswith(graph.namespace_manager.store.namespace("cd")):
                if predicate not in processed_predicates:
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
        self, label_dicts: List[Optional[EntityMetadata]]
    ) -> str | None:
        """Get the best label from a list of label dictionaries."""
        labels = [d.label for d in label_dicts if d is not None and d.label is not None]
        if not labels:
            return None
        # Return the longest, most descriptive label
        return max(labels, key=len)

    def _merge_predicate_info(
        self, info_dicts: List[Optional[PredicateMetadata]]
    ) -> PredicateMetadata:
        """Merge predicate information from multiple sources."""
        merged = PredicateMetadata(local_name="", is_explicit_property=False)

        for info in info_dicts:
            if info is None:
                continue
            for key in ["label", "comment", "domain", "range"]:
                current_value = getattr(merged, key)
                new_value = getattr(info, key)
                if current_value is None and new_value is not None:
                    setattr(merged, key, new_value)
                elif (
                    current_value is not None
                    and new_value is not None
                    and isinstance(new_value, str)
                    and len(new_value) > len(str(current_value))
                ):
                    setattr(merged, key, new_value)
            if info.is_explicit_property:
                merged.is_explicit_property = True

        return merged
