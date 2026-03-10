"""Embedding-based RDF graph aggregator.

This module provides the main aggregator class that orchestrates entity
disambiguation using embedding-based clustering.

Pipeline:
1. Collect entities from all content units
2. Normalize entities: e -> r(e) (string representation with semantic context)
3. Embed in parallel: r(e) -> v(e) (embedding vectors)
4. Cluster by similarity: v(e) -> g(e) (groups of similar entities)
5. Select representatives: g(e) -> e_rep (best entity per group)
6. Build normalised URIs: e_rep -> e' (PascalCase/camelCase under DEFAULT_IRI)
7. Rewrite graphs: apply mapping e -> e' to all triples
"""

import logging
from difflib import SequenceMatcher
from enum import StrEnum

from rdflib import URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from ontocast.onto.constants import DEFAULT_IRI, PROV, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph

from .clustering import ClusterRepresentativeSelector, EntityClusterer
from .normalizer import EntityNormalizer, EntityRepresentation
from .rewriter import GraphRewriter
from .uri_builder import EntityRole, URIBuilder

logger = logging.getLogger(__name__)


class EntityClassification(StrEnum):
    """Classification of entities during aggregation."""

    FACT = "fact"
    KNOWN_ONTOLOGY = "known_ontology"
    TENTATIVE_ONTOLOGY = "tentative_ontology"


_STANDARD_NAMESPACES = (
    str(RDF),
    str(RDFS),
    str(OWL),
    str(XSD),
    str(SCHEMA),
    str(PROV),
)


class EmbeddingBasedAggregator:
    """Main aggregator using embedding-based entity disambiguation.

    Pipeline stages:
    1. Entity normalisation (with semantic context)
    2. Parallel embedding
    3. Similarity-based clustering
    4. Representative selection (prefer ontology, then simplicity)
    5. URI normalisation (PascalCase/camelCase under DEFAULT_IRI)
    6. Graph rewriting

    ContentUnit types are handled as follows:
    - ``facts``: entities under ``base_iri`` are normalised.
    - ``ontology``: all other entities are considered ontology entities and preserved.
    """

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.80,
        add_sameas_links: bool = True,
        base_iri: str = DEFAULT_IRI,
    ):
        """Initialise the embedding-based aggregator.

        Args:
            embedding_model: Name of sentence transformer model.
            similarity_threshold: Cosine similarity threshold for clustering (0-1).
            add_sameas_links: Whether to add owl:sameAs for merged entities.
            base_iri: Base IRI for fact entity URIs (default: DEFAULT_IRI).
                Entities under this namespace are facts; everything else is
                treated as an ontology entity and left unchanged.
        """
        self.base_iri = base_iri

        # Pipeline components
        self.normalizer = EntityNormalizer(facts_iri=self.base_iri)
        self.clusterer = EntityClusterer(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
        )
        self.selector = ClusterRepresentativeSelector()
        self.uri_builder = URIBuilder(base_iri=self.base_iri)
        self.rewriter = GraphRewriter(add_sameas_links=add_sameas_links)

    @staticmethod
    def _entity_in_namespace(entity: URIRef, namespace: URIRef | str | None) -> bool:
        """Return True when *entity* is under the provided namespace."""
        if namespace is None:
            return False
        entity_str = str(entity)
        namespace_str = str(namespace)

        # Accept exact prefix namespaces (e.g. ``.../facts`` used with Turtle
        # ``@prefix cd: <.../facts>`` → ``.../factsConviction1``) and slash/hash
        # namespace variants.
        if entity_str.startswith(namespace_str):
            return True

        slash_variant = namespace_str.rstrip("/") + "/"
        hash_variant = namespace_str.rstrip("#") + "#"
        return entity_str.startswith(slash_variant) or entity_str.startswith(
            hash_variant
        )

    def _is_fact_entity_in_unit(self, entity: URIRef, unit: ContentUnit) -> bool:
        """Classify whether an entity should be treated as a fact in this unit.

        Facts are entities in either:
        - the configured base facts namespace (``base_iri``), or
        - the unit document namespace (``unit.doc_iri``).
        """
        return self._entity_in_namespace(
            entity, self.base_iri
        ) or self._entity_in_namespace(entity, unit.doc_iri)

    @staticmethod
    def _is_standard_ontology_entity(entity: URIRef) -> bool:
        """Return True for entities from built-in standard RDF vocabularies."""
        entity_str = str(entity)
        return any(entity_str.startswith(prefix) for prefix in _STANDARD_NAMESPACES)

    def _build_known_ontology_entities(
        self, ontology_graph: RDFGraph | None
    ) -> set[URIRef]:
        """Build a set of known ontology entities from ontology and std vocabularies."""
        known_entities: set[URIRef] = set()

        if ontology_graph is not None:
            for s, p, o in ontology_graph:
                if isinstance(s, URIRef):
                    known_entities.add(s)
                if isinstance(p, URIRef):
                    known_entities.add(p)
                if isinstance(o, URIRef):
                    known_entities.add(o)

        return known_entities

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in text.split() if len(token) > 2}

    @staticmethod
    def _role_key(representation: EntityRepresentation) -> str:
        role = (
            representation.role
            if representation.role is not None
            else EntityRole.INSTANCE
        )
        return str(role)

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _are_roles_compatible(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> bool:
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        return self._role_key(left_rep) == self._role_key(right_rep)

    def _are_types_compatible(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> bool:
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        left_types = set(left_rep.types)
        right_types = set(right_rep.types)
        if not left_types or not right_types:
            return True
        return bool(left_types & right_types)

    def _are_lexical_aliases(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> bool:
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        if left_rep.normal_form == right_rep.normal_form:
            return True

        left_label_tokens = {
            self.normalizer.normalize_string(label)
            for label in left_rep.labels
            if label.strip()
        }
        right_label_tokens = {
            self.normalizer.normalize_string(label)
            for label in right_rep.labels
            if label.strip()
        }
        if left_label_tokens & right_label_tokens:
            return True
        if left_label_tokens and right_label_tokens:
            max_label_overlap = 0.0
            for left_label in left_label_tokens:
                left_tokens = self._tokenize(left_label)
                for right_label in right_label_tokens:
                    right_tokens = self._tokenize(right_label)
                    overlap = self._jaccard(left_tokens, right_tokens)
                    max_label_overlap = max(max_label_overlap, overlap)
            if max_label_overlap >= 0.2:
                return True

        ratio = SequenceMatcher(
            None, left_rep.normal_form, right_rep.normal_form
        ).ratio()
        if ratio >= 0.82:
            return True

        left_tokens = self._tokenize(left_rep.normal_form)
        right_tokens = self._tokenize(right_rep.normal_form)
        if len(left_tokens) >= 2 and len(right_tokens) >= 2:
            if self._jaccard(left_tokens, right_tokens) >= 0.75:
                return True

        return False

    def _can_merge_as_identity(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> bool:
        return (
            self._are_roles_compatible(left, right, representations)
            and self._are_types_compatible(left, right, representations)
            and self._are_lexical_aliases(left, right, representations)
        )

    def _cluster_entities_by_role(
        self, representations: dict[URIRef, EntityRepresentation]
    ) -> list[list[URIRef]]:
        grouped_entities: dict[str, dict[URIRef, EntityRepresentation]] = {}
        for entity, representation in representations.items():
            grouped_entities.setdefault(self._role_key(representation), {})[entity] = (
                representation
            )

        all_clusters: list[list[URIRef]] = []
        for role_representations in grouped_entities.values():
            role_clusters, _ = self.clusterer.cluster_entities(role_representations)
            all_clusters.extend(role_clusters)
        return all_clusters

    def _select_ontology_anchor_candidates(
        self,
        tentative_entities: list[URIRef],
        tentative_representations: dict[URIRef, EntityRepresentation],
        ontology_graph: RDFGraph | None,
        known_ontology_entities: set[URIRef],
    ) -> set[URIRef]:
        """Pick a small ontology anchor set for tentative ontology-like entities."""
        if (
            ontology_graph is None
            or not tentative_entities
            or not known_ontology_entities
        ):
            return set()

        ontology_entities = [
            entity
            for entity in known_ontology_entities
            if not self._is_standard_ontology_entity(entity)
        ]
        if not ontology_entities:
            return set()

        ontology_graphs = {entity: ontology_graph for entity in ontology_entities}
        ontology_representations = self.normalizer.create_representations_batch(
            ontology_entities, ontology_graphs
        )

        token_index: dict[str, set[URIRef]] = {}
        for entity, representation in ontology_representations.items():
            for token in self._tokenize(representation.representation):
                token_index.setdefault(token, set()).add(entity)

        selected: set[URIRef] = set()
        for tentative_entity in tentative_entities:
            tentative_representation = tentative_representations.get(tentative_entity)
            if tentative_representation is None:
                continue
            tentative_tokens = self._tokenize(tentative_representation.representation)
            if not tentative_tokens:
                continue

            candidate_pool: set[URIRef] = set()
            for token in tentative_tokens:
                candidate_pool.update(token_index.get(token, set()))

            if not candidate_pool:
                continue

            scored: list[tuple[int, URIRef]] = []
            for candidate in candidate_pool:
                candidate_representation = ontology_representations.get(candidate)
                if candidate_representation is None:
                    continue
                candidate_tokens = self._tokenize(
                    candidate_representation.representation
                )
                overlap = len(tentative_tokens & candidate_tokens)
                if overlap >= 2:
                    scored.append((overlap, candidate))

            scored.sort(key=lambda item: item[0], reverse=True)
            for _, candidate in scored[:3]:
                selected.add(candidate)

        return selected

    def _classify_entity_for_unit(
        self,
        entity: URIRef,
        unit: ContentUnit,
        known_ontology_entities: set[URIRef],
    ) -> EntityClassification:
        """Classify an entity as fact, known ontology, or tentative ontology."""
        if unit.type == OutputType.ONTOLOGIES:
            return EntityClassification.KNOWN_ONTOLOGY

        if self._is_fact_entity_in_unit(entity, unit):
            return EntityClassification.FACT

        if entity in known_ontology_entities or self._is_standard_ontology_entity(
            entity
        ):
            return EntityClassification.KNOWN_ONTOLOGY

        return EntityClassification.TENTATIVE_ONTOLOGY

    @staticmethod
    def _classification_priority(classification: EntityClassification) -> int:
        """Return priority for multi-unit classification merging."""
        if classification == EntityClassification.KNOWN_ONTOLOGY:
            return 3
        if classification == EntityClassification.TENTATIVE_ONTOLOGY:
            return 2
        return 1

    @staticmethod
    def _extract_local_name(entity: URIRef) -> str:
        entity_str = str(entity)
        if "#" in entity_str:
            return entity_str.rsplit("#", 1)[-1]
        trimmed = entity_str.rstrip("/")
        return trimmed.rsplit("/", 1)[-1] if "/" in trimmed else trimmed

    def _to_doc_namespace_entity(
        self,
        entity: URIRef,
        current_target: URIRef,
        doc_iri: URIRef | None,
    ) -> URIRef:
        """Force a mapped fact entity into its document namespace."""
        if doc_iri is None:
            return current_target
        if self._entity_in_namespace(current_target, doc_iri):
            return current_target

        target_local = self._extract_local_name(current_target)
        if not target_local:
            target_local = self._extract_local_name(entity)
        if not target_local:
            target_local = "Entity"
        return URIRef(f"{str(doc_iri).rstrip('/')}/{target_local}")

    def _force_fact_targets_to_doc_namespace(
        self,
        final_mapping: dict[URIRef, URIRef],
        source_entities: set[URIRef],
        entity_doc_iris: dict[URIRef, URIRef],
        entity_classification: dict[URIRef, EntityClassification],
    ) -> None:
        """Ensure all fact entities map to their source document namespace."""
        for entity in source_entities:
            if entity_classification.get(entity) != EntityClassification.FACT:
                continue
            current_target = final_mapping.get(entity, entity)
            final_mapping[entity] = self._to_doc_namespace_entity(
                entity=entity,
                current_target=current_target,
                doc_iri=entity_doc_iris.get(entity),
            )

    def _remove_base_namespace_sameas_leaks(self, graph: RDFGraph) -> None:
        """Drop owl:sameAs links that expose base facts namespace IRIs."""
        to_remove = []
        for s, p, o in graph.triples((None, OWL.sameAs, None)):
            if not isinstance(s, URIRef) or not isinstance(o, URIRef):
                continue
            if self._entity_in_namespace(s, self.base_iri) or self._entity_in_namespace(
                o, self.base_iri
            ):
                to_remove.append((s, p, o))

        for triple in to_remove:
            graph.remove(triple)

    def _collect_all_entities(
        self,
        units: list[ContentUnit],
        known_ontology_entities: set[URIRef] | None = None,
    ) -> tuple[
        list[URIRef],
        set[URIRef],
        dict[URIRef, RDFGraph],
        dict[URIRef, URIRef],
        dict[URIRef, EntityClassification],
    ]:
        """Collect all entities from all content unit graphs.

        Each entity is associated with the graph it was found in and the
        ``doc_iri`` of the :class:`ContentUnit` that produced it.  When an
        entity appears in several units the *last-seen* ``doc_iri`` wins (in
        practice most pipelines aggregate chunks of the same document, so all
        ``doc_iri`` values are identical).

        Args:
            units: List of content units to aggregate.

        Returns:
            Tuple of (
                entities,
                entity_to_graph,
                entity_to_doc_iri,
                entity_to_is_ontology,
            ).
        """
        entities: set[URIRef] = set()
        source_entities: set[URIRef] = set()
        entity_graphs: dict[URIRef, RDFGraph] = {}
        entity_doc_iris: dict[URIRef, URIRef] = {}
        entity_classification: dict[URIRef, EntityClassification] = {}
        known_entities = known_ontology_entities or set()

        for unit in units:
            if unit.graph is None:
                continue
            # Keep collection in the same URI space that rewrite/merge consumes
            # (unit.graph). Using graph_absolute here causes mapping keys to miss
            # during rewrite, because unit.graph still contains the original terms.
            for s, p, o in unit.graph:
                if isinstance(s, URIRef):
                    entities.add(s)
                    source_entities.add(s)
                    entity_graphs[s] = unit.graph
                    entity_doc_iris[s] = unit.doc_iri
                    current = entity_classification.get(s, EntityClassification.FACT)
                    candidate = self._classify_entity_for_unit(s, unit, known_entities)
                    entity_classification[s] = (
                        candidate
                        if self._classification_priority(candidate)
                        >= self._classification_priority(current)
                        else current
                    )
                if isinstance(o, URIRef):
                    entities.add(o)
                    source_entities.add(o)
                    entity_graphs[o] = unit.graph
                    entity_doc_iris[o] = unit.doc_iri
                    current = entity_classification.get(o, EntityClassification.FACT)
                    candidate = self._classify_entity_for_unit(o, unit, known_entities)
                    entity_classification[o] = (
                        candidate
                        if self._classification_priority(candidate)
                        >= self._classification_priority(current)
                        else current
                    )

        return (
            list(entities),
            source_entities,
            entity_graphs,
            entity_doc_iris,
            entity_classification,
        )

    def aggregate_graphs(
        self,
        units: list[ContentUnit],
        ontology_graph: RDFGraph | None = None,
    ) -> RDFGraph:
        """Aggregate multiple content unit graphs with embedding-based disambiguation.

        Args:
            units: List of ContentUnits to aggregate.
            ontology_graph: Optional selected ontology graph used to distinguish
                known ontology entities from tentative ontology-like aliases.

        Returns:
            Merged RDF graph with provenance annotations.
        """
        logger.info(f"Starting aggregation with metadata for {len(units)} units")

        if not units:
            return RDFGraph()

        # Steps 1-3: Collect, normalise, embed, cluster
        known_ontology_entities = self._build_known_ontology_entities(ontology_graph)
        (
            entities,
            source_entities,
            entity_graphs,
            entity_doc_iris,
            entity_classification,
        ) = self._collect_all_entities(units, known_ontology_entities)
        representations = self.normalizer.create_representations_batch(
            entities, entity_graphs
        )
        tentative_entities = [
            entity
            for entity, classification in entity_classification.items()
            if classification == EntityClassification.TENTATIVE_ONTOLOGY
        ]
        anchor_candidates = self._select_ontology_anchor_candidates(
            tentative_entities=tentative_entities,
            tentative_representations=representations,
            ontology_graph=ontology_graph,
            known_ontology_entities=known_ontology_entities,
        )
        if anchor_candidates and units and ontology_graph is not None:
            first_doc_iri = units[0].doc_iri
            for ontology_entity in anchor_candidates:
                if ontology_entity in entity_graphs:
                    continue
                entities.append(ontology_entity)
                entity_graphs[ontology_entity] = ontology_graph
                entity_doc_iris[ontology_entity] = first_doc_iri
                entity_classification[ontology_entity] = (
                    EntityClassification.KNOWN_ONTOLOGY
                )
                representations[ontology_entity] = (
                    self.normalizer.create_representation(
                        ontology_entity, ontology_graph
                    )
                )
        entity_is_known_ontology = {
            entity: classification == EntityClassification.KNOWN_ONTOLOGY
            for entity, classification in entity_classification.items()
        }

        # Representative selection should prefer known ontology entities only.
        for entity, is_known_ontology in entity_is_known_ontology.items():
            representation = representations.get(entity)
            if representation is not None:
                representation.is_ontology_entity = is_known_ontology
        clusters = self._cluster_entities_by_role(representations)

        # Steps 4-6: Select, build URIs, compose
        clustering_mapping = self.selector.create_mapping(clusters, representations)
        representatives = list(set(clustering_mapping.values()))
        uri_mapping = self.uri_builder.create_uri_mapping(
            representatives,
            representations,
            entity_doc_iris=entity_doc_iris,
            entity_is_ontology=entity_is_known_ontology,
        )
        final_mapping = URIBuilder.compose_mappings(clustering_mapping, uri_mapping)
        final_mapping = {
            entity: mapped
            for entity, mapped in final_mapping.items()
            if entity in source_entities
        }

        # Keep known ontology entities stable. Tentative ontology-like entities are:
        # - mapped to known ontology representatives when present in a mixed cluster
        # - preserved as-is when only tentative entities are present
        ontology_sameas_links: dict[URIRef, set[URIRef]] = {}
        suppress_sameas_origins: set[URIRef] = set()
        for cluster in clusters:
            known_ontology_entities_in_cluster = [
                entity
                for entity in cluster
                if entity_classification.get(entity)
                == EntityClassification.KNOWN_ONTOLOGY
            ]
            tentative_entities_in_cluster = [
                entity
                for entity in cluster
                if entity_classification.get(entity)
                == EntityClassification.TENTATIVE_ONTOLOGY
            ]

            for entity in known_ontology_entities_in_cluster:
                final_mapping[entity] = entity

            if known_ontology_entities_in_cluster:
                canonical_known_ontology = self.selector.select_representative(
                    known_ontology_entities_in_cluster,
                    representations,
                )
                for tentative_entity in tentative_entities_in_cluster:
                    if self._can_merge_as_identity(
                        tentative_entity,
                        canonical_known_ontology,
                        representations,
                    ):
                        final_mapping[tentative_entity] = canonical_known_ontology
                        suppress_sameas_origins.add(tentative_entity)
                    else:
                        final_mapping[tentative_entity] = tentative_entity

            elif tentative_entities_in_cluster:
                for tentative_entity in tentative_entities_in_cluster:
                    final_mapping[tentative_entity] = tentative_entity

            if len(known_ontology_entities_in_cluster) > 1:
                canonical = self.selector.select_representative(
                    known_ontology_entities_in_cluster,
                    representations,
                )
                aliases = {
                    entity
                    for entity in known_ontology_entities_in_cluster
                    if entity != canonical
                    and entity in source_entities
                    and canonical in source_entities
                    and self._can_merge_as_identity(entity, canonical, representations)
                }
                if aliases:
                    ontology_sameas_links.setdefault(canonical, set()).update(aliases)

        self._force_fact_targets_to_doc_namespace(
            final_mapping=final_mapping,
            source_entities=source_entities,
            entity_doc_iris=entity_doc_iris,
            entity_classification=entity_classification,
        )

        # Step 7: Rewrite and merge with provenance
        active_units = [u for u in units if u.graph is not None]
        merged_graph = self.rewriter.merge_graphs_with_provenance(
            active_units,
            final_mapping,
            extra_sameas_links=ontology_sameas_links,
            suppress_sameas_origins=suppress_sameas_origins,
        )
        self._remove_base_namespace_sameas_leaks(merged_graph)

        # metadata = {
        #     "entity_mapping": final_mapping,
        #     "entity_doc_iris": entity_doc_iris,
        #     "clusters": clusters,
        #     "representations": representations,
        #     "embeddings": embeddings,
        #     "num_entities": len(entities),
        #     "num_clusters": len(clusters),
        #     "num_unique_targets": len(set(final_mapping.values())),
        # }

        logger.info("Aggregation with metadata complete")
        return merged_graph


# Convenience function for backward compatibility
def aggregate_content_unit_graphs(
    units: list[ContentUnit],
    similarity_threshold: float = 0.80,
) -> RDFGraph:
    """Convenience function to aggregate content unit graphs.

    Args:
        units: List of content units to aggregate.
        similarity_threshold: Cosine similarity threshold for clustering.

    Returns:
        Aggregated RDF graph.
    """
    aggregator = EmbeddingBasedAggregator(
        similarity_threshold=similarity_threshold,
    )
    return aggregator.aggregate_graphs(units)


def aggregate_chunk_graphs(
    units: list[ContentUnit],
    similarity_threshold: float = 0.80,
) -> RDFGraph:
    """Backward-compatible alias for :func:`aggregate_content_unit_graphs`."""
    return aggregate_content_unit_graphs(
        units=units,
        similarity_threshold=similarity_threshold,
    )
