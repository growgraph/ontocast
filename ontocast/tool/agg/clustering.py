"""Embedding-based entity clustering for disambiguation.

This module handles the embedding and clustering of entity representations
to identify groups of similar entities.
"""

import importlib
import logging
from typing import Any

import numpy as np
from rdflib import URIRef
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .normalizer import EntityRepresentation

logger = logging.getLogger(__name__)


class EntityClusterer:
    """Clusters entities based on embedding similarity.

    This class handles the embedding of entity representations and
    grouping them into clusters of similar entities.
    """

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.80,
        min_cluster_size: int = 1,
    ):
        """Initialize the entity clusterer.

        Args:
            embedding_model: Name of the sentence transformer model to use
            similarity_threshold: Minimum cosine similarity for grouping (0-1)
            min_cluster_size: Minimum size for a cluster (1 allows singletons)
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self._embedder: Any | None = None

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            try:
                st = importlib.import_module("sentence_transformers")
            except ImportError as e:
                raise ImportError(
                    "Entity clustering requires the sentence-transformers package. "
                    "Install it with: uv add sentence-transformers"
                ) from e

            self._embedder = st.SentenceTransformer(self.embedding_model)
        return self._embedder

    def embed_representations(
        self, representations: dict[URIRef, EntityRepresentation]
    ) -> dict[URIRef, np.ndarray]:
        """Embed all entity representations in parallel.

        This is much faster than embedding one at a time.

        Args:
            representations: Dictionary mapping entities to their representations

        Returns:
            Dictionary mapping entities to their embedding vectors
        """
        if not representations:
            return {}

        # Prepare batch of texts
        entities = list(representations.keys())
        texts = [representations[e].representation for e in entities]

        logger.info(f"Embedding {len(texts)} entity representations in parallel...")

        # Batch embedding (much faster!)
        embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=len(texts) > 100
        )

        # Create mapping
        entity_embeddings = {
            entity: embedding for entity, embedding in zip(entities, embeddings)
        }

        logger.info(f"Embedded {len(entity_embeddings)} entities")
        return entity_embeddings

    def cluster_by_similarity(
        self,
        embeddings: dict[URIRef, np.ndarray],
        representations: dict[URIRef, EntityRepresentation],
    ) -> list[list[URIRef]]:
        """Cluster entities based on embedding similarity.

        Args:
            embeddings: Dictionary mapping entities to embeddings
            representations: Dictionary mapping entities to their representations

        Returns:
            List of clusters (each cluster is a list of entity URIs)
        """
        if not embeddings:
            return []

        entities = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[e] for e in entities])

        logger.info(f"Clustering {len(entities)} entities...")

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(embedding_matrix)

        # Convert similarity to distance for DBSCAN (must be non-negative)
        # DBSCAN uses epsilon as maximum distance, so we use 1 - similarity
        distance_matrix = np.maximum(0.0, 1.0 - similarity_matrix)

        # Use DBSCAN for clustering
        # eps is the maximum distance between two samples for them to be in same cluster
        # We want high similarity (low distance), so eps = 1 - threshold
        eps = 1 - self.similarity_threshold

        clusterer = DBSCAN(
            eps=eps, min_samples=self.min_cluster_size, metric="precomputed"
        )

        cluster_labels = clusterer.fit_predict(distance_matrix)

        # Group entities by cluster
        clusters_dict: dict[int, list[URIRef]] = {}
        for entity, label in zip(entities, cluster_labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(entity)

        # Convert to list of clusters
        clusters = list(clusters_dict.values())

        # Log statistics
        singleton_count = sum(1 for c in clusters if len(c) == 1)
        multi_count = sum(1 for c in clusters if len(c) > 1)
        max_size = max(len(c) for c in clusters) if clusters else 0

        logger.info(
            f"Formed {len(clusters)} clusters: "
            f"{singleton_count} singletons, "
            f"{multi_count} multi-entity clusters, "
            f"max cluster size: {max_size}"
        )

        return clusters

    def cluster_entities(
        self, representations: dict[URIRef, EntityRepresentation]
    ) -> tuple[list[list[URIRef]], dict[URIRef, np.ndarray]]:
        """Complete clustering pipeline: embed and cluster.

        Args:
            representations: Dictionary mapping entities to their representations

        Returns:
            Tuple of (clusters, embeddings)
            - clusters: List of entity groups
            - embeddings: Dictionary mapping entities to their embeddings
        """
        # Step 1: Embed all representations in parallel
        embeddings = self.embed_representations(representations)

        # Step 2: Cluster based on similarity
        clusters = self.cluster_by_similarity(embeddings, representations)

        return clusters, embeddings


class ClusterRepresentativeSelector:
    """Selects the best representative entity from a cluster.

    The selection criteria are:
    1. Prefer ontology entities over fact entities
    2. Among ontology entities (or fact entities), prefer simpler URIs
    """

    def __init__(self):
        """Initialize the representative selector."""
        pass

    def compute_simplicity_score(self, entity: URIRef) -> float:
        """Compute simplicity score for an entity URI.

        Lower score = simpler = better

        Args:
            entity: Entity URI

        Returns:
            Simplicity score (lower is better)
        """
        uri_str = str(entity)

        # Factors that increase complexity (decrease simplicity)
        score = 0.0

        # Length penalty (longer URIs are more complex)
        score += len(uri_str) * 0.1

        # Path depth penalty (more / means deeper hierarchy)
        score += uri_str.count("/") * 5

        # Underscore/hyphen penalty (more complex names)
        score += uri_str.count("_") * 2
        score += uri_str.count("-") * 2

        # Number penalty (URIs with numbers are often auto-generated)
        score += sum(c.isdigit() for c in uri_str) * 1

        return score

    def select_representative(
        self,
        cluster: list[URIRef],
        representations: dict[URIRef, EntityRepresentation],
        entity_is_known_ontology: dict[URIRef, bool] | None = None,
    ) -> URIRef:
        """Select the best representative entity from a cluster.

        Selection criteria:
        1. Prefer ontology entities
        2. Among same category, prefer simpler URIs

        Args:
            cluster: List of entity URIs in the cluster
            representations: Dictionary mapping entities to their representations

        Returns:
            The selected representative entity URI
        """
        if len(cluster) == 1:
            return cluster[0]

        known_ontology_map = entity_is_known_ontology or {}

        # Separate ontology entities from fact entities
        ontology_entities = [
            e
            for e in cluster
            if known_ontology_map.get(e, representations[e].is_ontology_entity)
        ]
        fact_entities = [
            e
            for e in cluster
            if not known_ontology_map.get(e, representations[e].is_ontology_entity)
        ]

        # Prefer ontology entities
        candidates = ontology_entities if ontology_entities else fact_entities

        # Among candidates, select the simplest
        best = min(candidates, key=self.compute_simplicity_score)

        logger.debug(
            f"Selected representative {best} from cluster of {len(cluster)} entities "
            f"({len(ontology_entities)} ontology, {len(fact_entities)} facts)"
        )

        return best

    def create_mapping(
        self,
        clusters: list[list[URIRef]],
        representations: dict[URIRef, EntityRepresentation],
        entity_is_known_ontology: dict[URIRef, bool] | None = None,
    ) -> dict[URIRef, URIRef]:
        """Create mapping from all entities to their cluster representatives.

        Args:
            clusters: List of entity clusters
            representations: Dictionary mapping entities to their representations

        Returns:
            Dictionary mapping each entity to its representative (e -> e')
        """
        mapping = {}

        for cluster in clusters:
            representative = self.select_representative(
                cluster,
                representations,
                entity_is_known_ontology=entity_is_known_ontology,
            )

            for entity in cluster:
                mapping[entity] = representative

        logger.info(
            f"Created mapping for {len(mapping)} entities "
            f"to {len(set(mapping.values()))} representatives"
        )

        return mapping
