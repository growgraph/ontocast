"""Global entity alignment across multiple RDF graphs."""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
from rdflib import URIRef

from ontocast.onto.iri_policy import split_namespace_local

from .aggregate import EmbeddingBasedAggregator
from .clustering import EntityClusterer
from .match_common import GraphEntityRef, cosine_similarity
from .match_models import (
    EntityAlignmentResult,
    EntityCluster,
    GraphEntityMember,
    MatchRegime,
    TaggedGraph,
)
from .normalizer import EntityNormalizer, EntityRepresentation

logger = logging.getLogger(__name__)


class EntityAligner:
    """Align entities globally across a list of tagged RDF graphs."""

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.80,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.normalizer: EntityNormalizer = EntityNormalizer()
        self.clusterer: EntityClusterer = EntityClusterer(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
        )
        self._compat = EmbeddingBasedAggregator(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            candidate_similarity_threshold=similarity_threshold,
        )

    @staticmethod
    def _namespace_set(types: list[URIRef]) -> set[str]:
        namespaces: set[str] = set()
        for entity_type in types:
            namespace, _ = split_namespace_local(str(entity_type))
            if namespace is not None:
                namespaces.add(namespace)
        return namespaces

    def _strict_types_compatible(
        self,
        left: GraphEntityRef,
        right: GraphEntityRef,
        representations: dict[GraphEntityRef, EntityRepresentation],
    ) -> bool:
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        if not left_rep.types or not right_rep.types:
            return True
        left_namespaces = self._namespace_set(left_rep.types)
        right_namespaces = self._namespace_set(right_rep.types)
        if not left_namespaces or not right_namespaces:
            return False
        return bool(left_namespaces & right_namespaces)

    def _normalized_label_tokens(self, rep: EntityRepresentation) -> set[str]:
        return {
            self.normalizer.normalize_string(label)
            for label in rep.labels + rep.alt_labels
            if label.strip()
        }

    def _exact_label_match(
        self,
        left: GraphEntityRef,
        right: GraphEntityRef,
        representations: dict[GraphEntityRef, EntityRepresentation],
    ) -> bool:
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        left_tokens = self._normalized_label_tokens(left_rep)
        right_tokens = self._normalized_label_tokens(right_rep)
        if not left_tokens or not right_tokens:
            return False
        return bool(left_tokens & right_tokens)

    def _class_instance_compatible(
        self,
        left: GraphEntityRef,
        right: GraphEntityRef,
        representations: dict[GraphEntityRef, EntityRepresentation],
    ) -> bool:
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        return left.entity in right_rep.types or right.entity in left_rep.types

    def _pair_compatible(
        self,
        left: GraphEntityRef,
        right: GraphEntityRef,
        representations: dict[GraphEntityRef, EntityRepresentation],
        regime: MatchRegime,
    ) -> bool:
        if self._class_instance_compatible(left, right, representations):
            if regime == MatchRegime.ONTOLOGY_STRICT:
                return self._strict_types_compatible(left, right, representations)
            return True

        pair_representations = {
            left.entity: representations[left],
            right.entity: representations[right],
        }
        if not self._compat._are_roles_compatible(
            left.entity, right.entity, pair_representations
        ):
            return False
        if not self._compat._are_lexical_aliases(
            left.entity, right.entity, pair_representations
        ):
            return False
        if regime == MatchRegime.ONTOLOGY_STRICT:
            if not self._strict_types_compatible(left, right, representations):
                return False
        return True

    def _connected_components(
        self,
        nodes: list[GraphEntityRef],
        edges: list[tuple[GraphEntityRef, GraphEntityRef]],
    ) -> list[list[GraphEntityRef]]:
        adjacency: dict[GraphEntityRef, set[GraphEntityRef]] = {
            node: set() for node in nodes
        }
        for left, right in edges:
            adjacency[left].add(right)
            adjacency[right].add(left)

        visited: set[GraphEntityRef] = set()
        components: list[list[GraphEntityRef]] = []
        for start in nodes:
            if start in visited:
                continue
            stack = [start]
            component: list[GraphEntityRef] = []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                stack.extend(
                    neighbor for neighbor in adjacency[node] if neighbor not in visited
                )
            component.sort(key=lambda ref: (ref.graph_id, str(ref.entity)))
            components.append(component)
        return components

    def align_graphs(
        self,
        graphs: list[TaggedGraph],
        *,
        regime: MatchRegime = MatchRegime.ONTOLOGY_LOOSE,
    ) -> EntityAlignmentResult:
        from .match_common import extract_entities

        refs: list[GraphEntityRef] = []
        representations: dict[GraphEntityRef, EntityRepresentation] = {}
        for tagged in graphs:
            for entity in extract_entities(tagged.graph):
                ref = GraphEntityRef(graph_id=tagged.id, entity=entity)
                refs.append(ref)
                representations[ref] = self.normalizer.create_representation(
                    entity, tagged.graph
                )

        if not refs:
            return EntityAlignmentResult(
                regime=regime,
                similarity_threshold=self.similarity_threshold,
                entity_count=0,
                cluster_count=0,
                clusters=[],
            )

        ordered_refs = list(representations.keys())
        texts = [representations[ref].representation for ref in ordered_refs]
        vectors = self.clusterer.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=len(texts) > 100
        )
        embeddings: dict[GraphEntityRef, np.ndarray] = {
            ref: vector for ref, vector in zip(ordered_refs, vectors)
        }

        edges: list[tuple[GraphEntityRef, GraphEntityRef]] = []
        edge_scores: dict[tuple[GraphEntityRef, GraphEntityRef], float] = {}
        for left, right in combinations(refs, 2):
            if left.graph_id == right.graph_id:
                continue
            if not self._pair_compatible(left, right, representations, regime):
                continue
            left_embedding = embeddings[left]
            right_embedding = embeddings[right]
            score = cosine_similarity(left_embedding, right_embedding)
            label_confirmed = self._exact_label_match(left, right, representations)
            if score < self.similarity_threshold and not label_confirmed:
                continue
            edge_score = score if score >= self.similarity_threshold else 1.0
            edges.append((left, right))
            edge_scores[(left, right)] = edge_score
            edge_scores[(right, left)] = edge_score

        adjacency: dict[GraphEntityRef, set[GraphEntityRef]] = {
            node: set() for node in refs
        }
        for left, right in edges:
            adjacency[left].add(right)
            adjacency[right].add(left)

        components = self._connected_components(refs, edges)
        clusters: list[EntityCluster] = []
        for component in components:
            component_set = set(component)
            members: list[GraphEntityMember] = []
            for ref in component:
                best_score: float | None = None
                for neighbor in adjacency[ref]:
                    if neighbor not in component_set:
                        continue
                    score = edge_scores.get((ref, neighbor))
                    if score is None:
                        continue
                    if best_score is None or score > best_score:
                        best_score = score
                members.append(
                    GraphEntityMember(
                        graph_id=ref.graph_id,
                        entity=ref.entity,
                        similarity=best_score,
                    )
                )
            clusters.append(EntityCluster(members=members))

        logger.info(
            "Aligned %s entities into %s clusters across %s graphs",
            len(refs),
            len(clusters),
            len(graphs),
        )
        return EntityAlignmentResult(
            regime=regime,
            similarity_threshold=self.similarity_threshold,
            entity_count=len(refs),
            cluster_count=len(clusters),
            clusters=clusters,
        )
