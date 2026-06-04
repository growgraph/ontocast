"""Embedding-based RDF graph aggregator.

This module provides the main aggregator class that orchestrates entity
disambiguation using embedding-based clustering.

Pipeline:
1. Collect entities from all content units
2. Normalize entities: e -> r(e) (string representation with semantic context)
3. Generate embedding-based identity candidates
4. Validate candidate merges with symbolic identity checks
5. Select canonical identity per validated cluster
6. Assign final URIs from canonical identity + document namespace policy
7. Rewrite graphs: apply mapping e -> e' to all triples
"""

import logging
import re
from difflib import SequenceMatcher
from enum import StrEnum
from itertools import combinations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rdflib import URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from ontocast.onto.constants import DEFAULT_IRI, PROV, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.iri_policy import is_in_namespace
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.representation_text import normalize_uri_local_name

from .clustering import ClusterRepresentativeSelector, EntityClusterer
from .normalizer import EntityNormalizer, EntityRepresentation
from .rewriter import GraphRewriter
from .uri_builder import EntityRole, URIBuilder

logger = logging.getLogger(__name__)
_INSTANCE_LOCAL_NAME_RE = re.compile(r"^(?P<stem>.+?)(?P<index>\d+)$")


class EntityClassification(StrEnum):
    """Classification of entities during aggregation."""

    FACT = "fact"
    KNOWN_ONTOLOGY = "known_ontology"
    TENTATIVE_ONTOLOGY = "tentative_ontology"


class EntityDecision(BaseModel):
    """Decision record for one entity across aggregation stages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    classification: EntityClassification
    identity_target: URIRef
    final_uri: URIRef | None = None
    suppress_fact_subject_assertions: bool = False
    suppress_sameas: bool = False


class _EntityCollectionState(BaseModel):
    """Mutable state for entity collection across content units."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    known_entities: set[URIRef]
    entities: set[URIRef] = Field(default_factory=set)
    source_entities: set[URIRef] = Field(default_factory=set)
    entity_graphs: dict[URIRef, RDFGraph] = Field(default_factory=dict)
    entity_doc_iris: dict[URIRef, URIRef] = Field(default_factory=dict)
    entity_classification: dict[URIRef, EntityClassification] = Field(
        default_factory=dict
    )
    direct_relation_pairs: set[frozenset[URIRef]] = Field(default_factory=set)


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
        candidate_similarity_threshold: float = 0.70,
        add_sameas_links: bool = True,
        base_iri: str = DEFAULT_IRI,
    ):
        """Initialise the embedding-based aggregator.

        Args:
            embedding_model: Name of sentence transformer model.
            similarity_threshold: Cosine similarity threshold for clustering (0-1).
            candidate_similarity_threshold: Lower cosine threshold used to
                generate permissive merge candidates before symbolic validation.
            add_sameas_links: Whether to add owl:sameAs for merged entities.
            base_iri: Base IRI for fact entity URIs (default: DEFAULT_IRI).
                Entities under this namespace are facts; everything else is
                treated as an ontology entity and left unchanged.
        """
        self.base_iri = base_iri
        self.candidate_similarity_threshold = candidate_similarity_threshold

        # Pipeline components
        self.normalizer = EntityNormalizer(facts_iri=self.base_iri)
        self.clusterer = EntityClusterer(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
        )
        self.selector = ClusterRepresentativeSelector()
        self.uri_builder = URIBuilder(base_iri=self.base_iri)
        self.rewriter = GraphRewriter(
            add_sameas_links=add_sameas_links,
            blocked_sameas_namespaces=(self.base_iri,),
        )

    @staticmethod
    def _entity_in_namespace(entity: URIRef, namespace: URIRef | str | None) -> bool:
        """Return True when *entity* is under the provided namespace."""
        if namespace is None:
            return False
        return is_in_namespace(str(entity), str(namespace), context="auto")

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
        return len(left & right) / len(union)

    @staticmethod
    def _instance_like_local_name(entity: URIRef) -> str | None:
        """Return normalized local name when URI ends with numeric suffix."""
        local_name = normalize_uri_local_name(entity).replace(" ", "")
        if not local_name:
            return None
        match = _INSTANCE_LOCAL_NAME_RE.match(local_name)
        if match is None:
            return None
        if len(match.group("stem")) < 3:
            return None
        return local_name

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

        left_instance_name = self._instance_like_local_name(left)
        right_instance_name = self._instance_like_local_name(right)
        if (
            left_instance_name is not None
            and right_instance_name is not None
            and left_instance_name == right_instance_name
        ):
            return True

        left_label_tokens = {
            self.normalizer.normalize_string(label)
            for label in left_rep.labels + left_rep.alt_labels
            if label.strip()
        }
        right_label_tokens = {
            self.normalizer.normalize_string(label)
            for label in right_rep.labels + right_rep.alt_labels
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

        left_normalized = left_rep.normal_form.strip()
        right_normalized = right_rep.normal_form.strip()
        if left_normalized and right_normalized:
            if left_normalized != right_normalized and (
                left_normalized.startswith(f"{right_normalized} ")
                or right_normalized.startswith(f"{left_normalized} ")
            ):
                return False

        ratio = SequenceMatcher(
            None, left_rep.normal_form, right_rep.normal_form
        ).ratio()
        if ratio >= 0.90:
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
        direct_relation_pairs: set[frozenset[URIRef]] | None = None,
    ) -> bool:
        if (
            direct_relation_pairs is not None
            and frozenset((left, right)) in direct_relation_pairs
        ):
            return False
        return (
            self._are_roles_compatible(left, right, representations)
            and self._are_types_compatible(left, right, representations)
            and self._are_lexical_aliases(left, right, representations)
        )

    def _cluster_entities_by_role(
        self, representations: dict[URIRef, EntityRepresentation]
    ) -> tuple[list[list[URIRef]], dict[URIRef, np.ndarray]]:
        grouped_entities: dict[str, dict[URIRef, EntityRepresentation]] = {}
        for entity, representation in representations.items():
            grouped_entities.setdefault(self._role_key(representation), {})[entity] = (
                representation
            )

        all_clusters: list[list[URIRef]] = []
        all_embeddings: dict[URIRef, np.ndarray] = {}
        original_threshold = self.clusterer.similarity_threshold
        self.clusterer.similarity_threshold = self.candidate_similarity_threshold
        try:
            for role_representations in grouped_entities.values():
                role_clusters, role_embeddings = self.clusterer.cluster_entities(
                    role_representations
                )
                all_clusters.extend(role_clusters)
                all_embeddings.update(role_embeddings)
        finally:
            self.clusterer.similarity_threshold = original_threshold
        return all_clusters, all_embeddings

    @staticmethod
    def _candidate_similarity(
        left: URIRef,
        right: URIRef,
        embeddings: dict[URIRef, np.ndarray],
    ) -> float | None:
        left_embedding = embeddings.get(left)
        right_embedding = embeddings.get(right)
        if left_embedding is None or right_embedding is None:
            return None

        denominator = float(
            np.linalg.norm(left_embedding) * np.linalg.norm(right_embedding)
        )
        if denominator == 0:
            return None
        return float(np.dot(left_embedding, right_embedding) / denominator)

    def _merge_validation_failures(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> list[str]:
        failures: list[str] = []
        if not self._are_roles_compatible(left, right, representations):
            failures.append("role")
        if not self._are_types_compatible(left, right, representations):
            failures.append("type")
        if not self._are_lexical_aliases(left, right, representations):
            failures.append("lexical")
        return failures

    def _build_identity_clusters(
        self,
        candidate_clusters: list[list[URIRef]],
        representations: dict[URIRef, EntityRepresentation],
        embeddings: dict[URIRef, np.ndarray],
        direct_relation_pairs: set[frozenset[URIRef]] | None = None,
    ) -> tuple[
        list[list[URIRef]], list[tuple[URIRef, URIRef, float | None, tuple[str, ...]]]
    ]:
        validated_clusters: list[list[URIRef]] = []
        rejected_merges: list[tuple[URIRef, URIRef, float | None, tuple[str, ...]]] = []

        for candidate_cluster in candidate_clusters:
            if len(candidate_cluster) <= 1:
                validated_clusters.append(candidate_cluster)
                continue

            parents: dict[URIRef, URIRef] = {
                entity: entity for entity in candidate_cluster
            }

            def find(entity: URIRef) -> URIRef:
                root = parents[entity]
                if root != entity:
                    parents[entity] = find(root)
                return parents[entity]

            def union(left: URIRef, right: URIRef) -> None:
                left_root = find(left)
                right_root = find(right)
                if left_root == right_root:
                    return
                if str(left_root) <= str(right_root):
                    parents[right_root] = left_root
                else:
                    parents[left_root] = right_root

            for left, right in combinations(candidate_cluster, 2):
                score = self._candidate_similarity(left, right, embeddings)
                if score is not None and score < self.candidate_similarity_threshold:
                    continue
                if self._can_merge_as_identity(
                    left,
                    right,
                    representations,
                    direct_relation_pairs=direct_relation_pairs,
                ):
                    union(left, right)
                    continue
                rejected_merges.append(
                    (
                        left,
                        right,
                        score,
                        tuple(
                            self._merge_validation_failures(
                                left, right, representations
                            )
                        ),
                    )
                )

            grouped: dict[URIRef, list[URIRef]] = {}
            for entity in candidate_cluster:
                grouped.setdefault(find(entity), []).append(entity)

            for group in grouped.values():
                sorted_group = sorted(group, key=str)
                validated_clusters.append(sorted_group)

        return validated_clusters, rejected_merges

    def _select_ontology_anchor_candidates(
        self,
        tentative_entities: list[URIRef],
        tentative_representations: dict[URIRef, EntityRepresentation],
        tentative_doc_iris: dict[URIRef, URIRef],
        ontology_graph: RDFGraph | None,
        known_ontology_entities: set[URIRef],
    ) -> dict[URIRef, URIRef]:
        """Pick ontology anchors and preserve the triggering document IRI."""
        if (
            ontology_graph is None
            or not tentative_entities
            or not known_ontology_entities
        ):
            return {}

        ontology_entities = [
            entity
            for entity in known_ontology_entities
            if not self._is_standard_ontology_entity(entity)
        ]
        if not ontology_entities:
            return {}

        ontology_graphs = {entity: ontology_graph for entity in ontology_entities}
        ontology_representations = self.normalizer.create_representations_batch(
            ontology_entities, ontology_graphs
        )

        token_index: dict[str, set[URIRef]] = {}
        for entity, representation in ontology_representations.items():
            for token in self._tokenize(representation.representation):
                token_index.setdefault(token, set()).add(entity)

        selected: dict[URIRef, URIRef] = {}
        for tentative_entity in tentative_entities:
            tentative_representation = tentative_representations.get(tentative_entity)
            if tentative_representation is None:
                continue
            tentative_doc_iri = tentative_doc_iris.get(tentative_entity)
            if tentative_doc_iri is None:
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

            scored.sort(key=lambda item: (-item[0], str(item[1])))
            for _, candidate in scored[:3]:
                selected.setdefault(candidate, tentative_doc_iri)

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
    def _merge_into_context_graph(target: RDFGraph, source: RDFGraph) -> None:
        """Merge source triples/namespaces into a per-entity context graph."""
        target += source

    def _register_entity(
        self,
        *,
        entity: URIRef,
        unit: ContentUnit,
        state: _EntityCollectionState,
    ) -> None:
        """Register one URI entity with merged context and stable classification."""
        state.entities.add(entity)
        state.source_entities.add(entity)
        if entity not in state.entity_graphs:
            state.entity_graphs[entity] = unit.graph.copy()
        else:
            self._merge_into_context_graph(state.entity_graphs[entity], unit.graph)
        state.entity_doc_iris.setdefault(entity, unit.doc_iri)
        current = state.entity_classification.get(entity, EntityClassification.FACT)
        candidate = self._classify_entity_for_unit(entity, unit, state.known_entities)
        state.entity_classification[entity] = (
            candidate
            if self._classification_priority(candidate)
            >= self._classification_priority(current)
            else current
        )

    @staticmethod
    def _register_direct_relation(
        state: _EntityCollectionState,
        subject: URIRef,
        obj: URIRef,
    ) -> None:
        """Record direct subject-object URI relation pair in collection state."""
        if subject == obj:
            return
        state.direct_relation_pairs.add(frozenset((subject, obj)))

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
        set[frozenset[URIRef]],
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
        state = _EntityCollectionState(known_entities=known_ontology_entities or set())

        for unit in units:
            if unit.graph is None:
                continue
            unit.graph.sanitize_prefixes_namespaces()
            # Keep collection in the same URI space that rewrite/merge consumes
            # (unit.graph). Using graph_absolute here causes mapping keys to miss
            # during rewrite, because unit.graph still contains the original terms.
            for s, p, o in unit.graph:
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    self._register_direct_relation(state=state, subject=s, obj=o)
                for term in (s, p, o):
                    if isinstance(term, URIRef):
                        self._register_entity(entity=term, unit=unit, state=state)

        return (
            list(state.entities),
            state.source_entities,
            state.entity_graphs,
            state.entity_doc_iris,
            state.entity_classification,
            state.direct_relation_pairs,
        )

    def aggregate_graphs(
        self,
        units: list[ContentUnit],
        ontology_graph: RDFGraph,
    ) -> RDFGraph:
        """Aggregate multiple content unit graphs with embedding-based disambiguation.

        Args:
            units: List of ContentUnits to aggregate.
            ontology_graph: Selected ontology graph used to distinguish
                known ontology entities from tentative ontology-like aliases.

        Returns:
            Merged RDF graph with provenance annotations.
        """
        logger.info(f"Starting aggregation with metadata for {len(units)} units")
        if ontology_graph is None:
            raise ValueError("ontology_graph must not be None for facts aggregation")

        if not units:
            return RDFGraph()

        # Steps 1-3: Collect, normalise, candidate clustering
        known_ontology_entities = self._build_known_ontology_entities(ontology_graph)
        (
            entities,
            source_entities,
            entity_graphs,
            entity_doc_iris,
            entity_classification,
            direct_relation_pairs,
        ) = self._collect_all_entities(units, known_ontology_entities)
        representations = self.normalizer.create_representations_batch(
            entities, entity_graphs
        )
        decisions: dict[URIRef, EntityDecision] = {
            entity: EntityDecision(
                classification=classification,
                identity_target=entity,
            )
            for entity, classification in entity_classification.items()
        }
        tentative_entities = [
            entity
            for entity, decision in decisions.items()
            if decision.classification == EntityClassification.TENTATIVE_ONTOLOGY
        ]
        anchor_candidates = self._select_ontology_anchor_candidates(
            tentative_entities=tentative_entities,
            tentative_representations=representations,
            tentative_doc_iris=entity_doc_iris,
            ontology_graph=ontology_graph,
            known_ontology_entities=known_ontology_entities,
        )
        if anchor_candidates:
            for ontology_entity, anchor_doc_iri in anchor_candidates.items():
                if ontology_entity in entity_graphs:
                    continue
                entities.append(ontology_entity)
                entity_graphs[ontology_entity] = ontology_graph
                entity_doc_iris[ontology_entity] = anchor_doc_iri
                entity_classification[ontology_entity] = (
                    EntityClassification.KNOWN_ONTOLOGY
                )
                decisions[ontology_entity] = EntityDecision(
                    classification=EntityClassification.KNOWN_ONTOLOGY,
                    identity_target=ontology_entity,
                )
                representations[ontology_entity] = (
                    self.normalizer.create_representation(
                        ontology_entity, ontology_graph
                    )
                )
        entity_is_known_ontology = {
            entity: decision.classification == EntityClassification.KNOWN_ONTOLOGY
            for entity, decision in decisions.items()
        }
        if logger.isEnabledFor(logging.INFO):
            known_count = sum(
                1 for is_known in entity_is_known_ontology.values() if is_known
            )
            fact_count = sum(
                1
                for decision in decisions.values()
                if decision.classification == EntityClassification.FACT
            )
            logger.info(
                "Aggregation entity classification stats: fact=%d known_ontology=%d "
                "tentative_ontology=%d",
                fact_count,
                known_count,
                len(tentative_entities),
            )

        candidate_clusters, embeddings = self._cluster_entities_by_role(representations)
        clusters, rejected_merges = self._build_identity_clusters(
            candidate_clusters=candidate_clusters,
            representations=representations,
            embeddings=embeddings,
            direct_relation_pairs=direct_relation_pairs,
        )
        if rejected_merges:
            logger.info(
                "Rejected %d candidate merges after symbolic validation",
                len(rejected_merges),
            )
            for left, right, score, failed_checks in rejected_merges:
                logger.debug(
                    "Rejected candidate merge: %s <-> %s (score=%s, failed=%s)",
                    left,
                    right,
                    f"{score:.3f}" if score is not None else "n/a",
                    ",".join(failed_checks) if failed_checks else "unknown",
                )

        # Step 4: Canonical identity mapping (no URI policy yet)
        identity_mapping = self.selector.create_mapping(
            clusters,
            representations,
            entity_is_known_ontology=entity_is_known_ontology,
        )

        # Keep known ontology entities stable. Tentative ontology-like entities are:
        # - mapped to known ontology representatives when present in a mixed cluster
        # - preserved as-is when only tentative entities are present
        suppress_sameas_origins: set[URIRef] = set()
        suppress_fact_subject_sources: set[URIRef] = set()
        for cluster in clusters:
            known_ontology_entities_in_cluster = [
                entity
                for entity in cluster
                if decisions.get(entity) is not None
                and decisions[entity].classification
                == EntityClassification.KNOWN_ONTOLOGY
            ]
            tentative_entities_in_cluster = [
                entity
                for entity in cluster
                if decisions.get(entity) is not None
                and decisions[entity].classification
                == EntityClassification.TENTATIVE_ONTOLOGY
            ]
            fact_entities_in_cluster = [
                entity
                for entity in cluster
                if decisions.get(entity) is not None
                and decisions[entity].classification == EntityClassification.FACT
            ]

            for entity in known_ontology_entities_in_cluster:
                identity_mapping[entity] = entity

            if known_ontology_entities_in_cluster:
                canonical_known_ontology = self.selector.select_representative(
                    known_ontology_entities_in_cluster,
                    representations,
                    entity_is_known_ontology=entity_is_known_ontology,
                )
                for tentative_entity in tentative_entities_in_cluster:
                    if self._can_merge_as_identity(
                        tentative_entity,
                        canonical_known_ontology,
                        representations,
                        direct_relation_pairs=direct_relation_pairs,
                    ):
                        identity_mapping[tentative_entity] = canonical_known_ontology
                        decisions[tentative_entity].suppress_sameas = True
                    else:
                        identity_mapping[tentative_entity] = tentative_entity
                for fact_entity in fact_entities_in_cluster:
                    if self._can_merge_as_identity(
                        fact_entity,
                        canonical_known_ontology,
                        representations,
                        direct_relation_pairs=direct_relation_pairs,
                    ):
                        identity_mapping[fact_entity] = canonical_known_ontology
                        decisions[fact_entity].suppress_sameas = True
                        decisions[fact_entity].suppress_fact_subject_assertions = True
                    else:
                        identity_mapping[fact_entity] = fact_entity

            elif tentative_entities_in_cluster:
                # In mixed FACT + TENTATIVE clusters with no known ontology
                # entity, prefer the FACT side when symbolic identity checks
                # agree (e.g. hallucinated ontology prefix on an instance).
                if fact_entities_in_cluster:
                    canonical_fact = self.selector.select_representative(
                        fact_entities_in_cluster,
                        representations,
                        entity_is_known_ontology=entity_is_known_ontology,
                    )
                    for fact_entity in fact_entities_in_cluster:
                        identity_mapping[fact_entity] = canonical_fact
                    for tentative_entity in tentative_entities_in_cluster:
                        if self._can_merge_as_identity(
                            tentative_entity,
                            canonical_fact,
                            representations,
                            direct_relation_pairs=direct_relation_pairs,
                        ):
                            identity_mapping[tentative_entity] = canonical_fact
                            decisions[tentative_entity].suppress_sameas = True
                        else:
                            identity_mapping[tentative_entity] = tentative_entity
                else:
                    for tentative_entity in tentative_entities_in_cluster:
                        identity_mapping[tentative_entity] = tentative_entity

        for entity, target in identity_mapping.items():
            if entity in decisions:
                decisions[entity].identity_target = target

        suppress_sameas_origins = {
            entity for entity, decision in decisions.items() if decision.suppress_sameas
        }
        suppress_fact_subject_sources = {
            entity
            for entity, decision in decisions.items()
            if decision.suppress_fact_subject_assertions
        }

        # Step 5: URI assignment from canonical identity + namespace policy
        final_mapping = self.uri_builder.create_entity_uri_mapping(
            identity_mapping=identity_mapping,
            representations=representations,
            entity_doc_iris=entity_doc_iris,
            entity_is_ontology={
                entity: (
                    decisions.get(entity) is not None
                    and decisions[entity].classification != EntityClassification.FACT
                )
                for entity in representations
            },
        )
        for entity, final_uri in final_mapping.items():
            if entity in decisions:
                decisions[entity].final_uri = final_uri
        known_ontology_entities_all = {
            entity
            for entity, decision in decisions.items()
            if decision.classification == EntityClassification.KNOWN_ONTOLOGY
        }
        assert all(
            identity_mapping.get(entity, entity) == entity
            for entity in known_ontology_entities_all
        ), "Known ontology entities must remain identity-mapped"
        assert not (known_ontology_entities_all & suppress_sameas_origins), (
            "Known ontology entities cannot be suppress_sameas origins"
        )
        assert not (known_ontology_entities_all & suppress_fact_subject_sources), (
            "Known ontology entities cannot be suppress_fact_subject origins"
        )
        assert all(entity in decisions for entity in source_entities), (
            "Every source entity must have a decision record"
        )
        final_mapping = {
            entity: mapped
            for entity, mapped in final_mapping.items()
            if entity in source_entities
        }

        # Step 7: Rewrite and merge with provenance
        active_units = [u for u in units if u.graph is not None and len(u.graph) > 0]
        merged_graph = self.rewriter.merge_graphs_with_provenance(
            active_units,
            final_mapping,
            suppress_sameas_origins=suppress_sameas_origins,
            suppress_fact_subject_sources=suppress_fact_subject_sources,
        )

        logger.info("Aggregation with metadata complete")
        return merged_graph

    def postprocess_facts_units(
        self,
        units: list[ContentUnit],
        ontology_graph: RDFGraph,
    ) -> RDFGraph:
        """Sanitize facts units, then run aggregation/normalization.

        This method is intentionally safe for both single-unit and multi-unit
        inputs so unit-pipeline and graph-pipeline paths share the same
        post-processing behavior.
        """
        for unit in units:
            unit.sanitize()
        return self.aggregate_graphs(units=units, ontology_graph=ontology_graph)
