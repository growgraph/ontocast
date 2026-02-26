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

from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph

from .clustering import ClusterRepresentativeSelector, EntityClusterer
from .normalizer import EntityNormalizer
from .rewriter import GraphRewriter
from .uri_builder import URIBuilder

logger = logging.getLogger(__name__)


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
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
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

    def _collect_all_entities(
        self,
        units: list[ContentUnit],
    ) -> tuple[
        list[URIRef],
        dict[URIRef, RDFGraph],
        dict[URIRef, URIRef],
        dict[URIRef, bool],
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
        entity_graphs: dict[URIRef, RDFGraph] = {}
        entity_doc_iris: dict[URIRef, URIRef] = {}
        entity_is_ontology: dict[URIRef, bool] = {}

        for unit in units:
            if unit.graph is None:
                continue
            # Keep collection in the same URI space that rewrite/merge consumes
            # (unit.graph). Using graph_absolute here causes mapping keys to miss
            # during rewrite, because unit.graph still contains the original terms.
            for s, p, o in unit.graph:
                if isinstance(s, URIRef):
                    entities.add(s)
                    entity_graphs[s] = unit.graph
                    entity_doc_iris[s] = unit.doc_iri
                    entity_is_ontology[s] = (
                        entity_is_ontology.get(s, False)
                        or unit.type == OutputType.ONTOLOGIES
                    )
                if isinstance(o, URIRef):
                    entities.add(o)
                    entity_graphs[o] = unit.graph
                    entity_doc_iris[o] = unit.doc_iri
                    entity_is_ontology[o] = (
                        entity_is_ontology.get(o, False)
                        or unit.type == OutputType.ONTOLOGIES
                    )

        return list(entities), entity_graphs, entity_doc_iris, entity_is_ontology

    def aggregate_graphs(self, units: list[ContentUnit]) -> RDFGraph:
        """Aggregate multiple content unit graphs with embedding-based disambiguation.

        Args:
            units: List of ContentUnits to aggregate.

        Returns:
            Merged RDF graph with provenance annotations.
        """
        logger.info(f"Starting aggregation with metadata for {len(units)} units")

        if not units:
            return RDFGraph()

        # Steps 1-3: Collect, normalise, embed, cluster
        entities, entity_graphs, entity_doc_iris, entity_is_ontology = (
            self._collect_all_entities(units)
        )
        representations = self.normalizer.create_representations_batch(
            entities, entity_graphs
        )
        clusters, embeddings = self.clusterer.cluster_entities(representations)

        # Steps 4-6: Select, build URIs, compose
        clustering_mapping = self.selector.create_mapping(clusters, representations)
        representatives = list(set(clustering_mapping.values()))
        uri_mapping = self.uri_builder.create_uri_mapping(
            representatives,
            representations,
            entity_doc_iris=entity_doc_iris,
            entity_is_ontology=entity_is_ontology,
        )
        final_mapping = URIBuilder.compose_mappings(clustering_mapping, uri_mapping)

        # Step 7: Rewrite and merge with provenance
        active_units = [u for u in units if u.graph is not None]
        merged_graph = self.rewriter.merge_graphs_with_provenance(
            active_units,
            final_mapping,
        )

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
def aggregate_chunk_graphs(
    units: list[ContentUnit],
    similarity_threshold: float = 0.85,
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
