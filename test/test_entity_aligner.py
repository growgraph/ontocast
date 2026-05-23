from __future__ import annotations

import numpy as np
import pytest
from rdflib import RDF, RDFS, Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.entity_aligner import EntityAligner
from ontocast.tool.agg.match_models import MatchRegime, TaggedGraph


def _graph(subject_ns: str, type_ns: str, predicate_ns: str) -> RDFGraph:
    graph = RDFGraph()
    entity = URIRef(f"{subject_ns}Alpha")
    target = URIRef(f"{subject_ns}Beta")
    graph.add((entity, RDF.type, URIRef(f"{type_ns}Person")))
    graph.add((entity, URIRef(f"{predicate_ns}relatedTo"), target))
    return graph


def test_align_identical_graphs_produces_cross_graph_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph("https://example/", "https://type.example/", "https://pred.example/")
    aligner = EntityAligner(similarity_threshold=0.1)

    def fake_encode(texts, **_kwargs):
        return [np.array([1.0, 0.0]) for _ in texts]

    monkeypatch.setattr(aligner.clusterer.embedder, "encode", fake_encode)
    result = aligner.align_graphs(
        [
            TaggedGraph(id="predicted", graph=graph),
            TaggedGraph(id="gt", graph=graph),
        ]
    )
    assert result.entity_count == 10
    assert result.cluster_count > 0
    assert any(len(cluster.members) >= 2 for cluster in result.clusters)


def test_strict_regime_fewer_cross_graph_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted_graph = _graph(
        "https://predicted.example/",
        "https://ontology-a.example/",
        "https://pred.example/",
    )
    gt_graph = _graph(
        "https://gt.example/",
        "https://ontology-b.example/",
        "https://pred.example/",
    )
    aligner = EntityAligner(similarity_threshold=0.1)

    def fake_encode(texts, **_kwargs):
        vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        return [vectors[index % len(vectors)] for index in range(len(texts))]

    monkeypatch.setattr(aligner.clusterer.embedder, "encode", fake_encode)

    loose = aligner.align_graphs(
        [
            TaggedGraph(id="predicted", graph=predicted_graph),
            TaggedGraph(id="gt", graph=gt_graph),
        ],
        regime=MatchRegime.ONTOLOGY_LOOSE,
    )
    strict = aligner.align_graphs(
        [
            TaggedGraph(id="predicted", graph=predicted_graph),
            TaggedGraph(id="gt", graph=gt_graph),
        ],
        regime=MatchRegime.ONTOLOGY_STRICT,
    )
    assert strict.cluster_count >= loose.cluster_count


def test_align_matches_identical_labels_despite_low_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted_graph = RDFGraph()
    gt_graph = RDFGraph()
    predicted_entity = URIRef("https://predicted.example/filmDirectedBy")
    gt_entity = URIRef("https://gt.example/seiji_mizushima")
    person_type = URIRef("https://ontology.example/Person")

    predicted_graph.add((predicted_entity, RDF.type, person_type))
    predicted_graph.add((predicted_entity, RDFS.label, Literal("Seiji Mizushima")))
    gt_graph.add((gt_entity, RDF.type, person_type))
    gt_graph.add((gt_entity, RDFS.label, Literal("Seiji Mizushima")))

    aligner = EntityAligner(similarity_threshold=0.99)

    def fake_encode(texts, **_kwargs):
        return [np.array([1.0, 0.0]), np.array([0.0, 1.0])] * (len(texts) // 2 + 1)

    monkeypatch.setattr(aligner.clusterer.embedder, "encode", fake_encode)
    result = aligner.align_graphs(
        [
            TaggedGraph(id="predicted", graph=predicted_graph),
            TaggedGraph(id="gt", graph=gt_graph),
        ]
    )
    cross_graph_pairs = [
        cluster
        for cluster in result.clusters
        if {member.graph_id for member in cluster.members} == {"predicted", "gt"}
        and len(cluster.members) >= 2
    ]
    assert cross_graph_pairs


def test_align_class_entity_matches_instance_of_that_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted_graph = RDFGraph()
    gt_graph = RDFGraph()
    city_class = URIRef("https://ontology.example/Q515")
    film = URIRef("https://predicted.example/fact1")
    city_instance = URIRef("https://gt.example/new_york_city")
    film_type = URIRef("https://ontology.example/Film")
    location_pred = URIRef("https://relations.example/P840")

    predicted_graph.add((film, RDF.type, film_type))
    predicted_graph.add((film, location_pred, city_class))
    gt_graph.add((city_instance, RDF.type, city_class))
    gt_graph.add((city_instance, RDFS.label, Literal("New York City")))

    aligner = EntityAligner(similarity_threshold=0.99)

    def fake_encode(texts, **_kwargs):
        return [np.array([1.0, 0.0, 0.0]) for _ in texts]

    monkeypatch.setattr(aligner.clusterer.embedder, "encode", fake_encode)
    result = aligner.align_graphs(
        [
            TaggedGraph(id="predicted", graph=predicted_graph),
            TaggedGraph(id="gt", graph=gt_graph),
        ]
    )
    members_by_entity = {
        member.entity: member
        for cluster in result.clusters
        for member in cluster.members
    }
    assert city_class in members_by_entity
    assert city_instance in members_by_entity
    assert (
        members_by_entity[city_class].graph_id
        != members_by_entity[city_instance].graph_id
    )
