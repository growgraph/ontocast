from __future__ import annotations

import numpy as np
import pytest
from rdflib import RDF, URIRef

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
