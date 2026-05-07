from __future__ import annotations

import numpy as np
import pytest
from rdflib import RDF, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.matcher import GroundTruthSide, MatchRegime, TripleSetMatcher


def _graph(subject_ns: str, type_ns: str, predicate_ns: str) -> RDFGraph:
    graph = RDFGraph()
    entity = URIRef(f"{subject_ns}Alpha")
    target = URIRef(f"{subject_ns}Beta")
    graph.add((entity, RDF.type, URIRef(f"{type_ns}Person")))
    graph.add((entity, URIRef(f"{predicate_ns}relatedTo"), target))
    return graph


def test_match_exact_graphs_have_perfect_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(
        "https://left.example/", "https://type.example/", "https://pred.example/"
    )
    matcher = TripleSetMatcher(similarity_threshold=0.1)

    def fake_embeddings(*_args, **_kwargs):
        return {
            URIRef("https://left.example/Alpha"): np.array([1.0, 0.0]),
            URIRef("https://left.example/Beta"): np.array([0.0, 1.0]),
            URIRef("https://pred.example/relatedTo"): np.array([0.5, 0.5]),
            URIRef(str(RDF.type)): np.array([0.2, 0.8]),
        }

    monkeypatch.setattr(matcher.clusterer, "embed_representations", fake_embeddings)
    result = matcher.match(graph, graph)
    assert result.metrics.precision == 1.0
    assert result.metrics.recall == 1.0
    assert result.metrics.f1 == 1.0


def test_strict_requires_type_namespace_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_graph = _graph(
        "https://left.example/", "https://ontology-a.example/", "https://pred.example/"
    )
    right_graph = _graph(
        "https://right.example/", "https://ontology-b.example/", "https://pred.example/"
    )
    matcher = TripleSetMatcher(similarity_threshold=0.1)

    embeddings = {
        URIRef("https://left.example/Alpha"): np.array([1.0, 0.0]),
        URIRef("https://left.example/Beta"): np.array([0.0, 1.0]),
        URIRef("https://right.example/Alpha"): np.array([1.0, 0.0]),
        URIRef("https://right.example/Beta"): np.array([0.0, 1.0]),
        URIRef("https://pred.example/relatedTo"): np.array([0.3, 0.7]),
        URIRef(str(RDF.type)): np.array([0.5, 0.2]),
        URIRef("https://ontology-a.example/Person"): np.array([0.1, 0.9]),
        URIRef("https://ontology-b.example/Person"): np.array([0.1, 0.9]),
    }
    monkeypatch.setattr(
        matcher.clusterer,
        "embed_representations",
        lambda *_args, **_kwargs: embeddings,
    )

    loose = matcher.match(left_graph, right_graph, regime=MatchRegime.ONTOLOGY_LOOSE)
    strict = matcher.match(left_graph, right_graph, regime=MatchRegime.ONTOLOGY_STRICT)

    assert len(loose.entity_matches) > len(strict.entity_matches)
    assert strict.metrics.true_positives < loose.metrics.true_positives


def test_match_is_deterministic_for_equal_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_graph = RDFGraph()
    right_graph = RDFGraph()
    left_graph.add((URIRef("https://l/A1"), RDF.type, URIRef("https://types/Thing")))
    left_graph.add((URIRef("https://l/A2"), RDF.type, URIRef("https://types/Thing")))
    right_graph.add((URIRef("https://r/A1"), RDF.type, URIRef("https://types/Thing")))
    right_graph.add((URIRef("https://r/A2"), RDF.type, URIRef("https://types/Thing")))
    matcher = TripleSetMatcher(similarity_threshold=0.1)
    monkeypatch.setattr(
        matcher.clusterer,
        "embed_representations",
        lambda *_args, **_kwargs: {
            URIRef("https://l/A1"): np.array([1.0, 0.0]),
            URIRef("https://l/A2"): np.array([1.0, 0.0]),
            URIRef("https://r/A1"): np.array([1.0, 0.0]),
            URIRef("https://r/A2"): np.array([1.0, 0.0]),
            URIRef("https://types/Thing"): np.array([0.0, 1.0]),
            URIRef(str(RDF.type)): np.array([0.0, 1.0]),
        },
    )

    result = matcher.match(
        left_graph,
        right_graph,
        regime=MatchRegime.ONTOLOGY_LOOSE,
        ground_truth_side=GroundTruthSide.RIGHT,
    )
    pairs = [
        (str(item.left_entity), str(item.right_entity))
        for item in result.entity_matches
    ]
    assert pairs == sorted(pairs)
