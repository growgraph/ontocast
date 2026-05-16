from __future__ import annotations

from rdflib import URIRef

from ontocast.tool.agg.match_derivation import derive_pair_matches
from ontocast.tool.agg.match_models import EntityCluster, GraphEntityMember


def test_derive_single_member_per_side() -> None:
    clusters = [
        EntityCluster(
            members=[
                GraphEntityMember(
                    graph_id="predicted:doc.ttl",
                    entity=URIRef("https://pred/A"),
                    similarity=0.9,
                ),
                GraphEntityMember(
                    graph_id="gt:doc.ttl",
                    entity=URIRef("https://gt/A"),
                    similarity=0.9,
                ),
            ]
        )
    ]
    matches = derive_pair_matches(
        clusters, "predicted:doc.ttl", "gt:doc.ttl", similarity_threshold=0.0
    )
    assert len(matches) == 1
    assert str(matches[0].predicted_entity).endswith("/A")
    assert str(matches[0].gt_entity).endswith("/A")


def test_derive_ignores_unrelated_graph_ids() -> None:
    clusters = [
        EntityCluster(
            members=[
                GraphEntityMember(
                    graph_id="predicted:a.ttl",
                    entity=URIRef("https://pred/A"),
                ),
                GraphEntityMember(graph_id="gt:b.ttl", entity=URIRef("https://gt/B")),
            ]
        )
    ]
    matches = derive_pair_matches(clusters, "predicted:a.ttl", "gt:a.ttl")
    assert matches == []
