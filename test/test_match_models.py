from __future__ import annotations

from rdflib import URIRef

from ontocast.tool.agg.match_models import (
    EntityCluster,
    EntityMatch,
    GraphEntityMember,
    coerce_uri_ref,
)


def test_coerce_uri_ref_accepts_string() -> None:
    uri = coerce_uri_ref("http://example.org/entity")
    assert isinstance(uri, URIRef)
    assert str(uri) == "http://example.org/entity"


def test_graph_entity_member_parses_json_string_entity() -> None:
    member = GraphEntityMember.model_validate(
        {
            "graph_id": "gt",
            "entity": "http://text2kg.bench/alan_wright",
            "similarity": 0.9,
        }
    )
    assert isinstance(member.entity, URIRef)
    assert str(member.entity) == "http://text2kg.bench/alan_wright"


def test_entity_match_parses_json_string_entities() -> None:
    match = EntityMatch.model_validate(
        {
            "predicted_entity": "http://predicted.example/a",
            "gt_entity": "http://gt.example/a",
            "similarity": 1.0,
        }
    )
    assert isinstance(match.predicted_entity, URIRef)
    assert isinstance(match.gt_entity, URIRef)
    assert str(match.predicted_entity) == "http://predicted.example/a"
    assert str(match.gt_entity) == "http://gt.example/a"


def test_entity_match_keeps_uriref_when_constructed_with_uriref() -> None:
    predicted = URIRef("http://predicted.example/a")
    gt = URIRef("http://gt.example/a")
    match = EntityMatch(predicted_entity=predicted, gt_entity=gt, similarity=1.0)
    assert isinstance(match.predicted_entity, URIRef)
    assert isinstance(match.gt_entity, URIRef)
    assert match.predicted_entity is predicted
    assert match.gt_entity is gt


def test_entity_cluster_accepts_string_entities_in_members() -> None:
    cluster = EntityCluster.model_validate(
        {
            "members": [
                {
                    "graph_id": "predicted",
                    "entity": "http://predicted.example/a",
                },
                {"graph_id": "gt", "entity": "http://gt.example/a"},
            ]
        }
    )
    assert isinstance(cluster.members[0].entity, URIRef)


def test_derive_matches_request_accepts_string_entities_in_clusters() -> None:
    from ontocast.cli.server import DeriveMatchesRequest

    request = DeriveMatchesRequest.model_validate(
        {
            "clusters": [
                {
                    "members": [
                        {
                            "graph_id": "predicted",
                            "entity": "http://predicted.example/a",
                        },
                        {"graph_id": "gt", "entity": "http://gt.example/a"},
                    ]
                }
            ],
            "predicted_graph_id": "predicted",
            "gt_graph_id": "gt",
        }
    )
    assert isinstance(request.clusters[0].members[0].entity, URIRef)
