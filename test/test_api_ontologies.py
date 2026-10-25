"""The ``/ontologies`` routes over a real ToolBox and its in-memory store.

Upload adopts a Turtle file as the root version of a new ontology; the list
shows one row per ontology with its head; the version history is read per
ontology; delete drops every version.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from ontocast.api.ontologies import build_ontology_router
from ontocast.config import Config, PathConfig, ServerConfig, ToolConfig
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

_IRI = "https://example.org/finance"

_TURTLE = f"""
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix fin: <{_IRI}#> .

<{_IRI}> a owl:Ontology ;
    dcterms:title "Finance" ;
    owl:versionInfo "2.1.0" .

fin:Account a owl:Class ; rdfs:label "Account" .
"""


@pytest.fixture
def client() -> TestClient:
    tools = ToolBox(
        Config(tool_config=ToolConfig(path_config=PathConfig(ontology_directory=None)))
    )
    app = FastAPI()
    app.include_router(
        build_ontology_router(
            tools,
            active_tenant="startup_t",
            active_project="startup_p",
            server_config=ServerConfig(),
        )
    )
    return TestClient(app)


def _upload(client: TestClient, turtle: str = _TURTLE) -> dict:
    response = client.post(
        "/ontologies",
        files={"file": ("finance.ttl", turtle.encode("utf-8"), "text/turtle")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_lists_one_row_per_ontology_with_its_head(client: TestClient) -> None:
    uploaded = _upload(client)
    assert uploaded["iri"] == _IRI
    assert uploaded["ontology_id"] == "finance"
    assert uploaded["version"] == "2.1.0"
    assert uploaded["version_iri"].startswith(f"{_IRI}#")

    rows = client.get("/ontologies").json()["ontologies"]

    assert len(rows) == 1
    (row,) = rows
    assert row["iri"] == _IRI
    assert row["ontology_id"] == "finance"
    assert row["title"] == "Finance"
    assert row["versions"] == 1
    assert row["diverged"] is False
    (head,) = row["heads"]
    assert head["version_iri"] == uploaded["version_iri"]
    assert head["version"] == "2.1.0"
    assert head["prior_version_iris"] == []
    assert head["removed"] is False


def test_a_second_upload_of_the_same_iri_is_a_conflict(client: TestClient) -> None:
    _upload(client)

    response = client.post(
        "/ontologies",
        files={"file": ("finance.ttl", _TURTLE.encode("utf-8"), "text/turtle")},
    )

    assert response.status_code == 409, response.text
    assert "already exists" in response.text
    assert len(client.get("/ontologies").json()["ontologies"]) == 1


def test_versions_lists_the_root(client: TestClient) -> None:
    uploaded = _upload(client)

    response = client.get("/ontologies/finance/versions")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["iri"] == _IRI
    assert body["ontology_id"] == "finance"
    assert [v["version_iri"] for v in body["versions"]] == [uploaded["version_iri"]]
    assert body["versions"][0]["prior_version_iris"] == []


def test_versions_of_an_unknown_ontology_is_a_404(client: TestClient) -> None:
    assert client.get("/ontologies/absent/versions").status_code == 404


def test_delete_drops_every_version(client: TestClient) -> None:
    _upload(client)

    response = client.delete(f"/ontologies/{quote(_IRI, safe='')}")

    assert response.status_code == 200, response.text
    assert response.json()["iri"] == _IRI
    assert client.get("/ontologies").json()["ontologies"] == []
    assert client.get("/ontologies/finance/versions").status_code == 404
    # The IRI is free again.
    assert _upload(client)["iri"] == _IRI


_TURTLE_V2 = f"""
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix fin: <{_IRI}#> .

<{_IRI}> a owl:Ontology ;
    dcterms:title "Finance" ;
    owl:versionInfo "2.2.0" .

fin:Account a owl:Class ; rdfs:label "Account" .
fin:Ledger a owl:Class ; rdfs:label "Ledger" .
"""


def _version_key(version_iri: str) -> str:
    return version_iri.rsplit("#", 1)[-1]


def test_put_uploads_a_new_version_under_the_current_head(client: TestClient) -> None:
    root = _upload(client)

    response = client.put(
        f"/ontologies/{quote(_IRI, safe='')}",
        files={"file": ("finance.ttl", _TURTLE_V2.encode("utf-8"), "text/turtle")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["iri"] == _IRI
    assert body["version_iri"] != root["version_iri"]
    assert body["version"] == "2.2.0", "author label wins when higher than the head"

    versions = client.get("/ontologies/finance/versions").json()["versions"]
    assert len(versions) == 2
    tip = next(v for v in versions if v["version_iri"] == body["version_iri"])
    assert tip["prior_version_iris"] == [root["version_iri"]]
    assert tip["removed"] is False


def test_put_of_unchanged_content_writes_nothing(client: TestClient) -> None:
    root = _upload(client)

    response = client.put(
        f"/ontologies/{quote(_IRI, safe='')}",
        files={"file": ("finance.ttl", _TURTLE.encode("utf-8"), "text/turtle")},
    )

    assert response.status_code == 200, response.text
    assert response.json()["version_iri"] == root["version_iri"]
    assert len(client.get("/ontologies/finance/versions").json()["versions"]) == 1


def test_get_version_returns_turtle(client: TestClient) -> None:
    root = _upload(client)
    key = _version_key(root["version_iri"])

    response = client.get(f"/ontologies/finance/versions/{key}")

    assert response.status_code == 200, response.text
    assert "text/turtle" in response.headers["content-type"]
    assert "Account" in response.text
    assert root["version_iri"] in response.text or "owl:versionIRI" in response.text


def test_delete_version_tombstones_and_falls_back_to_prior_head(
    client: TestClient,
) -> None:
    root = _upload(client)
    tip = client.put(
        f"/ontologies/{quote(_IRI, safe='')}",
        files={"file": ("finance.ttl", _TURTLE_V2.encode("utf-8"), "text/turtle")},
    ).json()
    tip_key = _version_key(tip["version_iri"])

    response = client.delete(f"/ontologies/finance/versions/{tip_key}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["version_iri"] == tip["version_iri"]
    assert [h["version_iri"] for h in body["heads"]] == [root["version_iri"]]

    versions = client.get("/ontologies/finance/versions").json()["versions"]
    removed = next(v for v in versions if v["version_iri"] == tip["version_iri"])
    assert removed["removed"] is True

    turtle = client.get(f"/ontologies/finance/versions/{tip_key}")
    assert turtle.status_code == 200
    assert "Ledger" not in turtle.text


def test_diff_reports_added_and_removed_triples(client: TestClient) -> None:
    root = _upload(client)
    tip = client.put(
        f"/ontologies/{quote(_IRI, safe='')}",
        files={"file": ("finance.ttl", _TURTLE_V2.encode("utf-8"), "text/turtle")},
    ).json()

    response = client.get(
        "/ontologies/finance/diff",
        params={
            "from": _version_key(root["version_iri"]),
            "to": _version_key(tip["version_iri"]),
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["from_iri"] == root["version_iri"]
    assert body["to_iri"] == tip["version_iri"]
    assert body["added_count"] >= 1
    assert any("Ledger" in line for line in body["added"])
    assert body["removed_count"] == 0
