"""Tests for the Fuseki catalog reads that avoid materializing every ontology.

These exercise the HTTP shapes only -- requests are intercepted, so no Fuseki
server is required.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from ontocast.tool.triple_manager.core import TripleStoreUnavailableError
from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager

pytestmark = pytest.mark.unit

_ONTO_IRI = "https://example.org/catalog"
_GRAPH_URI = f"{_ONTO_IRI}#abc123"

_TURTLE = f"""
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <{_ONTO_IRI}#> .

<{_ONTO_IRI}> a owl:Ontology ; owl:versionInfo "1.0.0" .
ex:Thing a owl:Class ; rdfs:label "Thing" .
"""


class _RecordingTransport:
    """Captures SPARQL POSTs and answers Graph Store GETs with fixed Turtle."""

    def __init__(self, bindings: list[dict[str, Any]]) -> None:
        self.bindings = bindings
        self.posted_queries: list[str] = []
        self.posted_urls: list[str] = []
        self.fetched_graph_urls: list[str] = []

    async def post(
        self, url: str, data: dict[str, str], **kwargs: Any
    ) -> httpx.Response:
        # Dataset administration also POSTs here; only SPARQL carries a query.
        if "query" not in data:
            return httpx.Response(200, request=httpx.Request("POST", url))
        self.posted_urls.append(url)
        self.posted_queries.append(data["query"])
        return httpx.Response(
            200,
            json={"results": {"bindings": self.bindings}},
            request=httpx.Request("POST", url),
        )

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.fetched_graph_urls.append(url)
        return httpx.Response(200, text=_TURTLE, request=httpx.Request("GET", url))


@pytest.fixture
def manager_and_transport(monkeypatch: pytest.MonkeyPatch):
    """A Fuseki manager whose httpx client is replaced by a recorder."""

    def _make(bindings: list[dict[str, Any]]):
        transport = _RecordingTransport(bindings)
        manager = FusekiTripleStoreManager(
            uri="http://fuseki.invalid:3030",
            dataset="facts",
            ontologies_dataset="ontologies",
        )
        monkeypatch.setattr(httpx.AsyncClient, "post", transport.post)
        monkeypatch.setattr(httpx.AsyncClient, "get", transport.get)
        return manager, transport

    return _make


def test_fuseki_supports_sparql_select() -> None:
    manager = FusekiTripleStoreManager(uri="http://fuseki.invalid:3030")
    assert manager.supports_sparql_select() is True


@pytest.mark.anyio
async def test_aselect_targets_ontologies_dataset(manager_and_transport) -> None:
    manager, transport = manager_and_transport([{"g": {"value": _GRAPH_URI}}])

    rows = await manager.aselect("SELECT ?g WHERE { GRAPH ?g { ?s ?p ?o } }")

    assert rows == [{"g": _GRAPH_URI}]
    assert transport.posted_urls == [
        "http://fuseki.invalid:3030/ontologies/sparql",
    ]


@pytest.mark.anyio
async def test_aselect_can_target_facts_dataset(manager_and_transport) -> None:
    manager, transport = manager_and_transport([])

    await manager.aselect("SELECT ?s WHERE { ?s ?p ?o }", store="facts")

    assert transport.posted_urls == ["http://fuseki.invalid:3030/facts/sparql"]


@pytest.mark.anyio
async def test_aselect_follows_tenancy_switch(manager_and_transport) -> None:
    manager, transport = manager_and_transport([])

    await manager.update_tenancy("acme", "demo")
    await manager.aselect("SELECT ?g WHERE { GRAPH ?g { ?s ?p ?o } }")

    assert transport.posted_urls == [
        "http://fuseki.invalid:3030/acme--demo--ontologies/sparql",
    ]


@pytest.mark.anyio
async def test_aselect_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors must surface, not become an empty (and thus ambiguous) result set."""

    async def failing_post(self, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", failing_post)
    manager = FusekiTripleStoreManager(uri="http://fuseki.invalid:3030")

    with pytest.raises(httpx.HTTPStatusError):
        await manager.aselect("SELECT ?g WHERE { GRAPH ?g { ?s ?p ?o } }")


@pytest.mark.anyio
async def test_ontology_catalog_reads_headers_without_fetching_graphs(
    manager_and_transport,
) -> None:
    manager, transport = manager_and_transport(
        [
            {
                "g": {"value": _GRAPH_URI},
                "onto": {"value": _ONTO_IRI},
                "version": {"value": "1.0.0"},
                "identifier": {"value": "hash:abc123"},
                "parent": {"value": "urn:hash:parent1"},
            },
            {
                "g": {"value": _GRAPH_URI},
                "onto": {"value": _ONTO_IRI},
                "version": {"value": "1.0.0"},
                "identifier": {"value": "hash:abc123"},
                "parent": {"value": "urn:hash:parent2"},
            },
        ]
    )

    headers = await manager.afetch_ontology_catalog()

    # The two parent rows are one ontology version, not two.
    assert len(headers) == 1
    assert headers[0].iri == _ONTO_IRI
    assert headers[0].graph_uri == _GRAPH_URI
    assert headers[0].hash == "abc123"
    assert headers[0].parent_hashes == ["parent1", "parent2"]
    assert transport.fetched_graph_urls == []
    assert manager.catalog_io_stats()["graph_fetches"] == 0


@pytest.mark.anyio
async def test_afetch_ontologies_by_iri_fetches_only_requested_graph(
    manager_and_transport,
) -> None:
    other_graph = "https://example.org/other#def456"
    manager, transport = manager_and_transport(
        [
            {
                "g": {"value": _GRAPH_URI},
                "onto": {"value": _ONTO_IRI},
                "identifier": {"value": "hash:abc123"},
            },
            {
                "g": {"value": other_graph},
                "onto": {"value": "https://example.org/other"},
                "identifier": {"value": "hash:def456"},
            },
        ]
    )

    ontologies = await manager.afetch_ontologies_by_iri([_ONTO_IRI])

    assert [onto.iri for onto in ontologies] == [_ONTO_IRI]
    assert len(transport.fetched_graph_urls) == 1
    assert _GRAPH_URI.replace("#", "%23") in transport.fetched_graph_urls[0]
    assert manager.catalog_io_stats()["full_catalog_fetches"] == 0


@pytest.mark.anyio
async def test_afetch_ontologies_by_iri_skips_io_when_nothing_matches(
    manager_and_transport,
) -> None:
    manager, transport = manager_and_transport(
        [{"g": {"value": _GRAPH_URI}, "onto": {"value": _ONTO_IRI}}]
    )

    assert await manager.afetch_ontologies_by_iri(["https://example.org/absent"]) == []
    assert transport.fetched_graph_urls == []


@pytest.mark.anyio
async def test_aconstruct_requests_turtle_and_parses_it(
    manager_and_transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONSTRUCT goes to /sparql with a Turtle Accept header, not the JSON one."""
    manager, _ = manager_and_transport([])
    seen: dict[str, Any] = {}

    async def post(
        _self: Any, url: str, data: dict[str, str], **kwargs: Any
    ) -> httpx.Response:
        seen["url"] = url
        seen["query"] = data["query"]
        seen["headers"] = kwargs.get("headers")
        assert "format" not in data
        return httpx.Response(200, text=_TURTLE, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    graph = await manager.aconstruct("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")

    assert seen["url"] == "http://fuseki.invalid:3030/ontologies/sparql"
    assert seen["headers"] == {"Accept": "text/turtle"}
    assert len(graph) == 4
    assert manager.catalog_io_stats()["construct_queries"] == 1


@pytest.mark.anyio
async def test_aconstruct_targets_the_facts_dataset_when_asked(
    manager_and_transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = manager_and_transport([])
    seen: dict[str, Any] = {}

    async def post(
        _self: Any, url: str, data: dict[str, str], **kwargs: Any
    ) -> httpx.Response:
        seen["url"] = url
        return httpx.Response(200, text="", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    await manager.aconstruct("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }", store="facts")
    assert seen["url"] == "http://fuseki.invalid:3030/facts/sparql"


@pytest.mark.anyio
async def test_aconstruct_raises_on_http_error(
    manager_and_transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty graph would be indistinguishable from 'nothing matched'."""
    manager, _ = manager_and_transport([])

    async def post(
        _self: Any, url: str, data: dict[str, str], **kwargs: Any
    ) -> httpx.Response:
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    with pytest.raises(httpx.HTTPStatusError):
        await manager.aconstruct("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")


@pytest.mark.anyio
async def test_catalog_listing_raises_instead_of_returning_empty(
    manager_and_transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty catalog is grounds for pruning the vector index.

    So a listing failure must raise rather than degrade to ``[]`` -- otherwise a
    transient network error is indistinguishable from "no ontologies stored"
    and ``ToolBox.initialize`` deletes every indexed ontology.
    """
    manager, _ = manager_and_transport([])

    async def post(
        _self: Any, url: str, data: dict[str, str], **kwargs: Any
    ) -> httpx.Response:
        if "query" not in data:
            return httpx.Response(200, request=httpx.Request("POST", url))
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    with pytest.raises(TripleStoreUnavailableError):
        await manager.afetch_ontologies()


@pytest.mark.anyio
async def test_partial_catalog_fetch_is_reported_as_incomplete(
    manager_and_transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A catalog missing some graphs must not read as authoritative.

    The absent ontologies would otherwise look like orphans to the vector-store
    prune and be deleted.
    """
    manager, _ = manager_and_transport(
        [
            {"g": {"value": f"{_ONTO_IRI}-a#v1"}},
            {"g": {"value": f"{_ONTO_IRI}-b#v1"}},
        ]
    )

    async def get(_self: Any, url: str, **kwargs: Any) -> httpx.Response:
        if "-b" in url:
            return httpx.Response(503, text="", request=httpx.Request("GET", url))
        return httpx.Response(200, text=_TURTLE, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", get)

    ontologies = await manager.afetch_ontologies()

    assert len(ontologies) == 1
    assert manager.last_catalog_was_complete() is False


@pytest.mark.anyio
async def test_complete_catalog_fetch_reports_authoritative(
    manager_and_transport,
) -> None:
    manager, _ = manager_and_transport([{"g": {"value": _GRAPH_URI}}])

    await manager.afetch_ontologies()

    assert manager.last_catalog_was_complete() is True


@pytest.mark.parametrize(
    ("auth", "expected"),
    [
        ("admin/secret", ("admin", "secret")),
        ("admin:secret", ("admin", "secret")),
        ("admin/pa:ss", ("admin", "pa:ss")),
        ("admin:pa/ss", ("admin", "pa/ss")),
    ],
)
def test_fuseki_auth_accepts_both_separator_forms(
    auth: str, expected: tuple[str, str]
) -> None:
    """The colon form previously parsed to *no* auth header at all."""
    manager = FusekiTripleStoreManager(uri="http://fuseki.invalid:3030", auth=auth)

    prepared = manager._prepare_auth()

    assert prepared is not None
    request = httpx.Request("GET", "http://fuseki.invalid:3030/")
    flow = prepared.auth_flow(request)
    authorized = next(flow)
    username, password = expected
    import base64

    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    assert authorized.headers["Authorization"] == f"Basic {token}"


def test_fuseki_auth_without_separator_is_rejected_at_construction() -> None:
    """Better a clear startup error than an unexplained 401 later."""
    with pytest.raises(ValueError, match="user:password"):
        FusekiTripleStoreManager(uri="http://fuseki.invalid:3030", auth="admin")
