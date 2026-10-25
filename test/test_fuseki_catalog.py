"""Tests for the Fuseki catalog reads that avoid materializing every ontology.

These exercise the HTTP shapes only -- requests are intercepted, so no Fuseki
server is required.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import unquote

import httpx
import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.identity import OntologyIdentity
from ontocast.onto.ontology import Ontology, OntologyHeader
from ontocast.onto.ontology_header import mint_ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.triple_manager.core import (
    TripleStoreUnavailableError,
    TripleStoreWriteError,
)
from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager
from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever
from ontocast.toolbox import ToolBox
from test.ontology_helpers import next_version, stamped_ontology

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


def _terms(iri: str, *names: str) -> RDFGraph:
    graph = RDFGraph()
    graph.add((URIRef(iri), RDF.type, OWL.Ontology))
    for name in names:
        term = URIRef(f"{iri}#{name}")
        graph.add((term, RDF.type, OWL.Class))
        graph.add((term, RDFS.label, Literal(name)))
    return graph


def _stored_version(iri: str = _ONTO_IRI, *names: str) -> Ontology:
    """A stamped root version of the ontology ``iri`` declaring ``names``."""
    return stamped_ontology(
        _terms(iri, *(names or ("Thing",))), ontology_id=iri.rsplit("/", 1)[-1]
    )


def _header_rows(
    ontology: Ontology, *, priors: list[str] | None = None
) -> list[dict[str, Any]]:
    """The SPARQL JSON bindings the header query yields for one stored version.

    One row per prior version (the query's small cross product), or one row
    for a root.
    """
    header = OntologyHeader.from_ontology(ontology)
    assert header.created_at and header.modified_at
    base = {
        "g": header.graph_uri,
        "onto": header.iri,
        "versionIri": header.version_iri,
        "version": header.version,
        "prefix": header.ontology_id,
        "title": header.title,
        "created": header.created_at.isoformat(),
        "modified": header.modified_at.isoformat(),
    }
    if header.description:
        base["description"] = header.description
    prior_iris = header.prior_version_iris if priors is None else priors
    rows = [dict(base, prior=prior) for prior in prior_iris] or [base]
    return [{name: {"value": value} for name, value in row.items()} for row in rows]


class _RecordingTransport:
    """Captures SPARQL POSTs and answers Graph Store GETs with stored versions."""

    def __init__(
        self, bindings: list[dict[str, Any]], versions: list[Ontology]
    ) -> None:
        self.bindings = bindings
        self.turtle_by_graph = {
            cast(str, version.version_iri): version.graph.serialize(format="turtle")
            for version in versions
        }
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
        graph_uri = unquote(url.split("graph=", 1)[1])
        turtle = self.turtle_by_graph.get(graph_uri)
        if turtle is None:
            return httpx.Response(404, text="", request=httpx.Request("GET", url))
        return httpx.Response(200, text=turtle, request=httpx.Request("GET", url))


@pytest.fixture
def manager_and_transport(monkeypatch: pytest.MonkeyPatch):
    """A Fuseki manager whose httpx client is replaced by a recorder.

    ``versions`` are the stored versions the Graph Store serves; ``bindings``
    default to their header rows.
    """

    def _make(
        bindings: list[dict[str, Any]] | None = None,
        versions: Sequence[Ontology] = (),
    ):
        if bindings is None:
            bindings = [row for version in versions for row in _header_rows(version)]
        transport = _RecordingTransport(bindings, list(versions))
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

    with pytest.raises(TripleStoreUnavailableError):
        await manager.aselect("SELECT ?g WHERE { GRAPH ?g { ?s ?p ?o } }")


@pytest.mark.anyio
async def test_ontology_catalog_reads_headers_without_fetching_graphs(
    manager_and_transport,
) -> None:
    root = _stored_version()
    merged = next_version(root, _terms(_ONTO_IRI, "Thing", "Gadget"))
    root_iri = cast(str, root.version_iri)
    other_prior = f"{_ONTO_IRI}#other"
    manager, transport = manager_and_transport(
        _header_rows(merged, priors=[root_iri, other_prior])
    )

    headers = await manager.afetch_ontology_catalog()

    # The two prior rows are one ontology version, not two.
    assert len(headers) == 1
    assert headers[0].iri == _ONTO_IRI
    assert headers[0].graph_uri == merged.version_iri
    assert headers[0].version_iri == merged.version_iri
    assert headers[0].prior_version_iris == sorted([root_iri, other_prior])
    assert headers[0].ontology_id == "catalog"
    assert headers[0].version == merged.version
    assert transport.fetched_graph_urls == []
    assert manager.catalog_io_stats()["graph_fetches"] == 0


@pytest.mark.anyio
async def test_catalog_rejects_a_graph_without_the_version_header(
    manager_and_transport,
) -> None:
    """A foreign graph in the ontologies dataset is a fault, never a gap."""
    from ontocast.onto.ontology_header import CatalogContractError

    manager, _ = manager_and_transport(
        [
            {
                "g": {"value": _GRAPH_URI},
                "onto": {"value": _ONTO_IRI},
                "version": {"value": "1.0.0"},
            }
        ]
    )

    with pytest.raises(CatalogContractError) as excinfo:
        await manager.afetch_ontology_catalog()
    assert excinfo.value.graph_uri == _GRAPH_URI


@pytest.mark.anyio
async def test_afetch_ontologies_by_iri_fetches_only_requested_graph(
    manager_and_transport,
) -> None:
    wanted = _stored_version()
    other = _stored_version("https://example.org/other")
    manager, transport = manager_and_transport(versions=[wanted, other])

    ontologies = await manager.afetch_ontologies_by_iri([_ONTO_IRI])

    assert [onto.iri for onto in ontologies] == [_ONTO_IRI]
    assert ontologies[0].version_iri == wanted.version_iri
    assert len(transport.fetched_graph_urls) == 1
    assert (
        cast(str, wanted.version_iri).replace("#", "%23")
        in (transport.fetched_graph_urls[0])
    )
    assert manager.catalog_io_stats()["full_catalog_fetches"] == 0


@pytest.mark.anyio
async def test_afetch_ontologies_by_iri_skips_io_when_nothing_matches(
    manager_and_transport,
) -> None:
    manager, transport = manager_and_transport(versions=[_stored_version()])

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
    served = _stored_version(f"{_ONTO_IRI}-a")
    lost = _stored_version(f"{_ONTO_IRI}-b")
    manager, transport = manager_and_transport(versions=[served, lost])
    serve = transport.get

    async def get(_self: Any, url: str, **kwargs: Any) -> httpx.Response:
        if "-b" in url:
            return httpx.Response(503, text="", request=httpx.Request("GET", url))
        return await serve(url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "get", get)

    ontologies = await manager.afetch_ontologies()

    assert [onto.iri for onto in ontologies] == [served.iri]
    assert manager.last_catalog_was_complete() is False


@pytest.mark.anyio
async def test_complete_catalog_fetch_reports_authoritative(
    manager_and_transport,
) -> None:
    manager, _ = manager_and_transport(versions=[_stored_version()])

    ontologies = await manager.afetch_ontologies()

    assert [onto.iri for onto in ontologies] == [_ONTO_IRI]
    assert manager.last_catalog_was_complete() is True


@pytest.mark.anyio
async def test_aserialize_refuses_an_unstamped_ontology(manager_and_transport) -> None:
    manager, _ = manager_and_transport([])
    unstamped = Ontology(graph=_terms(_ONTO_IRI, "Thing"), iri=_ONTO_IRI)

    with pytest.raises(ValueError, match="not stamped as a version"):
        await manager.aserialize(unstamped)


def _refusing_put(monkeypatch: pytest.MonkeyPatch, status: int) -> list[str]:
    """Answer every Graph Store PUT with ``status``; returns the URLs seen."""
    put_urls: list[str] = []

    async def put(_self: Any, url: str, **kwargs: Any) -> httpx.Response:
        put_urls.append(url)
        return httpx.Response(
            status, text="disk full", request=httpx.Request("PUT", url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "put", put)
    return put_urls


@pytest.mark.anyio
async def test_a_refused_write_raises_a_write_error(
    manager_and_transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = manager_and_transport([])
    put_urls = _refusing_put(monkeypatch, 500)
    version = _stored_version()

    with pytest.raises(TripleStoreWriteError) as excinfo:
        await manager.aserialize(version)

    assert excinfo.value.graph_uri == version.version_iri
    assert excinfo.value.status == 500
    assert len(put_urls) == 1
    assert "/ontologies/data?graph=" in put_urls[0]


@pytest.mark.anyio
async def test_a_refused_write_leaves_no_version_registered_or_indexed(
    manager_and_transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store is written before anything downstream can point at a version."""
    manager, _ = manager_and_transport([])
    _refusing_put(monkeypatch, 500)

    indexed: list[Ontology] = []
    ontology_manager = OntologyManager()
    ontology_manager.register_vector_store(
        cast(
            OntologyPatchRetriever,
            SimpleNamespace(
                vector_store=SimpleNamespace(
                    reindex_ontology=lambda o: indexed.append(o) or 1
                )
            ),
        )
    )
    artifact = mint_ontology(
        OntologyIdentity(ontology_id="catalog", iri=_ONTO_IRI, title="Catalog"),
        _terms(_ONTO_IRI, "Thing"),
    )
    state = AgentState()
    state.reduced_ontology_artifacts = [artifact]
    stub = SimpleNamespace(
        ontology_manager=ontology_manager, triple_store_manager=manager
    )

    with pytest.raises(TripleStoreWriteError):
        await ToolBox.aserialize(cast(ToolBox, stub), state)

    assert ontology_manager.get_heads_by_iri(_ONTO_IRI) == []
    assert ontology_manager.get_ontology_versions_by_iri(_ONTO_IRI) == []
    assert indexed == []
    assert state.ontology_versions_written == []


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
