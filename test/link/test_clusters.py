"""Read-time clusters, the union view, the CLI and the HTTP route."""

import asyncio
from typing import Any

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from ontocast.link import EntityCluster, build, cluster_of
from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager
from ontocast.toolbox import ToolBox
from test.link.helpers import (
    DOC1,
    DOC2,
    DOC3,
    DOMAIN,
    VALUE_CLASSES,
    document,
    e,
    make_tools,
    store_documents,
)

pytestmark = pytest.mark.unit

DOCS = [DOC1, DOC2, DOC3]


def _linked(docs: list[str]) -> ToolBox:
    tools = make_tools()
    store = tools.triple_store_manager
    assert store is not None
    asyncio.run(store_documents(tools, docs))
    asyncio.run(
        build(
            store,
            tools.ontology_manager.identity_declarations(),
            domain=DOMAIN,
            value_classes=VALUE_CLASSES,
        )
    )
    return tools


def _cluster(tools: ToolBox, iri: str) -> EntityCluster | None:
    store = tools.triple_store_manager
    assert store is not None
    return asyncio.run(
        cluster_of(store, tools.ontology_manager.identity_declarations(), iri)
    )


def test_one_entity_across_documents_is_one_cluster() -> None:
    cluster = _cluster(_linked(DOCS), str(e(DOC1, "echr")))
    assert cluster is not None
    members = {m.iri: m for m in cluster.members}
    assert set(members) == {str(e(DOC1, "echr")), str(e(DOC2, "court"))}
    assert members[str(e(DOC1, "echr"))].reason == "self"
    assert members[str(e(DOC2, "court"))].reason == "label"
    assert members[str(e(DOC2, "court"))].graph == DOC2


def test_a_functional_conflict_keeps_a_candidate_apart() -> None:
    cluster = _cluster(_linked(DOCS), str(e(DOC1, "echr")))
    assert cluster is not None
    kept = {k.iri: k.veto for k in cluster.kept_apart}
    assert kept == {str(e(DOC3, "echr")): f"functional {DOMAIN}/lk#seat"}


def test_keys_and_identifiers_cluster_across_labels() -> None:
    tools = _linked(DOCS)
    paper = _cluster(tools, str(e(DOC2, "study")))
    assert paper is not None
    assert {m.iri for m in paper.members} == {
        str(e(DOC1, "paper")),
        str(e(DOC2, "study")),
    }
    author = _cluster(tools, str(e(DOC1, "janeDoe")))
    assert author is not None
    assert {m.reason for m in author.members} == {"self", "identifier"}


def test_an_unlinked_entity_is_its_own_cluster() -> None:
    cluster = _cluster(_linked(DOCS), str(e(DOC1, "sample1")))
    assert cluster is not None
    assert [m.iri for m in cluster.members] == [str(e(DOC1, "sample1"))]


def test_an_unknown_entity_has_no_cluster() -> None:
    assert _cluster(_linked(DOCS), f"{DOMAIN}/doc/none/x") is None


def test_clusters_do_not_depend_on_store_order() -> None:
    forward = _cluster(_linked(DOCS), str(e(DOC2, "court")))
    backward = _cluster(_linked(list(reversed(DOCS))), str(e(DOC2, "court")))
    assert forward == backward


def test_the_union_view_joins_across_documents() -> None:
    tools = _linked(DOCS)
    store = tools.triple_store_manager
    assert store is not None
    query = (
        "PREFIX skos: <http://www.w3.org/2004/02/skos/core#> "
        "SELECT DISTINCT ?a ?b WHERE { ?a skos:closeMatch ?m . ?b skos:closeMatch ?m . "
        f"FILTER(?a = <{e(DOC1, 'paper')}> && ?a != ?b) }}"
    )
    assert asyncio.run(store.aselect(query, store="facts")) == []
    rows = asyncio.run(store.aselect(query, store="facts", union=True))
    assert rows == [{"a": str(e(DOC1, "paper")), "b": str(e(DOC2, "study"))}]


def test_fuseki_asks_for_the_union_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        posted.append(kwargs.get("data", {}))
        return httpx.Response(
            200, json={"results": {"bindings": []}}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    manager = FusekiTripleStoreManager(uri="http://fuseki.invalid:3030")
    asyncio.run(manager.aselect("SELECT * WHERE { ?s ?p ?o }", store="facts"))
    asyncio.run(
        manager.aselect("SELECT * WHERE { ?s ?p ?o }", store="facts", union=True)
    )
    assert "default-graph-uri" not in posted[0]
    assert posted[1]["default-graph-uri"] == "urn:x-arq:UnionGraph"


def test_link_build_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from ontocast.cli import server as server_module

    tools = make_tools()
    asyncio.run(store_documents(tools, DOCS))

    class Runtime:
        def __init__(self) -> None:
            self.tools = tools
            self.config = tools.config

    async def bootstrap(**_: Any) -> Runtime:
        return Runtime()

    monkeypatch.setattr(server_module, "_abootstrap_tools", bootstrap)
    monkeypatch.setattr(
        type(tools.get_atomic_tools()), "value_classes", lambda self: VALUE_CLASSES
    )
    monkeypatch.setattr(tools.config.tool_config.domain, "current_domain", DOMAIN)
    runner = CliRunner()
    first = runner.invoke(server_module.cli, ["link", "build"])
    assert first.exit_code == 0, first.output
    assert "3 document(s) linked, 7 link(s), 0 links graph(s) dropped" in first.output
    removed = runner.invoke(server_module.cli, ["link", "remove", "--graph", DOC1])
    assert removed.exit_code == 0, removed.output
    assert "1 links graph(s) dropped" in removed.output
    added = runner.invoke(server_module.cli, ["link", "add", "--graph", DOC1])
    assert "1 document(s) linked, 3 link(s)" in added.output


def test_cluster_route() -> None:
    from fastapi import FastAPI

    from ontocast.api.entities import build_entities_router
    from ontocast.config import ServerConfig

    tools = _linked(DOCS)
    app = FastAPI()
    app.include_router(
        build_entities_router(
            tools,
            active_tenant="t",
            active_project="p",
            server_config=ServerConfig(),
        )
    )
    client = TestClient(app)
    hit = client.get("/entities/cluster", params={"iri": str(e(DOC1, "echr"))})
    assert hit.status_code == 200, hit.text
    body = hit.json()
    assert {m["iri"] for m in body["members"]} == {
        str(e(DOC1, "echr")),
        str(e(DOC2, "court")),
    }
    assert body["kept_apart"][0]["iri"] == str(e(DOC3, "echr"))
    miss = client.get("/entities/cluster", params={"iri": f"{DOMAIN}/doc/none/x"})
    assert miss.status_code == 404


def test_document_fixture_has_the_value_node() -> None:
    """Guards the fixture: the quantity exists, so 'never linked' tests something."""
    assert len(list(document(DOC1).subjects())) > 5
