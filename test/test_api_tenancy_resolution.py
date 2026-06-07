"""Tests for shared API tenancy resolution."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.testclient import TestClient

from ontocast.api.ontologies import build_ontology_router
from ontocast.api.tenancy_resolution import (
    apply_request_tenancy,
    request_has_tenancy_query_params,
    resolve_tenant_project,
    stores_use_tenancy_partitions,
)
from ontocast.config import ServerConfig
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.tenancy import DEFAULT_PROJECT, DEFAULT_TENANT
from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager
from ontocast.toolbox import ToolBox


def _http_request(query_string: bytes) -> Request:
    scope: dict = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/ontologies",
        "raw_path": b"/ontologies",
        "root_path": "",
        "scheme": "http",
        "query_string": query_string,
        "headers": [],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 80),
    }

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    return Request(scope, receive)


def test_resolve_tenant_project_uses_defaults_for_none() -> None:
    t, p = resolve_tenant_project(None, None)
    assert t == DEFAULT_TENANT
    assert p == DEFAULT_PROJECT


def test_resolve_tenant_project_strips() -> None:
    t, p = resolve_tenant_project("  a  ", "  b  ")
    assert t == "a"
    assert p == "b"


def test_request_has_tenancy_query_params() -> None:
    assert not request_has_tenancy_query_params(_http_request(b""))
    assert request_has_tenancy_query_params(_http_request(b"tenant=x"))
    assert request_has_tenancy_query_params(_http_request(b"project=y"))
    assert request_has_tenancy_query_params(_http_request(b"tenant=a&project=b"))


def test_stores_use_tenancy_partitions_true_when_vector_store() -> None:
    tools = cast(
        ToolBox,
        SimpleNamespace(vector_store=object(), triple_store_manager=None),
    )
    assert stores_use_tenancy_partitions(tools) is True


def test_stores_use_tenancy_partitions_true_when_fuseki_triple_store() -> None:
    fuseki = object.__new__(FusekiTripleStoreManager)
    tools = cast(
        ToolBox,
        SimpleNamespace(vector_store=None, triple_store_manager=fuseki),
    )
    assert stores_use_tenancy_partitions(tools) is True


def test_stores_use_tenancy_partitions_true_when_in_memory_triple_store() -> None:
    in_memory = InMemoryTripleStoreManager()
    tools = cast(
        ToolBox,
        SimpleNamespace(vector_store=None, triple_store_manager=in_memory),
    )
    assert stores_use_tenancy_partitions(tools) is True


def test_stores_use_tenancy_partitions_false_for_plain_object_triple_store() -> None:
    tools = cast(
        ToolBox,
        SimpleNamespace(vector_store=None, triple_store_manager=object()),
    )
    assert stores_use_tenancy_partitions(tools) is False


def test_apply_request_tenancy_no_query_uses_active() -> None:
    tools = SimpleNamespace(
        vector_store=None,
        triple_store_manager=object(),
        update_tenancy_with_vector_mode=AsyncMock(),
    )
    req = _http_request(b"")
    t, p = asyncio.run(
        apply_request_tenancy(
            req,
            cast(ToolBox, tools),
            active_tenant="startup_t",
            active_project="startup_p",
            initialize_vector_store=False,
        )
    )
    assert (t, p) == ("startup_t", "startup_p")
    tools.update_tenancy_with_vector_mode.assert_not_called()


def test_apply_request_tenancy_with_query_calls_update_when_partitioned() -> None:
    tools = SimpleNamespace(
        vector_store=object(),
        triple_store_manager=None,
        update_tenancy_with_vector_mode=AsyncMock(),
    )
    req = _http_request(b"tenant=acme&project=p1")
    t, p = asyncio.run(
        apply_request_tenancy(
            req,
            cast(ToolBox, tools),
            active_tenant="startup_t",
            active_project="startup_p",
            initialize_vector_store=True,
        )
    )
    assert (t, p) == ("acme", "p1")
    tools.update_tenancy_with_vector_mode.assert_awaited_once_with(
        "acme",
        "p1",
        initialize_vector_store=True,
        fail_on_vector_store_error=False,
    )


def test_apply_request_tenancy_resolves_without_update_when_not_partitioned() -> None:
    tools = SimpleNamespace(
        vector_store=None,
        triple_store_manager=object(),
        update_tenancy_with_vector_mode=AsyncMock(),
    )
    req = _http_request(b"tenant=acme")
    t, p = asyncio.run(
        apply_request_tenancy(
            req,
            cast(ToolBox, tools),
            active_tenant="startup_t",
            active_project="startup_p",
            initialize_vector_store=False,
        )
    )
    assert (t, p) == ("acme", DEFAULT_PROJECT)
    tools.update_tenancy_with_vector_mode.assert_not_called()


@pytest.mark.parametrize(
    ("ontology_context_mode", "initialize_vector_store"),
    [
        (OntologyContextMode.SELECTED_SINGLE_ONTOLOGY, False),
        (OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY, True),
        (OntologyContextMode.FIXED_SINGLE_ONTOLOGY, False),
    ],
)
def test_ontology_delete_with_tenant_query_calls_update_tenancy(
    ontology_context_mode: OntologyContextMode,
    initialize_vector_store: bool,
) -> None:
    update = AsyncMock()
    delete_by_iri = AsyncMock()
    tools = SimpleNamespace(
        vector_store=object(),
        triple_store_manager=None,
        update_tenancy_with_vector_mode=update,
        delete_ontology_by_iri=delete_by_iri,
    )
    app = FastAPI()
    app.include_router(
        build_ontology_router(
            cast(ToolBox, tools),
            active_tenant="startup_t",
            active_project="startup_p",
            server_config=ServerConfig(ontology_context_mode=ontology_context_mode),
        )
    )
    client = TestClient(app)
    r = client.delete(
        "/ontologies/https%3A%2F%2Fexample.org%2Fonto",
        params={"tenant": "acme", "project": "p1"},
    )
    assert r.status_code == 200
    update.assert_awaited_once_with(
        "acme",
        "p1",
        initialize_vector_store=initialize_vector_store,
        fail_on_vector_store_error=False,
    )
    delete_by_iri.assert_awaited_once_with("https://example.org/onto")
