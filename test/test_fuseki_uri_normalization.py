"""Tests for Fuseki base URI normalization."""

import pytest

from ontocast.tool.triple_manager.fuseki import normalize_fuseki_server_uri

pytestmark = pytest.mark.unit


def test_normalize_strips_fragment_ui_style() -> None:
    assert (
        normalize_fuseki_server_uri("http://localhost:3032/#/dataset/my_dataset")
        == "http://localhost:3032"
    )


def test_normalize_strips_trailing_slash() -> None:
    assert (
        normalize_fuseki_server_uri("http://localhost:3032/") == "http://localhost:3032"
    )


def test_normalize_preserves_path_prefix() -> None:
    assert (
        normalize_fuseki_server_uri("http://example.com/rdf-proxy/fuseki/")
        == "http://example.com/rdf-proxy/fuseki"
    )


def test_normalize_none() -> None:
    assert normalize_fuseki_server_uri(None) is None


def test_normalize_malformed_unchanged() -> None:
    assert normalize_fuseki_server_uri("not-a-url") == "not-a-url"


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 403])
async def test_dataset_creation_refused_for_credentials_names_fuseki_auth(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """A protected Fuseki without FUSEKI_AUTH must fail at startup, by name."""
    import httpx

    from ontocast.tool.triple_manager.fuseki import (
        FusekiAccessError,
        FusekiTripleStoreManager,
    )

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _request: httpx.Response(status))
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )
    monkeypatch.delenv("FUSEKI_AUTH", raising=False)
    manager = FusekiTripleStoreManager(uri="http://fuseki.invalid:3030")

    with pytest.raises(FusekiAccessError, match="FUSEKI_AUTH"):
        await manager.init_dataset("facts")
