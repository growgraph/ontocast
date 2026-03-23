"""Tests for Fuseki base URI normalization."""

from ontocast.tool.triple_manager.fuseki import normalize_fuseki_server_uri


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
