"""The effective-config record: complete, JSON-safe and free of credentials.

A run's record is what tells two arms apart for settings that make no LLM call
(aggregation, chunking, vector store), so it must cover every setting, and it
is written beside outputs that get shared, so it must never carry a secret.
"""

import json
import re
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

from ontocast.api import app as app_module
from ontocast.api.app import create_app
from ontocast.config import Config, ServerConfig
from ontocast.config.env_names import iter_settings_fields
from ontocast.config.snapshot import REDACTED, SECRET_FIELD_NAMES, config_snapshot
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

#: Field names that look like credentials and are not.
_NOT_SECRETS = {"prompt_cache_key", "natural_key_merge", "lexical_token_jaccard"}
_CREDENTIAL_LIKE = re.compile(r"(^|_)(key|auth|token|password|secret)(_|$)")


def _env_name(group_name: str, field_name: str) -> str:
    for field in iter_settings_fields(Config):
        if field.group.__name__ == group_name and field.name == field_name:
            return field.env_name
    raise KeyError((group_name, field_name))


def test_every_setting_is_recorded_and_json_safe() -> None:
    snapshot = config_snapshot(Config())

    assert set(snapshot) == {field.env_name for field in iter_settings_fields(Config)}
    json.dumps(snapshot)


def test_records_the_settings_that_make_no_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGG_ENABLED", "false")
    monkeypatch.setenv("CHUNK_CHUNKING_MODE", "naive")
    monkeypatch.setenv("ONTOLOGY_UPDATE_MODE", "insert_only")

    snapshot = config_snapshot(Config())

    assert snapshot["AGG_ENABLED"] is False
    assert snapshot["CHUNK_CHUNKING_MODE"] == "naive"
    assert snapshot["ONTOLOGY_UPDATE_MODE"] == "insert_only"
    assert "FACTS_ACCEPT_BLOCKING_SEVERITY" in snapshot
    assert "FACTS_COMPLETION_PASSES" in snapshot


def test_credentials_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_key = _env_name("LLMConfig", "api_key")
    monkeypatch.setenv(llm_key, "sk-very-secret")

    snapshot = config_snapshot(Config())

    assert snapshot[llm_key] == REDACTED
    assert "sk-very-secret" not in json.dumps(snapshot)


def test_url_credentials_are_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    fuseki_uri = _env_name("FusekiConfig", "uri")
    monkeypatch.setenv(fuseki_uri, "http://admin:hunter2@fuseki.local:3030/ds")

    snapshot = config_snapshot(Config())

    assert snapshot[fuseki_uri] == f"http://{REDACTED}@fuseki.local:3030/ds"


def test_every_credential_like_field_is_classified() -> None:
    """A new credential field must join SECRET_FIELD_NAMES or this list."""
    unclassified = {
        field.name
        for field in iter_settings_fields(Config)
        if _CREDENTIAL_LIKE.search(field.name)
        and field.name not in SECRET_FIELD_NAMES
        and field.name not in _NOT_SECRETS
    }
    assert unclassified == set()


def test_config_route_serves_the_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module, "create_agent_graph", lambda _tools, **_kwargs: SimpleNamespace()
    )
    monkeypatch.setenv("AGG_ENABLED", "false")
    tools = cast(
        ToolBox,
        SimpleNamespace(
            llm=None,
            config=Config(),
            chunker=SimpleNamespace(chunking_mode="naive"),
        ),
    )
    app = create_app(
        tools=tools,
        server_config=ServerConfig(),
        active_tenant="tenant-a",
        active_project="project-a",
    )

    response = TestClient(app).get("/config")

    assert response.status_code == 200
    body = response.json()
    assert body["chunking_mode"] == "naive"
    assert body["settings"]["AGG_ENABLED"] is False
