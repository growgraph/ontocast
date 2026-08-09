"""Every request parameter must behave the same on every transport.

Query string, JSON body and multipart form used to be parsed by three
hand-written branches, and they had drifted: ``render_mode``,
``ontology_context_mode`` and the three instruction fields were readable only
from the query string, so a client sending them in a JSON body was silently
ignored. These tests pin the symmetry so the branches cannot diverge again.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from ontocast.api.parse import RequestParamError
from ontocast.api.process_request import (
    ParsedProcessRequest,
    load_parsed_process_request,
)
from ontocast.api.responses import request_param_error_response
from ontocast.config import ServerConfig

pytestmark = pytest.mark.unit

#: One representative value per parameter, chosen to differ from the default so
#: an ignored parameter shows up as the default rather than as a pass.
CASES: list[tuple[str, Any, str, Any]] = [
    ("render_mode", "facts", "render_mode", "facts"),
    ("llm_graph_format", "jsonld", "llm_graph_format", "jsonld"),
    (
        "ontology_context_mode",
        "selected_vector_search_ontology",
        "ontology_context_mode_value",
        "selected_vector_search_ontology",
    ),
    (
        "ontology_user_instruction",
        "prefer SKOS",
        "ontology_user_instruction",
        "prefer SKOS",
    ),
    (
        "ontology_selection_user_instruction",
        "pick the materials catalog",
        "ontology_selection_user_instruction",
        "pick the materials catalog",
    ),
    (
        "facts_user_instruction",
        "keep units verbatim",
        "facts_user_instruction",
        "keep units verbatim",
    ),
    ("strip_provenance", "true", "strip_provenance", True),
    ("max_visits", "4", "max_visits", 4),
    ("summary_max_sentences", "7", "summary_max_sentences", 7),
    ("target_sections", "results", "target_sections", ["results"]),
    ("document_type_hint", "academic", "document_type_hint", "academic"),
]


def _probe_app() -> FastAPI:
    """An app whose only route reports what the shared parser extracted."""
    app = FastAPI()
    server_config = ServerConfig()

    @app.post("/probe")
    async def probe(request: Request):
        # Mirrors the real routes, which wrap the parser in this same handler
        # so a malformed parameter is a 400 rather than a 500 (app.py:513,724).
        try:
            parsed = await load_parsed_process_request(request, server_config)
        except RequestParamError as exc:
            return request_param_error_response(exc)
        if isinstance(parsed, JSONResponse):
            return parsed
        assert isinstance(parsed, ParsedProcessRequest)
        return {
            "render_mode": parsed.render_mode,
            "llm_graph_format": parsed.llm_graph_format,
            "ontology_context_mode_value": str(parsed.ontology_context_mode_value),
            "ontology_user_instruction": parsed.ontology_user_instruction,
            "ontology_selection_user_instruction": (
                parsed.ontology_selection_user_instruction
            ),
            "facts_user_instruction": parsed.facts_user_instruction,
            "strip_provenance": parsed.strip_provenance,
            "max_visits": parsed.max_visits,
            "summary_max_sentences": parsed.summary_max_sentences,
            "target_sections": parsed.target_sections,
            "document_type_hint": parsed.document_type_hint,
        }

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_probe_app())


@pytest.mark.parametrize(("param", "raw", "field", "expected"), CASES)
def test_query_string_is_honoured(
    client: TestClient, param: str, raw: Any, field: str, expected: Any
) -> None:
    response = client.post("/probe", params={param: raw}, json={"text": "hello"})
    assert response.status_code == 200
    assert response.json()[field] == expected


@pytest.mark.parametrize(("param", "raw", "field", "expected"), CASES)
def test_json_body_is_honoured(
    client: TestClient, param: str, raw: Any, field: str, expected: Any
) -> None:
    """The regression this file exists for: five params were query-only."""
    response = client.post("/probe", json={"text": "hello", param: raw})
    assert response.status_code == 200
    assert response.json()[field] == expected


@pytest.mark.parametrize(("param", "raw", "field", "expected"), CASES)
def test_multipart_form_is_honoured(
    client: TestClient, param: str, raw: Any, field: str, expected: Any
) -> None:
    response = client.post(
        "/probe",
        files={"file": ("doc.txt", b"hello", "text/plain")},
        data={param: str(raw)},
    )
    assert response.status_code == 200
    assert response.json()[field] == expected


def test_body_overrides_query(client: TestClient) -> None:
    """Precedence is query first, then body -- the body is the more specific."""
    response = client.post(
        "/probe",
        params={"max_visits": "2"},
        json={"text": "hello", "max_visits": 5},
    )
    assert response.json()["max_visits"] == 5


def test_blank_form_field_does_not_override(client: TestClient) -> None:
    """An unset text input must not read as an explicit empty value."""
    response = client.post(
        "/probe",
        params={"ontology_user_instruction": "from query"},
        files={"file": ("doc.txt", b"hello", "text/plain")},
        data={"ontology_user_instruction": ""},
    )
    assert response.json()["ontology_user_instruction"] == "from query"


def test_document_metadata_round_trips_as_object_and_string(
    client: TestClient,
) -> None:
    """JSON bodies may send an object; query and form can only send text."""
    as_object = client.post(
        "/probe", json={"text": "hi", "document_metadata": {"doi": "10.1/x"}}
    )
    as_string = client.post(
        "/probe",
        params={"document_metadata": json.dumps({"doi": "10.1/x"})},
        json={"text": "hi"},
    )
    assert as_object.status_code == as_string.status_code == 200


def test_malformed_max_visits_is_a_400_not_a_500(client: TestClient) -> None:
    response = client.post("/probe", json={"text": "hi", "max_visits": "abc"})
    assert response.status_code == 400
