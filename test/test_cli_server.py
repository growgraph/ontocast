from types import SimpleNamespace
from typing import cast

import pytest

from ontocast.api.schemas import ProcessResultData
from ontocast.cli.server import (
    parse_ontology_context_mode_param,
    validate_ontology_context_mode,
)
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.retrieval_capabilities import OntologyContextConfigError
from ontocast.toolbox import ToolBox


def test_parse_ontology_context_mode_param_accepts_request_override() -> None:
    result = parse_ontology_context_mode_param(
        "vector_retrieval",
        OntologyContextMode.FULL_TTL,
    )
    assert result == OntologyContextMode.VECTOR_RETRIEVAL


def _tools(vector_store: object | None, patch_retriever: object | None) -> ToolBox:
    return cast(
        ToolBox,
        SimpleNamespace(
            vector_store=vector_store,
            patch_retriever=patch_retriever,
        ),
    )


def test_validate_ontology_context_mode_rejects_vector_without_qdrant() -> None:
    with pytest.raises(OntologyContextConfigError, match="vector_retrieval"):
        validate_ontology_context_mode(
            OntologyContextMode.VECTOR_RETRIEVAL,
            _tools(None, None),
        )


def test_validate_ontology_context_mode_allows_full_ttl_without_vector_store() -> None:
    validate_ontology_context_mode(
        OntologyContextMode.FULL_TTL,
        _tools(None, None),
    )


def test_validate_ontology_context_mode_allows_vector_when_both_set() -> None:
    validate_ontology_context_mode(
        OntologyContextMode.VECTOR_RETRIEVAL,
        _tools(object(), object()),
    )


def test_process_result_data_uses_artifacts_and_deprecates_singular_ontology() -> None:
    payload = ProcessResultData(
        facts="",
        ontology=None,
        ontology_artifacts=[{"iri": "https://example.org/o", "ttl": ""}],
    )
    assert payload.ontology is None
    assert len(payload.ontology_artifacts) == 1
