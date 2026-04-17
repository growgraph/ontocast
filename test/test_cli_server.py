import pytest

from ontocast.api.schemas import ProcessResultData
from ontocast.cli.server import (
    parse_ontology_context_mode_param,
    parse_ontology_selection_policy_param,
    validate_ontology_context_mode,
)
from ontocast.onto.enum import OntologyContextMode, OntologySelectionPolicy


def test_parse_ontology_context_mode_param_accepts_request_override() -> None:
    result = parse_ontology_context_mode_param(
        "retrieved_induced_graph",
        OntologyContextMode.FULL_TTL,
    )

    assert result == OntologyContextMode.RETRIEVED_INDUCED_GRAPH


def test_validate_ontology_context_mode_rejects_missing_vector_store() -> None:
    with pytest.raises(ValueError, match="requires configured vector store"):
        validate_ontology_context_mode(
            OntologyContextMode.RETRIEVED_INDUCED_GRAPH,
            OntologySelectionPolicy.STRICT_RETRIEVAL,
            None,
            None,
        )


def test_validate_ontology_context_mode_allows_full_ttl_without_vector_store() -> None:
    validate_ontology_context_mode(
        OntologyContextMode.FULL_TTL,
        OntologySelectionPolicy.STRICT_RETRIEVAL,
        None,
        None,
    )


def test_validate_ontology_context_mode_allows_fallback_policy_without_vector_store() -> (
    None
):
    validate_ontology_context_mode(
        OntologyContextMode.RETRIEVED_INDUCED_GRAPH,
        OntologySelectionPolicy.RETRIEVAL_WITH_LLM_FALLBACK,
        None,
        None,
    )


def test_parse_ontology_selection_policy_param_accepts_override() -> None:
    result = parse_ontology_selection_policy_param(
        "llm_selector_only",
        OntologySelectionPolicy.RETRIEVAL_WITH_LLM_FALLBACK,
    )
    assert result == OntologySelectionPolicy.LLM_SELECTOR_ONLY


def test_process_result_data_uses_artifacts_and_deprecates_singular_ontology() -> None:
    payload = ProcessResultData(
        facts="",
        ontology=None,
        ontology_artifacts=[{"iri": "https://example.org/o", "ttl": ""}],
    )
    assert payload.ontology is None
    assert len(payload.ontology_artifacts) == 1
