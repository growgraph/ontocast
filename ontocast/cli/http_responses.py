"""JSON error bodies for HTTP routes (keeps FastAPI handlers thin)."""

from fastapi.responses import JSONResponse

from ontocast.api.schemas import StatusErrorBody
from ontocast.onto.retrieval_capabilities import (
    OntologyContextConfigError,
    VectorStoreUnavailableError,
)


def invalid_max_visits_response() -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=StatusErrorBody(
            error="max_visits must be an integer >= 1",
            error_type="ValidationError",
        ).model_dump(),
    )


def missing_fixed_catalog_ontology_id_response() -> JSONResponse:
    """400 when ontology_context_mode is fixed_single_ontology but id is absent."""
    return JSONResponse(
        status_code=400,
        content=StatusErrorBody(
            error=(
                "ontology_context_mode=fixed_single_ontology requires "
                "non-empty ontology_context_fixed_ontology_id (query, form field, or JSON)."
            ),
            error_type="ValidationError",
        ).model_dump(),
    )


def ontology_context_config_error_response(
    error: OntologyContextConfigError,
) -> JSONResponse:
    error_code = None
    status_code = 400
    if isinstance(error, VectorStoreUnavailableError):
        error_code = error.error_code
        status_code = 409
    return JSONResponse(
        status_code=status_code,
        content=StatusErrorBody(
            error=str(error),
            error_type=type(error).__name__,
            error_code=error_code,
        ).model_dump(),
    )
