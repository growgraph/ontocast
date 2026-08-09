"""JSON error bodies for HTTP routes (keeps FastAPI handlers thin)."""

from fastapi.responses import JSONResponse

from ontocast.api.parse import RequestParamError
from ontocast.api.schemas import StatusErrorBody
from ontocast.onto.retrieval_capabilities import (
    OntologyContextConfigError,
    VectorStoreUnavailableError,
)


def request_param_error_response(error: RequestParamError) -> JSONResponse:
    """400 for any malformed request parameter.

    Replaces the previous per-message special cases, under which every
    parameter error but one returned 500.
    """
    return JSONResponse(
        status_code=400,
        content=StatusErrorBody(
            error=str(error),
            error_type="ValidationError",
            error_code=f"invalid_param:{error.param}",
        ).model_dump(),
    )


def document_conversion_error_response(
    error: Exception, stage: str | None
) -> JSONResponse:
    """422 when an uploaded document could not be converted.

    Both /process and /process_unit answer this way; previously only
    /process_unit did, so the same unreadable file produced 422 on one route
    and 500 on the other.
    """
    return JSONResponse(
        status_code=422,
        content=StatusErrorBody(
            error=str(error),
            error_type="DocumentConversionError",
            error_code=f"conversion_failed:{stage}" if stage else "conversion_failed",
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
