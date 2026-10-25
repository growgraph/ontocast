"""JSON error bodies for HTTP routes (keeps FastAPI handlers thin)."""

from fastapi.responses import JSONResponse

from ontocast.api.parse import RequestParamError
from ontocast.api.schemas import ProcessErrorResponse, StatusErrorBody
from ontocast.onto.retrieval_capabilities import (
    OntologyContextConfigError,
    VectorStoreUnavailableError,
)
from ontocast.onto.run_meter import BudgetExceededError
from ontocast.tool.chunk.prepare import SectionSelectionEmptyError


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


def conversion_failure(error: Exception, stage: str | None) -> ProcessErrorResponse:
    """A document that could not be converted (its item answers 422)."""
    return ProcessErrorResponse(
        error=str(error),
        error_type="DocumentConversionError",
        error_code=f"conversion_failed:{stage}" if stage else "conversion_failed",
    )


def section_selection_failure(
    error: SectionSelectionEmptyError,
) -> ProcessErrorResponse:
    """A section selection that matched nothing in this document (422).

    Not 400: the parameters are individually well-formed and only fail against
    this particular document once it has been classified.
    """
    return ProcessErrorResponse(
        error=str(error),
        error_type="SectionSelectionEmpty",
        error_code=f"empty_section_selection:{error.param}",
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


def budget_failure(error: BudgetExceededError) -> ProcessErrorResponse:
    """A document that reached its token cap before it finished (422).

    ``error_details.run_meter`` carries the meter's usage at the refusal.
    """
    return ProcessErrorResponse(
        error=str(error),
        error_type="BudgetExceededError",
        error_code="budget_exceeded",
        error_details={"run_meter": dict(error.snapshot)},
    )
