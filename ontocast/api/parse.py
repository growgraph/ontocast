"""Shared HTTP query/body parsing for API server routes."""

import json
import logging
from enum import StrEnum
from typing import TypeVar

from ontocast.config.section_labels import (
    all_known_label_ids,
    normalise_user_section_label,
)
from ontocast.onto.enum import LLMGraphFormat, OntologyContextMode, RenderMode

logger = logging.getLogger(__name__)


class RequestParamError(ValueError):
    """A request parameter was malformed.

    Carries the offending parameter name so handlers can answer 400 by type
    rather than by comparing exception message strings -- the previous scheme,
    which meant every parameter error except one well-known message surfaced as
    a 500, and reworded messages silently changed status codes.

    Subclasses :class:`ValueError` so existing ``except ValueError`` callers
    (and library users calling the parsers directly) keep working.
    """

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param


_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _parse_enum_param(
    value: object,
    default: _EnumT,
    enum_cls: type[_EnumT],
    param: str,
) -> _EnumT:
    """Parse one string-enum request parameter, or raise.

    These three parameters used to warn and silently fall back to the default,
    while every other parser in this module raised
    :class:`RequestParamError` -> HTTP 400. That split meant a typo'd
    ``render_mode`` quietly ran the *wrong pipeline* and returned 200, which is
    a worse outcome than a rejected request. One contract now: unparseable
    means 400.

    Args:
        value: Raw parameter value; ``None`` or blank selects the default.
        default: Value to use when the parameter is absent.
        enum_cls: The enum to parse into.
        param: Parameter name, used in the error message.

    Returns:
        The parsed enum member.

    Raises:
        RequestParamError: The value was present but not a member.
    """
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    normalized = str(value).lower().strip()
    if not normalized:
        return default
    try:
        return enum_cls(normalized)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_cls)
        raise RequestParamError(param, f"{param} must be one of: {allowed}") from exc


def parse_render_mode_param(value, default: RenderMode) -> RenderMode:
    """Parse optional ``render_mode`` override from request params."""
    return _parse_enum_param(value, default, RenderMode, "render_mode")


def parse_llm_graph_format_param(
    value: str | LLMGraphFormat | None,
    default: LLMGraphFormat,
) -> LLMGraphFormat:
    """Parse optional ``llm_graph_format`` override from request params."""
    return _parse_enum_param(value, default, LLMGraphFormat, "llm_graph_format")


def parse_ontology_context_mode_param(
    value: str | OntologyContextMode | None,
    default: OntologyContextMode,
) -> OntologyContextMode:
    """Parse optional ``ontology_context_mode`` override from request params."""
    return _parse_enum_param(
        value, default, OntologyContextMode, "ontology_context_mode"
    )


def resolve_ontology_context_mode(
    requested_mode: OntologyContextMode,
    fixed_ontology_id: str,
) -> OntologyContextMode:
    """Resolve effective ontology context mode for a request.

    A non-empty ``ontology_context_fixed_ontology_id`` forces fixed catalog mode.
    This allows clients to pick fixed ontology context per request even when the
    server default mode differs.
    """
    if fixed_ontology_id.strip():
        return OntologyContextMode.FIXED_SINGLE_ONTOLOGY
    return requested_mode


def parse_strip_provenance_param(value: str | None) -> bool:
    """Parse ``strip_provenance`` query/form value."""
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    logger.warning(
        "Invalid strip_provenance %r, treating as false",
        value,
    )
    return False


def _normalise_section_tokens(raw_tokens: list[str]) -> tuple[list[str], list[str]]:
    """Map user-supplied section names onto canonical labels.

    Args:
        raw_tokens: Section names as the caller wrote them.

    Returns:
        The recognised canonical labels, and the tokens that matched nothing.
    """
    result: list[str] = []
    dropped: list[str] = []
    for token in raw_tokens:
        normalised = normalise_user_section_label(token)
        if normalised is None:
            logger.warning("Unrecognised section label %r \u2014 skipping", token)
            dropped.append(token)
        else:
            result.append(normalised)
    return result, dropped


def _resolve_section_tokens(raw_tokens: list[str], param: str) -> list[str]:
    """Normalise section tokens, rejecting a list where nothing was recognised.

    A partially recognised list still expresses a real intent, so unknown
    tokens are dropped with a warning. A list where *every* token is unknown is
    a caller mistake whose effect is the opposite of the request: the empty
    result reads downstream as an explicit "no sections", which *replaces* the
    resolved schema's defaults instead of adding to them. Failing here keeps a
    typo from silently disabling section handling the caller never touched.

    Args:
        raw_tokens: Non-empty section names as the caller wrote them.
        param: Parameter name, used in the error message.

    Returns:
        Canonical labels for the recognised tokens.

    Raises:
        RequestParamError: Tokens were supplied and none were recognised.
    """
    resolved, dropped = _normalise_section_tokens(raw_tokens)
    if raw_tokens and not resolved:
        known = ", ".join(sorted(all_known_label_ids()))
        raise RequestParamError(
            param,
            f"{param} recognised none of {dropped}; known section labels are: {known}",
        )
    return resolved


def parse_sections_list_param(
    value: str | list[str] | None, param: str = "sections"
) -> list[str]:
    """Parse a section list from comma-separated text or JSON array.

    Args:
        value: Raw parameter value.
        param: Parameter name, used only in error messages.

    Raises:
        RequestParamError: The value started with ``[`` but was not a JSON
            array, or tokens were supplied and none named a known section
            label.
    """
    if value is None:
        return []
    if isinstance(value, list):
        raw_tokens = [str(item).strip() for item in value if str(item).strip()]
        return _resolve_section_tokens(raw_tokens, param)
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RequestParamError(
                param, f"{param} must be valid JSON or a comma-separated list"
            ) from exc
        if not isinstance(parsed, list):
            raise RequestParamError(param, f"{param} JSON must be an array")
        raw_tokens = [str(item).strip() for item in parsed if str(item).strip()]
        return _resolve_section_tokens(raw_tokens, param)
    raw_tokens = [part.strip() for part in raw.split(",") if part.strip()]
    return _resolve_section_tokens(raw_tokens, param)


def parse_document_type_hint_param(value: str | None) -> str | None:
    """Parse optional document_type_hint; empty strings become None."""
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def parse_section_schema_id_param(value: str | None) -> str | None:
    """Parse optional section_schema_id; empty strings become None."""
    if value is None:
        return None
    stripped = str(value).strip().lower()
    return stripped or None


def parse_summary_max_sentences_param(value: str | int | None, default: int) -> int:
    """Parse optional summary_max_sentences (positive integer)."""
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise RequestParamError(
            "summary_max_sentences", "summary_max_sentences must be a positive integer"
        ) from exc
    if parsed < 1:
        raise RequestParamError(
            "summary_max_sentences", "summary_max_sentences must be a positive integer"
        )
    return parsed


def parse_max_visits_param(value: str | int | None, default: int) -> int:
    """Parse optional ``max_visits`` override from query/form/json metadata."""
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise RequestParamError(
            "max_visits", "max_visits must be an integer >= 1"
        ) from exc
    if parsed < 1:
        raise RequestParamError("max_visits", "max_visits must be an integer >= 1")
    return parsed


def parse_document_metadata_param(
    value: str | dict[str, object] | None,
) -> dict[str, object]:
    """Parse optional ``document_metadata`` from query/form/JSON.

    Accepts a dict (already-parsed JSON) or a JSON object string. Empty /
    missing values yield ``{}``.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if v is not None}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RequestParamError(
                "document_metadata", "document_metadata must be a JSON object"
            ) from exc
        if not isinstance(parsed, dict):
            raise RequestParamError(
                "document_metadata", "document_metadata must be a JSON object"
            )
        return {str(k): v for k, v in parsed.items() if v is not None}
    raise RequestParamError(
        "document_metadata", "document_metadata must be a JSON object"
    )
