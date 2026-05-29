"""Shared HTTP query/body parsing for CLI server routes."""

import json
import logging

from ontocast.onto.enum import LLMGraphFormat, OntologyContextMode, RenderMode

logger = logging.getLogger(__name__)


def parse_render_mode_param(value, default: RenderMode) -> RenderMode:
    if value is None:
        return default
    if isinstance(value, RenderMode):
        return value
    if isinstance(value, str):
        normalized = value.lower().strip()
        try:
            return RenderMode(normalized)
        except ValueError:
            logger.warning(
                "Invalid render_mode '%s', using default '%s'",
                value,
                default.value,
            )
    return default


def parse_llm_graph_format_param(
    value: str | LLMGraphFormat | None,
    default: LLMGraphFormat,
) -> LLMGraphFormat:
    """Parse optional ``llm_graph_format`` override from request params."""
    if value is None:
        return default
    if isinstance(value, LLMGraphFormat):
        return value
    if isinstance(value, str):
        normalized = value.lower().strip()
        try:
            return LLMGraphFormat(normalized)
        except ValueError:
            logger.warning(
                "Invalid llm_graph_format '%s', using default '%s'",
                value,
                default.value,
            )
    return default


def parse_ontology_context_mode_param(
    value: str | OntologyContextMode | None,
    default: OntologyContextMode,
) -> OntologyContextMode:
    if value is None:
        return default
    if isinstance(value, OntologyContextMode):
        return value
    if isinstance(value, str):
        normalized = value.lower().strip()
        try:
            return OntologyContextMode(normalized)
        except ValueError:
            logger.warning(
                "Invalid ontology_context_mode '%s', using default '%s'",
                value,
                default.value,
            )
    return default


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


def parse_sections_list_param(value: str | list[str] | None) -> list[str]:
    """Parse a section list from comma-separated text or JSON array."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("section list JSON must be an array")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_summary_max_sentences_param(value: str | int | None, default: int) -> int:
    """Parse optional summary_max_sentences (positive integer)."""
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("summary_max_sentences must be a positive integer") from exc
    if parsed < 1:
        raise ValueError("summary_max_sentences must be a positive integer")
    return parsed


def parse_max_visits_param(value: str | int | None, default: int) -> int:
    """Parse optional ``max_visits`` override from query/form/json metadata."""
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("max_visits must be an integer >= 1") from exc
    if parsed < 1:
        raise ValueError("max_visits must be an integer >= 1")
    return parsed
