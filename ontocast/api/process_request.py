"""Shared ``/process`` and ``/process_unit`` request body parsing."""

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from ontocast.api.parse import (
    parse_document_metadata_param,
    parse_document_type_hint_param,
    parse_llm_graph_format_param,
    parse_max_visits_param,
    parse_ontology_context_mode_param,
    parse_render_mode_param,
    parse_section_schema_id_param,
    parse_sections_list_param,
    parse_strip_provenance_param,
    parse_summary_max_sentences_param,
    resolve_ontology_context_mode,
)
from ontocast.api.responses import missing_fixed_catalog_ontology_id_response
from ontocast.api.schemas import StatusErrorBody
from ontocast.config import ServerConfig
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.state import AgentState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ParamSpec:
    """How one request parameter is read, on every transport alike.

    ``parse`` receives the raw value and the value accumulated so far (which is
    the default for parsers that take one). ``skip_blank`` reproduces the
    multipart guards that ignored empty form fields for scalars, so an unset
    text input does not read as "explicitly empty".
    """

    parse: Callable[[Any, Any], Any]
    skip_blank: bool = False


def _identity_str(raw: Any, _current: Any) -> str:
    return str(raw)


#: The single source of truth for request parameters. Query string, JSON body
#: and multipart form are all read through this table, in that precedence
#: order, so a parameter cannot be supported on one transport and silently
#: ignored on another -- which is what happened to ``render_mode``,
#: ``ontology_context_mode`` and the three instruction fields, all of which
#: were query-only despite being documented as request parameters.
_PARAM_SPECS: dict[str, _ParamSpec] = {
    "render_mode": _ParamSpec(_identity_str, skip_blank=True),
    "llm_graph_format": _ParamSpec(_identity_str, skip_blank=True),
    "ontology_context_mode": _ParamSpec(_identity_str, skip_blank=True),
    "ontology_user_instruction": _ParamSpec(_identity_str, skip_blank=True),
    "ontology_selection_user_instruction": _ParamSpec(_identity_str, skip_blank=True),
    "facts_user_instruction": _ParamSpec(_identity_str, skip_blank=True),
    "ontology_context_fixed_ontology_id": _ParamSpec(
        lambda raw, _cur: str(raw).strip(), skip_blank=True
    ),
    "strip_provenance": _ParamSpec(
        lambda raw, _cur: parse_strip_provenance_param(str(raw)), skip_blank=True
    ),
    "max_visits": _ParamSpec(parse_max_visits_param, skip_blank=True),
    "summary_max_sentences": _ParamSpec(
        parse_summary_max_sentences_param, skip_blank=True
    ),
    "target_sections": _ParamSpec(
        lambda raw, _cur: parse_sections_list_param(raw, "target_sections")
    ),
    "exclude_sections": _ParamSpec(
        lambda raw, _cur: parse_sections_list_param(raw, "exclude_sections")
    ),
    "summarize_sections": _ParamSpec(
        lambda raw, _cur: parse_sections_list_param(raw, "summarize_sections")
    ),
    "document_type_hint": _ParamSpec(
        lambda raw, _cur: parse_document_type_hint_param(str(raw))
    ),
    "section_schema_id": _ParamSpec(
        lambda raw, _cur: parse_section_schema_id_param(str(raw))
    ),
    "document_metadata": _ParamSpec(
        lambda raw, _cur: parse_document_metadata_param(raw)
    ),
}


def _apply_source(values: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Fold one transport's raw values into ``values`` through the spec table."""
    for name, raw in source.items():
        spec = _PARAM_SPECS.get(name)
        if spec is None or raw is None:
            continue
        if spec.skip_blank and isinstance(raw, str) and not raw.strip():
            continue
        values[name] = spec.parse(raw, values[name])


@dataclass(frozen=True)
class ParsedProcessRequest:
    """Fields shared by ``/process`` and ``/process_unit`` after reading the body."""

    files_dict: dict[str, bytes]
    max_visits: int
    strip_provenance: bool
    ontology_user_instruction: str
    ontology_selection_user_instruction: str
    facts_user_instruction: str
    ontology_context_fixed_ontology_id: str
    render_mode: str | None
    llm_graph_format: str | None
    ontology_context_mode_value: OntologyContextMode
    target_sections: list[str] | None
    exclude_sections: list[str] | None
    summarize_sections: list[str] | None
    summary_max_sentences: int
    document_type_hint: str | None
    section_schema_id: str | None
    document_metadata: dict[str, object]


async def load_parsed_process_request(
    request: Request,
    server_config: ServerConfig,
    *,
    log_label: str = "process",
) -> ParsedProcessRequest | JSONResponse:
    """Read request parameters from query string, JSON body or multipart form.

    All three transports are read through :data:`_PARAM_SPECS`, in precedence
    order (query, then body/form), so every parameter is honoured on every
    transport. The body branches differ only in how raw values are obtained --
    a decoded JSON object, or the form's multi-items -- never in which
    parameters they support.
    """
    content_type = request.headers.get("content-type") or ""
    logger.debug("%s Content-Type: %s", log_label, content_type)

    values: dict[str, Any] = {
        "render_mode": None,
        "llm_graph_format": None,
        "ontology_context_mode": None,
        "ontology_user_instruction": "",
        "ontology_selection_user_instruction": "",
        "facts_user_instruction": "",
        "ontology_context_fixed_ontology_id": "",
        "strip_provenance": False,
        "max_visits": server_config.max_visits_per_node,
        "summary_max_sentences": 5,
        "target_sections": None,
        "exclude_sections": None,
        "summarize_sections": None,
        "document_type_hint": None,
        "section_schema_id": None,
        "document_metadata": {},
    }

    _apply_source(values, dict(request.query_params))

    if content_type.startswith("application/json"):
        bytes_data = await request.body()
        logger.debug("%s JSON body length: %s", log_label, len(bytes_data))
        files_dict = {"input.json": bytes_data}
        try:
            parsed_obj = json.loads(bytes_data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug(
                "%s JSON body could not be decoded for parameter extraction",
                log_label,
            )
        else:
            if isinstance(parsed_obj, dict):
                _apply_source(values, parsed_obj)
    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        files_dict = {}
        form_values: dict[str, Any] = {}
        for key, value in form.multi_items():
            if isinstance(value, StarletteUploadFile):
                files_dict[key] = await value.read()
            else:
                form_values[key] = str(value)
        _apply_source(values, form_values)
        if not files_dict:
            return JSONResponse(
                status_code=400,
                content=StatusErrorBody(
                    error="No file provided",
                    error_type="ValidationError",
                ).model_dump(),
            )
    else:
        return JSONResponse(
            status_code=400,
            content=StatusErrorBody(
                error=f"Unsupported content type: {content_type}",
                error_type="ValidationError",
            ).model_dump(),
        )

    ontology_context_mode_value: OntologyContextMode = (
        parse_ontology_context_mode_param(
            values["ontology_context_mode"],
            server_config.ontology_context_mode,
        )
    )

    fixed_ontology_id = values["ontology_context_fixed_ontology_id"]
    ontology_context_mode_value = resolve_ontology_context_mode(
        ontology_context_mode_value,
        fixed_ontology_id,
    )
    if (
        ontology_context_mode_value == OntologyContextMode.FIXED_SINGLE_ONTOLOGY
        and not fixed_ontology_id
    ):
        return missing_fixed_catalog_ontology_id_response()

    return ParsedProcessRequest(
        files_dict=files_dict,
        max_visits=values["max_visits"],
        strip_provenance=values["strip_provenance"],
        ontology_user_instruction=values["ontology_user_instruction"],
        ontology_selection_user_instruction=values[
            "ontology_selection_user_instruction"
        ],
        facts_user_instruction=values["facts_user_instruction"],
        ontology_context_fixed_ontology_id=fixed_ontology_id,
        render_mode=values["render_mode"],
        llm_graph_format=values["llm_graph_format"],
        ontology_context_mode_value=ontology_context_mode_value,
        target_sections=values["target_sections"],
        exclude_sections=values["exclude_sections"],
        summarize_sections=values["summarize_sections"],
        summary_max_sentences=values["summary_max_sentences"],
        document_type_hint=values["document_type_hint"],
        section_schema_id=values["section_schema_id"],
        document_metadata=values["document_metadata"],
    )


def build_agent_state_from_parsed(
    parsed: ParsedProcessRequest,
    *,
    server_config: ServerConfig,
    resolved_tenant: str,
    resolved_project: str,
    max_chunks: int | None,
) -> AgentState:
    """Construct ``AgentState`` after tenancy resolution and enum parsing."""
    render_mode_value = parse_render_mode_param(
        parsed.render_mode,
        server_config.render_mode,
    )
    llm_graph_format_value = parse_llm_graph_format_param(
        parsed.llm_graph_format,
        server_config.llm_graph_format,
    )
    return AgentState(
        raw_input=parsed.files_dict,
        max_visits=parsed.max_visits,
        max_chunks=max_chunks,
        render_mode=render_mode_value,
        llm_graph_format=llm_graph_format_value,
        ontology_context_mode=parsed.ontology_context_mode_value,
        tenant=resolved_tenant,
        project=resolved_project,
        ontology_user_instruction=parsed.ontology_user_instruction,
        ontology_selection_user_instruction=parsed.ontology_selection_user_instruction,
        facts_user_instruction=parsed.facts_user_instruction,
        ontology_context_fixed_ontology_id=parsed.ontology_context_fixed_ontology_id,
        target_sections=parsed.target_sections,
        exclude_sections=parsed.exclude_sections,
        summarize_sections=parsed.summarize_sections,
        summary_max_sentences=parsed.summary_max_sentences,
        document_type_hint=parsed.document_type_hint,
        section_schema_id=parsed.section_schema_id,
        document_metadata=dict(parsed.document_metadata),
    )
