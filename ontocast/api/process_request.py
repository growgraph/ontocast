"""Shared ``/process`` and ``/process_unit`` request body parsing."""

import json
import logging
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from ontocast.api.parse import (
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
    summarize_sections: list[str] | None
    summary_max_sentences: int
    document_type_hint: str | None
    section_schema_id: str | None


async def load_parsed_process_request(
    request: Request,
    server_config: ServerConfig,
    *,
    log_label: str = "process",
) -> ParsedProcessRequest | JSONResponse:
    """Read JSON or multipart body plus query defaults (same semantics as legacy handlers)."""
    content_type = request.headers.get("content-type") or ""
    logger.debug("%s Content-Type: %s", log_label, content_type)

    render_mode = request.query_params.get("render_mode", None)
    llm_graph_format = request.query_params.get("llm_graph_format", None)
    ontology_context_mode = request.query_params.get("ontology_context_mode", None)
    ontology_user_instruction = request.query_params.get(
        "ontology_user_instruction", ""
    )
    ontology_selection_user_instruction = request.query_params.get(
        "ontology_selection_user_instruction", ""
    )
    facts_user_instruction = request.query_params.get("facts_user_instruction", "")
    ontology_context_fixed_ontology_id = request.query_params.get(
        "ontology_context_fixed_ontology_id", ""
    ).strip()
    strip_provenance = parse_strip_provenance_param(
        request.query_params.get("strip_provenance")
    )
    max_visits = parse_max_visits_param(
        request.query_params.get("max_visits"),
        server_config.max_visits_per_node,
    )
    ontology_context_mode_value: OntologyContextMode = (
        parse_ontology_context_mode_param(
            ontology_context_mode,
            server_config.ontology_context_mode,
        )
    )

    target_sections: list[str] | None = None
    if "target_sections" in request.query_params:
        target_sections = parse_sections_list_param(
            request.query_params.get("target_sections")
        )

    summarize_sections: list[str] | None = None
    if "summarize_sections" in request.query_params:
        summarize_sections = parse_sections_list_param(
            request.query_params.get("summarize_sections")
        )

    summary_max_sentences = parse_summary_max_sentences_param(
        request.query_params.get("summary_max_sentences"),
        default=5,
    )

    document_type_hint = parse_document_type_hint_param(
        request.query_params.get("document_type_hint")
    )
    section_schema_id = parse_section_schema_id_param(
        request.query_params.get("section_schema_id")
    )

    if content_type.startswith("application/json"):
        bytes_data = await request.body()
        logger.debug("%s JSON body length: %s", log_label, len(bytes_data))
        files_dict = {"input.json": bytes_data}
        try:
            parsed_obj = json.loads(bytes_data.decode("utf-8"))
            if isinstance(parsed_obj, dict):
                oid_raw = parsed_obj.get("ontology_context_fixed_ontology_id", "")
                if isinstance(oid_raw, str) and oid_raw.strip():
                    ontology_context_fixed_ontology_id = oid_raw.strip()
                max_visits = parse_max_visits_param(
                    parsed_obj.get("max_visits"),
                    max_visits,
                )
                body_format = parsed_obj.get("llm_graph_format")
                if body_format is not None:
                    llm_graph_format = body_format
                if "target_sections" in parsed_obj:
                    target_sections = parse_sections_list_param(
                        parsed_obj.get("target_sections")
                    )
                if "summarize_sections" in parsed_obj:
                    summarize_sections = parse_sections_list_param(
                        parsed_obj.get("summarize_sections")
                    )
                if "summary_max_sentences" in parsed_obj:
                    summary_max_sentences = parse_summary_max_sentences_param(
                        parsed_obj.get("summary_max_sentences"),
                        summary_max_sentences,
                    )
                if "document_type_hint" in parsed_obj:
                    raw_hint = parsed_obj.get("document_type_hint")
                    if raw_hint is not None:
                        document_type_hint = parse_document_type_hint_param(
                            str(raw_hint)
                        )
                if "section_schema_id" in parsed_obj:
                    raw_schema = parsed_obj.get("section_schema_id")
                    if raw_schema is not None:
                        section_schema_id = parse_section_schema_id_param(
                            str(raw_schema)
                        )
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug(
                "%s JSON body could not be decoded for ontology id preview",
                log_label,
            )
    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        files_dict = {}
        for key, value in form.multi_items():
            if isinstance(value, StarletteUploadFile):
                files_dict[key] = await value.read()
            elif key == "ontology_user_instruction" and value:
                ontology_user_instruction = str(value)
            elif key == "ontology_selection_user_instruction" and value:
                ontology_selection_user_instruction = str(value)
            elif key == "facts_user_instruction" and value:
                facts_user_instruction = str(value)
            elif key == "ontology_context_fixed_ontology_id" and value:
                ontology_context_fixed_ontology_id = str(value).strip()
            elif key == "strip_provenance" and value:
                strip_provenance = parse_strip_provenance_param(str(value))
            elif key == "max_visits" and value:
                max_visits = parse_max_visits_param(str(value), max_visits)
            elif key == "llm_graph_format" and value:
                llm_graph_format = str(value)
            elif key == "target_sections" and value is not None:
                target_sections = parse_sections_list_param(str(value))
            elif key == "summarize_sections" and value is not None:
                summarize_sections = parse_sections_list_param(str(value))
            elif key == "summary_max_sentences" and value:
                summary_max_sentences = parse_summary_max_sentences_param(
                    str(value),
                    summary_max_sentences,
                )
            elif key == "document_type_hint" and value is not None:
                document_type_hint = parse_document_type_hint_param(str(value))
            elif key == "section_schema_id" and value is not None:
                section_schema_id = parse_section_schema_id_param(str(value))
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

    ontology_context_mode_value = resolve_ontology_context_mode(
        ontology_context_mode_value,
        ontology_context_fixed_ontology_id,
    )
    if (
        ontology_context_mode_value == OntologyContextMode.FIXED_SINGLE_ONTOLOGY
        and not ontology_context_fixed_ontology_id
    ):
        return missing_fixed_catalog_ontology_id_response()

    return ParsedProcessRequest(
        files_dict=files_dict,
        max_visits=max_visits,
        strip_provenance=strip_provenance,
        ontology_user_instruction=ontology_user_instruction,
        ontology_selection_user_instruction=ontology_selection_user_instruction,
        facts_user_instruction=facts_user_instruction,
        ontology_context_fixed_ontology_id=ontology_context_fixed_ontology_id,
        render_mode=render_mode,
        llm_graph_format=llm_graph_format,
        ontology_context_mode_value=ontology_context_mode_value,
        target_sections=target_sections,
        summarize_sections=summarize_sections,
        summary_max_sentences=summary_max_sentences,
        document_type_hint=document_type_hint,
        section_schema_id=section_schema_id,
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
        ontology_max_triples=server_config.ontology_max_triples,
        tenant=resolved_tenant,
        project=resolved_project,
        ontology_user_instruction=parsed.ontology_user_instruction,
        ontology_selection_user_instruction=parsed.ontology_selection_user_instruction,
        facts_user_instruction=parsed.facts_user_instruction,
        ontology_context_fixed_ontology_id=parsed.ontology_context_fixed_ontology_id,
        target_sections=parsed.target_sections,
        summarize_sections=parsed.summarize_sections,
        summary_max_sentences=parsed.summary_max_sentences,
        document_type_hint=parsed.document_type_hint,
        section_schema_id=parsed.section_schema_id,
    )
