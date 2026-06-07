"""OntoCast CLI entry point for the API server and local batch processing.

Example:
    # Start the API server
    ontocast

    # Process files locally without starting the server
    ontocast --input-path ./document.pdf
"""

import asyncio
import logging
import pathlib

import click
import uvicorn
from langgraph.graph.state import CompiledStateGraph

from ontocast.api.app import create_app
from ontocast.api.parse import (
    parse_document_type_hint_param,
    parse_section_schema_id_param,
    parse_sections_list_param,
    parse_summary_max_sentences_param,
)
from ontocast.api.process_helpers import (
    flush_triple_configured_scope,
    get_supported_input_extensions,
    process_files_input,
)
from ontocast.api.tenancy_resolution import (
    resolve_tenant_project,
    stores_use_tenancy_partitions,
)
from ontocast.config import Config
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.retrieval_capabilities import validate_ontology_context_mode
from ontocast.onto.tenancy import DEFAULT_PROJECT, DEFAULT_TENANT
from ontocast.stategraph import create_agent_graph
from ontocast.toolbox import ToolBox
from ontocast.util.files import crawl_directories

logger = logging.getLogger(__name__)


def get_next_level(level: int) -> int:
    levels = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]

    try:
        idx = levels.index(level)
        return levels[min(idx + 1, len(levels) - 1)]
    except ValueError:
        return level


def _configure_logging(config: Config) -> None:
    """Configure root and module loggers from config."""
    if config.logging_level is None:
        return

    try:
        level = getattr(logging, config.logging_level.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f"Invalid log level: {config.logging_level}")
        global_level = get_next_level(level)
        logging.basicConfig(level=global_level, handlers=[logging.StreamHandler()])
        logging.getLogger("ontocast").setLevel(level)
    except Exception as e:
        logger.error("could set logging level correctly %s", e)


def _prepare_path_config(config: Config) -> None:
    """Expand configured directories and ensure working directory exists."""
    if config.tool_config.path_config.working_directory is not None:
        config.tool_config.path_config.working_directory = pathlib.Path(
            config.tool_config.path_config.working_directory
        ).expanduser()
        config.tool_config.path_config.working_directory.mkdir(
            parents=True, exist_ok=True
        )
    else:
        raise ValueError(
            "Working directory must be provided via CLI argument or "
            "WORKING_DIRECTORY config"
        )

    if config.tool_config.path_config.ontology_directory is not None:
        config.tool_config.path_config.ontology_directory = pathlib.Path(
            config.tool_config.path_config.ontology_directory
        ).expanduser()


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), default=None)
@click.option("--head-chunks", type=int, default=None)
@click.option(
    "--use-unit-pipeline/--no-use-unit-pipeline",
    default=False,
    help=(
        "When processing files with --input-path, run convert_document + "
        "run_unit_pipeline instead of the full workflow graph."
    ),
)
@click.option(
    "--tenant",
    type=str,
    default=None,
    help=(
        "Tenant id for dataset/collection names "
        f"(default {DEFAULT_TENANT!r} when omitted; not read from .env)."
    ),
)
@click.option(
    "--project",
    type=str,
    default=None,
    help=(
        "Project id for dataset/collection names "
        f"(default {DEFAULT_PROJECT!r} when omitted; not read from .env)."
    ),
)
@click.option(
    "--target-sections",
    type=str,
    default=None,
    help=(
        "Comma-separated section labels to keep when chunking (e.g. results,methods). "
        "Enables section tagging in the workflow graph."
    ),
)
@click.option(
    "--summarize-sections",
    type=str,
    default=None,
    help=(
        "Comma-separated section labels to summarize before extraction, or '*' / empty "
        "for all chunks. When set, runs the summarize_chunks graph node."
    ),
)
@click.option(
    "--summary-max-sentences",
    type=int,
    default=5,
    show_default=True,
    help="Max sentences per chunk summary when --summarize-sections is set.",
)
@click.option(
    "--document-type-hint",
    type=str,
    default=None,
    help=(
        "Optional free-text hint about the source material (e.g. 'SEC 10-K', "
        "'journal article') to resolve section label schema and LLM tagging."
    ),
)
@click.option(
    "--section-schema-id",
    type=str,
    default=None,
    help=(
        "Section label schema id (academic, financial, legal, clinical, manual, "
        "fiction, general). Overrides --document-type-hint when set."
    ),
)
def run(
    input_path: pathlib.Path | None,
    head_chunks: int | None,
    use_unit_pipeline: bool,
    tenant: str | None,
    project: str | None,
    target_sections: str | None,
    summarize_sections: str | None,
    summary_max_sentences: int,
    document_type_hint: str | None,
    section_schema_id: str | None,
):
    """Start the OntoCast API server or process local files in batch mode."""
    config = Config()
    config.validate_llm_config()
    _configure_logging(config)
    _prepare_path_config(config)

    if (
        config.server.ontology_context_mode == OntologyContextMode.FIXED_SINGLE_ONTOLOGY
        and not config.server.ontology_context_fixed_ontology_id.strip()
    ):
        raise ValueError(
            "ontology_context_mode=fixed_single_ontology requires "
            "ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID in the environment (or server "
            "config field ontology_context_fixed_ontology_id)."
        )

    tools: ToolBox = ToolBox(config)
    t_res, p_res = resolve_tenant_project(tenant, project)
    ontology_context_mode_value = config.server.ontology_context_mode
    vector_mode_enabled = (
        ontology_context_mode_value
        == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
    )
    if stores_use_tenancy_partitions(tools):
        asyncio.run(
            tools.update_tenancy_with_vector_mode(
                t_res,
                p_res,
                initialize_vector_store=vector_mode_enabled,
                fail_on_vector_store_error=vector_mode_enabled,
            )
        )

    if input_path is not None and config.clean:
        asyncio.run(flush_triple_configured_scope(tools))

    asyncio.run(
        tools.initialize(
            ontology_context_mode=ontology_context_mode_value,
            fail_on_vector_store_error=vector_mode_enabled,
        )
    )
    validate_ontology_context_mode(ontology_context_mode_value, tools)

    parsed_target_sections = (
        parse_sections_list_param(target_sections)
        if target_sections is not None
        else None
    )
    parsed_summarize_sections = (
        parse_sections_list_param(summarize_sections)
        if summarize_sections is not None
        else None
    )
    parsed_summary_max_sentences = parse_summary_max_sentences_param(
        summary_max_sentences,
        default=5,
    )
    parsed_document_type_hint = parse_document_type_hint_param(document_type_hint)
    parsed_section_schema_id = parse_section_schema_id_param(section_schema_id)

    workflow: CompiledStateGraph = create_agent_graph(tools)

    if input_path is not None:
        input_path = input_path.expanduser()
        files = sorted(
            crawl_directories(
                input_path,
                suffixes=get_supported_input_extensions(tools),
            )
        )
        asyncio.run(
            process_files_input(
                files,
                config=config,
                head_chunks=head_chunks,
                use_unit_pipeline=use_unit_pipeline,
                tools=tools,
                workflow=workflow,
                ontology_context_mode_value=ontology_context_mode_value,
                tenant=t_res,
                project=p_res,
                target_sections=parsed_target_sections,
                summarize_sections=parsed_summarize_sections,
                summary_max_sentences=parsed_summary_max_sentences,
                document_type_hint=parsed_document_type_hint,
                section_schema_id=parsed_section_schema_id,
            )
        )
    else:
        app = create_app(
            tools=tools,
            server_config=config.server,
            head_chunks=head_chunks,
            active_tenant=t_res,
            active_project=p_res,
        )
        logger.info("Starting Ontocast server on port %s", config.server.port)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=config.server.port,
            log_level="info",
        )


if __name__ == "__main__":
    run()
