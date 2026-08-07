"""OntoCast CLI entry point: ``serve`` (API) and ``process`` (local batch).

Example:
    # Start the API server
    ontocast serve

    # Process files locally without starting the server
    ontocast process --input-path ./document.pdf --output-dir ./out
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import click
import uvicorn
from langgraph.graph.state import CompiledStateGraph

from ontocast.api.app import create_app
from ontocast.api.parse import (
    parse_document_metadata_param,
    parse_document_type_hint_param,
    parse_max_visits_param,
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
from ontocast.cli.cache import cache as cache_cli
from ontocast.config import Config
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.retrieval_capabilities import validate_ontology_context_mode
from ontocast.onto.tenancy import DEFAULT_PROJECT, DEFAULT_TENANT
from ontocast.stategraph import create_agent_graph
from ontocast.toolbox import ToolBox
from ontocast.util.files import crawl_directories

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., object])


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


@dataclass
class BootstrappedRuntime:
    """Shared ToolBox + config after tenancy init."""

    config: Config
    tools: ToolBox
    tenant: str
    project: str
    ontology_context_mode: OntologyContextMode


def _bootstrap_tools(
    *,
    tenant: str | None,
    project: str | None,
    wipe_vector_store: bool | None,
    flush_on_clean: bool = False,
) -> BootstrappedRuntime:
    """Load config, construct ToolBox, apply tenancy, and initialize tools."""
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
                initialize_vector_store=False,
                fail_on_vector_store_error=vector_mode_enabled,
            )
        )

    if flush_on_clean and config.clean:
        asyncio.run(flush_triple_configured_scope(tools))

    asyncio.run(
        tools.initialize(
            ontology_context_mode=ontology_context_mode_value,
            fail_on_vector_store_error=vector_mode_enabled,
            wipe_vector_store=wipe_vector_store,
        )
    )
    validate_ontology_context_mode(ontology_context_mode_value, tools)
    return BootstrappedRuntime(
        config=config,
        tools=tools,
        tenant=t_res,
        project=p_res,
        ontology_context_mode=ontology_context_mode_value,
    )


def _shared_runtime_options(fn: F) -> F:
    """Click options shared by ``serve`` and ``process``."""
    options = [
        click.option("--head-chunks", type=int, default=None),
        click.option(
            "--max-visits",
            type=int,
            default=None,
            help=(
                "Render/critic retry budget per loop (default from MAX_VISITS / "
                "server config)."
            ),
        ),
        click.option(
            "--tenant",
            type=str,
            default=None,
            help=(
                "Tenant id for dataset/collection names "
                f"(default {DEFAULT_TENANT!r} when omitted; not read from .env)."
            ),
        ),
        click.option(
            "--project",
            type=str,
            default=None,
            help=(
                "Project id for dataset/collection names "
                f"(default {DEFAULT_PROJECT!r} when omitted; not read from .env)."
            ),
        ),
        click.option(
            "--wipe-vector-store/--no-wipe-vector-store",
            default=None,
            help=(
                "Drop the current tenant/project vector partition before ontology "
                "reindex (clean slate). Default follows VECTOR_STORE_WIPE_ON_INIT "
                "(false). Orphan IRIs are still pruned by default via "
                "VECTOR_STORE_PRUNE_ORPHAN_IRIS_ON_INIT."
            ),
        ),
    ]
    for option in reversed(options):
        fn = option(fn)  # type: ignore[assignment]
    return fn


@click.group()
def cli() -> None:
    """OntoCast: start the API server or process local files in batch mode."""


cli.add_command(cache_cli)


@cli.command("serve")
@_shared_runtime_options
def serve(
    head_chunks: int | None,
    max_visits: int | None,
    tenant: str | None,
    project: str | None,
    wipe_vector_store: bool | None,
) -> None:
    """Start the OntoCast API server."""
    runtime = _bootstrap_tools(
        tenant=tenant,
        project=project,
        wipe_vector_store=wipe_vector_store,
        flush_on_clean=False,
    )
    parsed_max_visits = parse_max_visits_param(
        max_visits,
        default=runtime.config.server.max_visits_per_node,
    )
    runtime.config.server.max_visits_per_node = parsed_max_visits
    app = create_app(
        tools=runtime.tools,
        server_config=runtime.config.server,
        head_chunks=head_chunks,
        active_tenant=runtime.tenant,
        active_project=runtime.project,
    )
    bind_host = runtime.config.server.host
    logger.info(
        "Starting Ontocast server on %s:%s", bind_host, runtime.config.server.port
    )
    if bind_host not in {"127.0.0.1", "localhost", "::1"}:
        logger.warning(
            "Binding %s: the server has no authentication and /flush is "
            "destructive. Put it behind a proxy that authenticates.",
            bind_host,
        )
    uvicorn.run(
        app,
        host=bind_host,
        port=runtime.config.server.port,
        log_level="info",
    )


@cli.command("process")
@_shared_runtime_options
@click.option(
    "--input-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="File or directory to process locally (no HTTP server).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help=(
        "Shared directory for facts and ontology Turtle dumps. "
        "When omitted (and no per-kind override), dumps are written next to each input."
    ),
)
@click.option(
    "--facts-output-dir",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Override directory for ``*.facts.ttl`` dumps (defaults to --output-dir).",
)
@click.option(
    "--ontology-output-dir",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help=(
        "Override directory for ``*.ontology.ttl`` dumps (defaults to --output-dir)."
    ),
)
@click.option(
    "--use-unit-pipeline/--no-use-unit-pipeline",
    default=False,
    help=(
        "Run convert_document + run_unit_pipeline instead of the full workflow graph."
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
    "--exclude-sections",
    type=str,
    default=None,
    help=(
        "Comma-separated section labels to drop when chunking (e.g. "
        "acknowledgements,appendix). Unset = the resolved schema's defaults; "
        "pass an empty string to disable exclusion."
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
@click.option(
    "--document-metadata",
    type=str,
    default=None,
    help=(
        "JSON object of caller-asserted document identity metadata "
        '(e.g. \'{"doi":"10.1234/example","title":"…"}\'). '
        "When omitted, the filename is used as dcterms:title "
        "(file:line for JSONL records)."
    ),
)
def process(
    head_chunks: int | None,
    max_visits: int | None,
    tenant: str | None,
    project: str | None,
    wipe_vector_store: bool | None,
    input_path: pathlib.Path,
    output_dir: pathlib.Path | None,
    facts_output_dir: pathlib.Path | None,
    ontology_output_dir: pathlib.Path | None,
    use_unit_pipeline: bool,
    target_sections: str | None,
    exclude_sections: str | None,
    summarize_sections: str | None,
    summary_max_sentences: int,
    document_type_hint: str | None,
    section_schema_id: str | None,
    document_metadata: str | None,
) -> None:
    """Process local files through the extraction pipeline (no HTTP server)."""
    runtime = _bootstrap_tools(
        tenant=tenant,
        project=project,
        wipe_vector_store=wipe_vector_store,
        flush_on_clean=True,
    )
    # The parsers are shared with the HTTP layer and signal bad input by
    # raising; surface that as a click usage error rather than a traceback.
    try:
        parsed_target_sections = (
            parse_sections_list_param(target_sections, param="target-sections")
            if target_sections is not None
            else None
        )
        parsed_exclude_sections = (
            parse_sections_list_param(exclude_sections, param="exclude-sections")
            if exclude_sections is not None
            else None
        )
        parsed_summarize_sections = (
            parse_sections_list_param(summarize_sections, param="summarize-sections")
            if summarize_sections is not None
            else None
        )
        parsed_summary_max_sentences = parse_summary_max_sentences_param(
            summary_max_sentences,
            default=5,
        )
        parsed_document_type_hint = parse_document_type_hint_param(document_type_hint)
        parsed_section_schema_id = parse_section_schema_id_param(section_schema_id)
        parsed_max_visits = parse_max_visits_param(
            max_visits,
            default=runtime.config.server.max_visits_per_node,
        )
        parsed_document_metadata = parse_document_metadata_param(document_metadata)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc
    runtime.config.server.max_visits_per_node = parsed_max_visits

    workflow: CompiledStateGraph = create_agent_graph(runtime.tools)
    input_path = input_path.expanduser()
    out_dir = output_dir.expanduser() if output_dir is not None else None
    facts_dir = facts_output_dir.expanduser() if facts_output_dir is not None else None
    ontology_dir = (
        ontology_output_dir.expanduser() if ontology_output_dir is not None else None
    )
    files = sorted(
        crawl_directories(
            input_path,
            suffixes=get_supported_input_extensions(runtime.tools),
        )
    )
    failed_files = asyncio.run(
        process_files_input(
            files,
            config=runtime.config,
            head_chunks=head_chunks,
            use_unit_pipeline=use_unit_pipeline,
            tools=runtime.tools,
            workflow=workflow,
            ontology_context_mode_value=runtime.ontology_context_mode,
            tenant=runtime.tenant,
            project=runtime.project,
            target_sections=parsed_target_sections,
            exclude_sections=parsed_exclude_sections,
            summarize_sections=parsed_summarize_sections,
            summary_max_sentences=parsed_summary_max_sentences,
            document_type_hint=parsed_document_type_hint,
            section_schema_id=parsed_section_schema_id,
            max_visits=parsed_max_visits,
            document_metadata=parsed_document_metadata,
            output_dir=out_dir,
            facts_output_dir=facts_dir,
            ontology_output_dir=ontology_dir,
        )
    )
    if failed_files:
        # Exit non-zero so a scripted pipeline can tell a partial or total
        # failure from a clean run.
        raise click.ClickException(
            f"{len(failed_files)} of {len(files)} input file(s) failed: "
            + ", ".join(str(path) for path in failed_files[:5])
            + (" ..." if len(failed_files) > 5 else "")
        )


# Backward-compatible alias for tests / direct imports of the old entry name.
run = cli


if __name__ == "__main__":
    cli()
