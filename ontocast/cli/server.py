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
from ontocast.cli.inspect_sections import main as inspect_sections_cli
from ontocast.config import Config
from ontocast.onto.enum import OntologyContextMode, RenderMode
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


def _resolve_seed_directory(
    value: str | pathlib.Path | None,
    *,
    source: str,
    kind: str,
    missing_is_fatal: bool,
) -> pathlib.Path | None:
    """Expand a configured seed directory and report what it resolved to.

    A directory the operator named is an assertion, and the three ways it can
    fail are not the same fault:

    * **Not named at all** -- no assertion, nothing to check. A run with no seed
      directory is legitimate: ``serve`` is filled through ``POST /ontologies``,
      and an ontology-rendering run creates its own vocabulary.
    * **Named and empty** -- plausibly deliberate, so a warning naming the
      resolved path, which is enough to spot a directory that is not the one
      that was meant.
    * **Named and absent** -- the assertion is false. Nothing downstream can
      tell this from "deliberately none", so the catalog silently comes out
      empty and every symptom appears several minutes later in retrieval. For a
      batch run, which has no later chance to be given ontologies, this is
      fatal here rather than mysterious there.

    Args:
        value: Configured directory, from the CLI flag or the environment.
        source: What named it, for the error message (flag or environment var).
        kind: Human name of what is seeded, e.g. ``"ontology"``.
        missing_is_fatal: Raise rather than warn when the path does not
            resolve. ``process`` sets this; ``serve`` does not, because a
            long-lived server may have the directory appear under it later.

    Returns:
        The expanded directory, or ``None`` when nothing usable was named.

    Raises:
        click.UsageError: The path was named, does not resolve, and this entry
            point cannot continue without it.
    """
    if value is None:
        logger.info("No %s seed directory configured (%s)", kind, source)
        return None
    directory = pathlib.Path(value).expanduser()
    if not directory.is_dir():
        message = (
            f"{source} points at {directory.absolute()}, which is not a "
            f"directory. No {kind} files can be seeded from it."
        )
        if missing_is_fatal:
            raise click.UsageError(message)
        logger.warning("%s", message)
        return directory
    logger.info("Using %s seed directory %s (%s)", kind, directory.absolute(), source)
    return directory


def _prepare_path_config(
    config: Config,
    *,
    ontology_dir: str | None = None,
    shapes_dir: str | None = None,
    missing_is_fatal: bool = False,
) -> None:
    """Apply seed-directory overrides and expand the configured directories.

    Called before ``ToolBox(config)``: the toolbox reads both fields lazily in
    ``initialize``, so an override applied any later would be ignored.

    ``ontology_dir`` / ``shapes_dir`` carry three distinguishable states.
    ``None`` means the flag was omitted and the environment stands. A path
    overrides it. The empty string clears it -- which is how a run declares "no
    seed ontologies, infer them" against an environment that sets one, and is
    why these are plain strings rather than ``click.Path``: ``pathlib.Path("")``
    is ``.``, so an empty flag would silently mean the working directory.
    """
    paths = config.tool_config.path_config
    facts_validation = config.tool_config.facts_validation

    if ontology_dir is not None:
        paths.ontology_directory = pathlib.Path(ontology_dir) if ontology_dir else None
        ontology_source = "--ontology-dir"
    else:
        ontology_source = "ONTOCAST_ONTOLOGY_DIRECTORY"

    if shapes_dir is not None:
        facts_validation.shapes_dir = shapes_dir or None
        shapes_source = "--shapes-dir"
    else:
        shapes_source = "FACTS_SHAPES_DIR"

    paths.ontology_directory = _resolve_seed_directory(
        paths.ontology_directory,
        source=ontology_source,
        kind="ontology",
        missing_is_fatal=missing_is_fatal,
    )
    resolved_shapes = _resolve_seed_directory(
        facts_validation.shapes_dir,
        source=shapes_source,
        kind="SHACL shapes",
        missing_is_fatal=missing_is_fatal,
    )
    # ``shapes_dir`` is typed ``str | None``, unlike ``ontology_directory``.
    facts_validation.shapes_dir = (
        str(resolved_shapes) if resolved_shapes is not None else None
    )


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
    ontology_dir: str | None = None,
    shapes_dir: str | None = None,
    flush_on_clean: bool = False,
    batch: bool = False,
) -> BootstrappedRuntime:
    """Load config, construct ToolBox, apply tenancy, and initialize tools.

    Args:
        ontology_dir: ``--ontology-dir``. ``None`` leaves
            ``ONTOCAST_ONTOLOGY_DIRECTORY`` in force; ``""`` clears it.
        shapes_dir: ``--shapes-dir``, same three states over ``FACTS_SHAPES_DIR``.
        batch: This is ``process``, not ``serve``. A batch run has no later
            chance to be given ontologies, so a named-but-absent seed directory
            is fatal, and an empty catalog is fatal for the one render mode that
            cannot create vocabulary of its own. A server is filled through
            ``POST /ontologies``, so neither is.
    """
    config = Config()
    config.validate_llm_config()
    _configure_logging(config)
    _prepare_path_config(
        config,
        ontology_dir=ontology_dir,
        shapes_dir=shapes_dir,
        missing_is_fatal=batch,
    )

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
            require_populated_catalog=(
                batch and config.server.render_mode == RenderMode.FACTS
            ),
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
            "--ontology-dir",
            type=str,
            default=None,
            help=(
                "Input catalog: directory of seed ontology *.ttl files, "
                "overriding ONTOCAST_ONTOLOGY_DIRECTORY. Pass an empty string "
                "to declare no seed ontologies, which an ontology-rendering "
                "run answers by creating them from the corpus."
            ),
        ),
        click.option(
            "--shapes-dir",
            type=str,
            default=None,
            help=(
                "Seed directory of SHACL shape files, overriding "
                "FACTS_SHAPES_DIR. Pass an empty string to declare no seed "
                "shapes."
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
cli.add_command(inspect_sections_cli)


@cli.command("serve")
@_shared_runtime_options
def serve(
    head_chunks: int | None,
    max_visits: int | None,
    tenant: str | None,
    project: str | None,
    ontology_dir: str | None,
    shapes_dir: str | None,
    wipe_vector_store: bool | None,
) -> None:
    """Start the OntoCast API server."""
    runtime = _bootstrap_tools(
        tenant=tenant,
        project=project,
        wipe_vector_store=wipe_vector_store,
        ontology_dir=ontology_dir,
        shapes_dir=shapes_dir,
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
    "--facts-user-instruction",
    type=str,
    default="",
    help=(
        "Deployment-specific guidance appended to the facts render and "
        "critic prompts (the same per-request slot the HTTP API exposes). "
        "The library prompt stays domain-neutral; domain refinements belong "
        "here or in the shapes."
    ),
)
@click.option(
    "--summarize-sections",
    type=str,
    default=None,
    help=(
        "Comma-separated section labels to summarize before extraction, or '*' / empty "
        "for all chunks. When set, summarization runs inside chunk preparation."
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
    "--keep-provenance/--strip-provenance",
    "keep_provenance",
    default=False,
    show_default=True,
    help=(
        "Keep chunk-level provenance in the dumped facts Turtle. Provenance is "
        "what lets a statement be traced back to its source span and "
        "re-verified against the document."
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
    ontology_dir: str | None,
    shapes_dir: str | None,
    wipe_vector_store: bool | None,
    input_path: pathlib.Path,
    output_dir: pathlib.Path | None,
    facts_output_dir: pathlib.Path | None,
    ontology_output_dir: pathlib.Path | None,
    use_unit_pipeline: bool,
    target_sections: str | None,
    exclude_sections: str | None,
    facts_user_instruction: str,
    summarize_sections: str | None,
    summary_max_sentences: int,
    document_type_hint: str | None,
    section_schema_id: str | None,
    keep_provenance: bool,
    document_metadata: str | None,
) -> None:
    """Process local files through the extraction pipeline (no HTTP server)."""
    runtime = _bootstrap_tools(
        tenant=tenant,
        project=project,
        wipe_vector_store=wipe_vector_store,
        ontology_dir=ontology_dir,
        shapes_dir=shapes_dir,
        flush_on_clean=True,
        batch=True,
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
    ontology_out_dir = (
        ontology_output_dir.expanduser() if ontology_output_dir is not None else None
    )
    supported_suffixes = get_supported_input_extensions(runtime.tools)
    try:
        files = sorted(crawl_directories(input_path, suffixes=supported_suffixes))
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--input-path") from exc
    if not files:
        # An empty crawl used to exit 0 with no output, which reads as success.
        raise click.ClickException(
            f"No supported input files under {input_path} "
            f"(looking for {', '.join(supported_suffixes)})."
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
            facts_user_instruction=facts_user_instruction,
            output_dir=out_dir,
            facts_output_dir=facts_dir,
            ontology_output_dir=ontology_out_dir,
            strip_provenance=not keep_provenance,
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
