"""OntoCast API server implementation.

This module provides a web server implementation for the OntoCast framework
using FastAPI/uvicorn. It exposes REST API endpoints for processing documents and
extracting semantic triples with ontology assistance.

The server supports:
- Health check endpoint (/health)
- Service information endpoint (/info)
- Document processing endpoint (/process)
- Triple store flush endpoint (/flush)
- Multiple input formats (JSON, multipart/form-data)
- Streaming workflow execution
- Comprehensive error handling and logging

The server integrates with the OntoCast workflow graph to process documents
through the complete pipeline: chunking, ontology selection, fact extraction,
and aggregation.

Example:
    # With Fuseki backend (auto-detected from FUSEKI_URI and FUSEKI_AUTH)
    ontocast

    # Process specific file
    ontocast --input-path ./document.pdf

    # Process with chunk limit
    ontocast --head-chunks 5
"""

import asyncio
import logging
import logging.config
import pathlib
from importlib import metadata

import click
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from starlette.datastructures import UploadFile as StarletteUploadFile

from ontocast.agent.convert_document import convert_document
from ontocast.agent.serialize import serialize as serialize_agent_state
from ontocast.api.ontologies import build_ontology_router
from ontocast.api.schemas import (
    FlushOkResponse,
    HealthErrorResponse,
    HealthOkResponse,
    InfoResponse,
    ProcessErrorResponse,
    ProcessOkResponse,
    ProcessResultData,
    ProcessResultMetadata,
    StatusErrorBody,
)
from ontocast.cli.util import crawl_directories
from ontocast.config import Config, ServerConfig
from ontocast.onto.enum import OntologyContextMode, RenderMode, Status
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import (
    OntologyContextConfigError,
    VectorStoreUnavailableError,
    require_vector_retrieval,
)
from ontocast.onto.state import AgentState
from ontocast.onto.tenancy import DEFAULT_PROJECT, DEFAULT_TENANT
from ontocast.stategraph import create_agent_graph
from ontocast.stategraph.helpers import build_ontology_delta_graph
from ontocast.stategraph.unit_pipeline import run_unit_pipeline
from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager
from ontocast.toolbox import ToolBox

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


def validate_ontology_context_mode(
    ontology_context_mode: OntologyContextMode,
    tools: ToolBox,
) -> None:
    if ontology_context_mode == OntologyContextMode.VECTOR_RETRIEVAL:
        require_vector_retrieval(tools)


def _ontology_context_error_response(error: OntologyContextConfigError) -> JSONResponse:
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


def _stores_use_tenancy_partitions(tools: ToolBox) -> bool:
    """True when Fuseki and/or Qdrant should be retargeted for tenant/project."""
    if tools.vector_store is not None:
        return True
    return isinstance(tools.triple_store_manager, FusekiTripleStoreManager)


def _resolve_tenant_project(tenant: str | None, project: str | None) -> tuple[str, str]:
    t = (tenant or DEFAULT_TENANT).strip()
    p = (project or DEFAULT_PROJECT).strip()
    if not t or not p:
        raise ValueError("tenant and project must be non-empty after resolution")
    return t, p


async def _flush_triple_configured_scope(tools: ToolBox) -> None:
    """Match POST /flush without tenant/project: triple store only, current scope."""
    if tools.triple_store_manager is not None:
        await tools.triple_store_manager.clean()


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
        return level  # fallback


def calculate_recursion_limit(
    head_chunks: int | None,
    server_config: ServerConfig,
) -> int:
    """Calculate the recursion limit based on max visits and head chunks.

    Args:
        head_chunks: Optional maximum number of chunks to process

    Returns:
        int: Calculated recursion limit
    """
    if head_chunks is not None:
        # If we know the number of chunks, calculate exact limit
        return max(
            server_config.base_recursion_limit,
            server_config.max_visits_per_node * head_chunks * 10,
        )
    else:
        # If we don't know chunks, use a conservative estimate
        return max(
            server_config.base_recursion_limit,
            server_config.max_visits_per_node * server_config.estimated_chunks * 10,
        )


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


def _build_file_state(
    file_path: pathlib.Path,
    *,
    config: Config,
    head_chunks: int | None,
    ontology_context_mode_value: OntologyContextMode,
    tenant: str | None,
    project: str | None,
) -> AgentState:
    return AgentState(
        files={file_path.as_posix(): file_path.read_bytes()},
        max_visits=config.server.max_visits_per_node,
        max_chunks=head_chunks,
        render_mode=config.server.render_mode,
        ontology_context_mode=ontology_context_mode_value,
        tenant=tenant,
        project=project,
    )


def _select_unit_facts_ontology_graph(onto_result, facts_result) -> RDFGraph:
    """Return ontology graph for facts post-processing in unit pipeline flows.

    Priority:
    1. facts_result.ontology_snapshot.graph (context that actually drove facts render)
    2. onto_result.current_ontology.graph (fallback when facts result is unavailable)
    3. empty graph
    """
    if facts_result is not None:
        return facts_result.ontology_snapshot.graph
    if (
        onto_result is not None
        and not onto_result.current_ontology.is_null()
        and len(onto_result.current_ontology.graph) > 0
    ):
        return onto_result.current_ontology.graph
    return RDFGraph()


async def _persist_unit_pipeline_outputs(
    state: AgentState,
    onto_result,
    facts_result,
    tools: ToolBox,
) -> None:
    """Serialize unit-pipeline outputs using the standard document serializer."""
    if onto_result is not None and not onto_result.current_ontology.is_null():
        state.reduced_ontology_artifacts = [onto_result.current_ontology]
    if facts_result is not None:
        ontology_graph = _select_unit_facts_ontology_graph(onto_result, facts_result)
        state.aggregated_facts = tools.aggregator.postprocess_facts_units(
            units=[facts_result.content_unit],
            ontology_graph=ontology_graph,
        )
    # Run synchronous serialization off the active event loop.
    await asyncio.to_thread(serialize_agent_state, state, tools)


async def _process_files_input(
    files: list[pathlib.Path],
    *,
    config: Config,
    head_chunks: int | None,
    use_unit_pipeline: bool,
    tools: ToolBox,
    workflow: CompiledStateGraph,
    ontology_context_mode_value: OntologyContextMode,
    tenant: str | None,
    project: str | None,
) -> None:
    recursion_limit = calculate_recursion_limit(head_chunks, config.server)
    for file_path in files:
        try:
            state = _build_file_state(
                file_path,
                config=config,
                head_chunks=head_chunks,
                ontology_context_mode_value=ontology_context_mode_value,
                tenant=tenant,
                project=project,
            )
            if use_unit_pipeline:
                state = convert_document(state, tools)
                if state.failure_stage is not None:
                    logger.error(
                        "Error processing %s: %s",
                        file_path,
                        state.failure_reason or "Document conversion failed",
                    )
                    continue
                onto_result, facts_result = await run_unit_pipeline(state, tools)
                await _persist_unit_pipeline_outputs(
                    state, onto_result, facts_result, tools
                )
            else:
                async for _ in workflow.astream(
                    state,
                    stream_mode="values",
                    config=RunnableConfig(recursion_limit=recursion_limit),
                ):
                    pass
        except Exception:
            logger.exception("Error processing %s", file_path)


def create_app(
    tools: ToolBox,
    server_config: ServerConfig,
    head_chunks: int | None = None,
    *,
    active_tenant: str,
    active_project: str,
) -> FastAPI:
    """Build the FastAPI application (routes + workflow).

    ``active_tenant`` / ``active_project`` match the Fuseki/Qdrant partition set at
    server startup; ``/process`` uses them when the request omits ``tenant`` /
    ``project`` query parameters.
    """

    app = FastAPI(title="ontocast", version=metadata.version("ontocast"))
    app.include_router(build_ontology_router(tools))

    workflow: CompiledStateGraph = create_agent_graph(tools)
    recursion_limit = calculate_recursion_limit(
        head_chunks,
        server_config,
    )

    @app.get("/health")
    async def health_check():
        try:
            if tools.llm is None:
                return JSONResponse(
                    status_code=503,
                    content=HealthErrorResponse(
                        error="LLM not initialized"
                    ).model_dump(),
                )
            return HealthOkResponse(
                llm_provider=tools.llm_provider, version=metadata.version("ontocast")
            )
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return JSONResponse(
                status_code=503,
                content=HealthErrorResponse(error=str(e)).model_dump(),
            )

    @app.get("/info", response_model=InfoResponse)
    async def info():
        return InfoResponse(version=metadata.version("ontocast"))

    @app.post("/flush")
    async def flush(
        tenant: str | None = Query(default=None),
        project: str | None = Query(default=None),
    ):
        try:
            if tools.triple_store_manager is None and tools.vector_store is None:
                return JSONResponse(
                    status_code=400,
                    content=StatusErrorBody(
                        error="No triple store or vector store configured",
                    ).model_dump(),
                )

            if tenant is not None or project is not None:
                t = (tenant or DEFAULT_TENANT).strip()
                p = (project or DEFAULT_PROJECT).strip()
                try:
                    await tools.clean_tenancy_data(t, p)
                except NotImplementedError as err:
                    return JSONResponse(
                        status_code=400,
                        content=StatusErrorBody(
                            error=str(err),
                            error_type=type(err).__name__,
                        ).model_dump(),
                    )
                message = (
                    f"Tenancy data flushed for tenant={t!r} project={p!r} "
                    "(triple and/or vector partitions)"
                )
            else:
                if tools.triple_store_manager is not None:
                    await tools.triple_store_manager.clean()
                message = "Triple store flushed successfully (configured scope)"
            return FlushOkResponse(message=message)
        except Exception as e:
            logger.error("Error flushing triple store: %s", e)
            return JSONResponse(
                status_code=500,
                content=StatusErrorBody(
                    error=str(e),
                    error_type=type(e).__name__,
                ).model_dump(),
            )

    @app.post("/process")
    async def process(request: Request):
        workflow_state: dict | None = None
        try:
            content_type = request.headers.get("content-type") or ""
            logger.debug("Content-Type: %s", content_type)

            request_tenant = request.query_params.get("tenant", None)
            request_project = request.query_params.get("project", None)
            has_tenancy_qs = (
                "tenant" in request.query_params or "project" in request.query_params
            )
            render_mode = request.query_params.get("render_mode", None)
            ontology_context_mode = request.query_params.get(
                "ontology_context_mode", None
            )
            ontology_user_instruction = request.query_params.get(
                "ontology_user_instruction", ""
            )
            facts_user_instruction = request.query_params.get(
                "facts_user_instruction", ""
            )
            ontology_context_mode_value: OntologyContextMode = (
                parse_ontology_context_mode_param(
                    ontology_context_mode,
                    server_config.ontology_context_mode,
                )
            )

            if content_type.startswith("application/json"):
                bytes_data = await request.body()
                logger.debug("JSON body length: %s", len(bytes_data))
                files_dict: dict[str, bytes] = {"input.json": bytes_data}
            elif content_type.startswith("multipart/form-data"):
                form = await request.form()
                files_dict = {}
                for key, value in form.multi_items():
                    if isinstance(value, StarletteUploadFile):
                        files_dict[key] = await value.read()
                    elif key == "ontology_user_instruction" and value:
                        ontology_user_instruction = str(value)
                    elif key == "facts_user_instruction" and value:
                        facts_user_instruction = str(value)
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

            if has_tenancy_qs:
                resolved_tenant, resolved_project = _resolve_tenant_project(
                    request_tenant, request_project
                )
                if _stores_use_tenancy_partitions(tools):
                    await tools.update_tenancy_with_vector_mode(
                        resolved_tenant,
                        resolved_project,
                        initialize_vector_store=(
                            ontology_context_mode_value
                            == OntologyContextMode.VECTOR_RETRIEVAL
                        ),
                        fail_on_vector_store_error=False,
                    )
            else:
                resolved_tenant, resolved_project = (
                    active_tenant,
                    active_project,
                )

            render_mode_value: RenderMode = parse_render_mode_param(
                render_mode,
                server_config.render_mode,
            )
            try:
                validate_ontology_context_mode(ontology_context_mode_value, tools)
            except OntologyContextConfigError as e:
                return _ontology_context_error_response(e)

            initial_state = AgentState(
                files=files_dict,
                max_visits=server_config.max_visits_per_node,
                max_chunks=head_chunks,
                render_mode=render_mode_value,
                ontology_context_mode=ontology_context_mode_value,
                ontology_max_triples=server_config.ontology_max_triples,
                tenant=resolved_tenant,
                project=resolved_project,
                ontology_user_instruction=ontology_user_instruction,
                facts_user_instruction=facts_user_instruction,
            )

            async for chunk in workflow.astream(
                initial_state,
                stream_mode="values",
                config=RunnableConfig(recursion_limit=recursion_limit),
            ):
                workflow_state = chunk

            if workflow_state is None:
                raise ValueError("Workflow did not return a valid state")

            budget_tracker_data: dict = {}
            if workflow_state.get("budget_tracker"):
                budget_tracker = workflow_state["budget_tracker"]
                budget_tracker_data = budget_tracker.model_dump()

            total_content_units = len(
                workflow_state.get("content_units", workflow_state.get("chunks", []))
            )
            state_render_mode = workflow_state.get("render_mode")
            render_facts_enabled = state_render_mode in (
                RenderMode.FACTS,
                RenderMode.ONTOLOGY_AND_FACTS,
                RenderMode.FACTS.value,
                RenderMode.ONTOLOGY_AND_FACTS.value,
            )
            if render_facts_enabled:
                processed_content_units = len(
                    workflow_state.get("parallel_facts_units", [])
                )
            else:
                processed_content_units = total_content_units
            chunks_remaining = max(total_content_units - processed_content_units, 0)
            ontology_artifacts = workflow_state.get("reduced_ontology_artifacts") or (
                workflow_state.get("ontology_artifacts", [])
            )

            return ProcessOkResponse(
                data=ProcessResultData(
                    facts=(
                        workflow_state["aggregated_facts"].serialize(format="turtle")
                        if workflow_state.get("aggregated_facts")
                        else ""
                    ),
                    ontology=None,
                    ontology_artifacts=[
                        {
                            "iri": artifact.iri,
                            "ontology_id": artifact.ontology_id,
                            "title": artifact.title,
                            "triples": len(artifact.graph),
                            "ttl": artifact.graph.serialize(format="turtle"),
                        }
                        for artifact in ontology_artifacts
                    ],
                ),
                metadata=ProcessResultMetadata(
                    status=workflow_state["status"],
                    chunks_processed=processed_content_units,
                    chunks_remaining=chunks_remaining,
                    budget=budget_tracker_data,
                    retrieval_metrics=workflow_state.get("retrieval_metrics", {}),
                ),
            )

        except Exception as e:
            logger.error("Error processing document: %s", e)
            logger.error("Error type: %s", type(e))
            logger.error("Error traceback:", exc_info=True)

            error_details = None
            if workflow_state:
                error_details = {
                    "stage": workflow_state.get("failure_stage", "unknown"),
                    "reason": workflow_state.get("failure_reason", "unknown"),
                }

            return JSONResponse(
                status_code=500,
                content=ProcessErrorResponse(
                    error=str(e),
                    error_type=type(e).__name__,
                    error_details=error_details,
                ).model_dump(),
            )

    @app.post("/process_unit")
    async def process_unit(request: Request):
        """Process a single small document or text without chunking or normalization.

        Runs ontology_loop and facts_loop sequentially for the entire input as
        one unit.  The ontology loop's output is fed directly into the facts
        loop so that fact extraction immediately uses the freshly-generated
        ontology.  Accepts the same content types and query parameters as
        ``/process``.
        """
        try:
            content_type = request.headers.get("content-type") or ""
            logger.debug("process_unit Content-Type: %s", content_type)

            request_tenant = request.query_params.get("tenant", None)
            request_project = request.query_params.get("project", None)
            has_tenancy_qs = (
                "tenant" in request.query_params or "project" in request.query_params
            )
            render_mode = request.query_params.get("render_mode", None)
            ontology_context_mode = request.query_params.get(
                "ontology_context_mode", None
            )
            ontology_user_instruction = request.query_params.get(
                "ontology_user_instruction", ""
            )
            facts_user_instruction = request.query_params.get(
                "facts_user_instruction", ""
            )
            ontology_context_mode_value: OntologyContextMode = (
                parse_ontology_context_mode_param(
                    ontology_context_mode,
                    server_config.ontology_context_mode,
                )
            )

            if content_type.startswith("application/json"):
                bytes_data = await request.body()
                logger.debug("process_unit JSON body length: %s", len(bytes_data))
                files_dict: dict[str, bytes] = {"input.json": bytes_data}
            elif content_type.startswith("multipart/form-data"):
                form = await request.form()
                files_dict = {}
                for key, value in form.multi_items():
                    if isinstance(value, StarletteUploadFile):
                        files_dict[key] = await value.read()
                    elif key == "ontology_user_instruction" and value:
                        ontology_user_instruction = str(value)
                    elif key == "facts_user_instruction" and value:
                        facts_user_instruction = str(value)
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

            if has_tenancy_qs:
                resolved_tenant, resolved_project = _resolve_tenant_project(
                    request_tenant, request_project
                )
                if _stores_use_tenancy_partitions(tools):
                    await tools.update_tenancy_with_vector_mode(
                        resolved_tenant,
                        resolved_project,
                        initialize_vector_store=(
                            ontology_context_mode_value
                            == OntologyContextMode.VECTOR_RETRIEVAL
                        ),
                        fail_on_vector_store_error=False,
                    )
            else:
                resolved_tenant, resolved_project = (
                    active_tenant,
                    active_project,
                )

            render_mode_value: RenderMode = parse_render_mode_param(
                render_mode,
                server_config.render_mode,
            )
            try:
                validate_ontology_context_mode(ontology_context_mode_value, tools)
            except OntologyContextConfigError as e:
                return _ontology_context_error_response(e)

            initial_state = AgentState(
                files=files_dict,
                max_visits=server_config.max_visits_per_node,
                max_chunks=1,
                render_mode=render_mode_value,
                ontology_context_mode=ontology_context_mode_value,
                ontology_max_triples=server_config.ontology_max_triples,
                tenant=resolved_tenant,
                project=resolved_project,
                ontology_user_instruction=ontology_user_instruction,
                facts_user_instruction=facts_user_instruction,
            )

            initial_state = convert_document(initial_state, tools)
            if initial_state.failure_stage is not None:
                return JSONResponse(
                    status_code=422,
                    content=ProcessErrorResponse(
                        error=initial_state.failure_reason
                        or "Document conversion failed",
                        error_type="ConversionError",
                        error_details={"stage": str(initial_state.failure_stage)},
                    ).model_dump(),
                )

            onto_result, facts_result = await run_unit_pipeline(initial_state, tools)
            failed_unit_state = None
            if onto_result is not None and onto_result.status == Status.FAILED:
                failed_unit_state = onto_result
            elif facts_result is not None and facts_result.status == Status.FAILED:
                failed_unit_state = facts_result
            if failed_unit_state is not None:
                return JSONResponse(
                    status_code=422,
                    content=ProcessErrorResponse(
                        error=failed_unit_state.failure_reason
                        or "Unit processing failed",
                        error_type="PipelineError",
                        error_details={
                            "stage": (
                                str(failed_unit_state.failure_stage)
                                if failed_unit_state.failure_stage is not None
                                else None
                            )
                        },
                    ).model_dump(),
                )

            budget_tracker_data: dict = initial_state.budget_tracker.model_dump()

            ontology_artifacts: list[dict] = []
            if onto_result is not None:
                delta_graph = build_ontology_delta_graph(onto_result)
                if len(delta_graph) > 0:
                    ontology_artifacts = [
                        {
                            "iri": onto_result.assembly_anchor_iri or "",
                            "ontology_id": None,
                            "title": "Unit ontology artifact",
                            "triples": len(delta_graph),
                            "ttl": delta_graph.serialize(format="turtle"),
                        }
                    ]

            facts_ttl = ""
            if facts_result is not None:
                ontology_graph = _select_unit_facts_ontology_graph(
                    onto_result, facts_result
                )
                postprocessed_facts = tools.aggregator.postprocess_facts_units(
                    units=[facts_result.content_unit],
                    ontology_graph=ontology_graph,
                )
                facts_ttl = postprocessed_facts.serialize(format="turtle")

            last_status = None
            if facts_result is not None:
                last_status = facts_result.status
            elif onto_result is not None:
                last_status = onto_result.status

            return ProcessOkResponse(
                data=ProcessResultData(
                    facts=facts_ttl,
                    ontology=None,
                    ontology_artifacts=ontology_artifacts,
                ),
                metadata=ProcessResultMetadata(
                    status=str(last_status) if last_status is not None else None,
                    chunks_processed=1,
                    chunks_remaining=0,
                    budget=budget_tracker_data,
                    retrieval_metrics=initial_state.retrieval_metrics,
                ),
            )

        except Exception as e:
            logger.error("Error in process_unit: %s", e)
            logger.error("Error type: %s", type(e))
            logger.error("Error traceback:", exc_info=True)
            return JSONResponse(
                status_code=500,
                content=ProcessErrorResponse(
                    error=str(e),
                    error_type=type(e).__name__,
                    error_details=None,
                ).model_dump(),
            )

    return app


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
def run(
    input_path: pathlib.Path | None,
    head_chunks: int | None,
    use_unit_pipeline: bool,
    tenant: str | None,
    project: str | None,
):
    """
    Main entry point for the OntoCast server/CLI.

    Backend selection is automatically inferred from available configuration:
    - Fuseki: If FUSEKI_URI and FUSEKI_AUTH are provided (preferred)
    - Filesystem Triple Store: If ONTOCAST_WORKING_DIRECTORY and
      ONTOCAST_ONTOLOGY_DIRECTORY are provided
    - Filesystem Manager: If ONTOCAST_WORKING_DIRECTORY is provided
      (can be combined with other backends)

    No explicit backend configuration flags are needed; backends are inferred.

    """

    config = Config()
    config.validate_llm_config()
    _configure_logging(config)
    _prepare_path_config(config)

    # Create ToolBox with config
    tools: ToolBox = ToolBox(config)
    t_res, p_res = _resolve_tenant_project(tenant, project)
    ontology_context_mode_value = config.server.ontology_context_mode
    vector_mode_enabled = (
        ontology_context_mode_value == OntologyContextMode.VECTOR_RETRIEVAL
    )
    if _stores_use_tenancy_partitions(tools):
        asyncio.run(
            tools.update_tenancy_with_vector_mode(
                t_res,
                p_res,
                initialize_vector_store=vector_mode_enabled,
                fail_on_vector_store_error=vector_mode_enabled,
            )
        )

    if input_path is not None and config.clean:
        asyncio.run(_flush_triple_configured_scope(tools))

    asyncio.run(
        tools.initialize(
            ontology_context_mode=ontology_context_mode_value,
            fail_on_vector_store_error=vector_mode_enabled,
        )
    )
    validate_ontology_context_mode(ontology_context_mode_value, tools)

    workflow: CompiledStateGraph = create_agent_graph(tools)

    if input_path is not None:
        input_path = input_path.expanduser()
        files = sorted(
            crawl_directories(
                input_path,
                suffixes=tuple([".json"] + list(tools.converter.supported_extensions)),
            )
        )
        asyncio.run(
            _process_files_input(
                files,
                config=config,
                head_chunks=head_chunks,
                use_unit_pipeline=use_unit_pipeline,
                tools=tools,
                workflow=workflow,
                ontology_context_mode_value=ontology_context_mode_value,
                tenant=t_res,
                project=p_res,
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
