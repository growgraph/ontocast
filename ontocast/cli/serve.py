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
from ontocast.onto.enum import OntologyContextMode, RenderMode
from ontocast.onto.state import AgentState
from ontocast.onto.tenancy import DEFAULT_PROJECT, DEFAULT_TENANT
from ontocast.stategraph import create_agent_graph
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


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


def create_app(
    tools: ToolBox,
    server_config: ServerConfig,
    head_chunks: int | None = None,
) -> FastAPI:
    """Build the FastAPI application (routes + workflow)."""

    app = FastAPI(title="ontocast", version=metadata.version("ontocast"))
    app.include_router(build_ontology_router(tools))

    workflow: CompiledStateGraph = create_agent_graph(tools)
    recursion_limit = calculate_recursion_limit(
        head_chunks,
        server_config,
    )

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
                    f"Invalid render_mode '{value}', using default '{default.value}'"
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

            resolved_tenant: str | None = None
            resolved_project: str | None = None
            if has_tenancy_qs:
                resolved_tenant = (request_tenant or DEFAULT_TENANT).strip()
                resolved_project = (request_project or DEFAULT_PROJECT).strip()
                await tools.update_tenancy(resolved_tenant, resolved_project)

            render_mode_value: RenderMode = parse_render_mode_param(
                render_mode,
                server_config.render_mode,
            )
            ontology_context_mode_value: OntologyContextMode = (
                parse_ontology_context_mode_param(
                    ontology_context_mode,
                    server_config.ontology_context_mode,
                )
            )

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

            return ProcessOkResponse(
                data=ProcessResultData(
                    facts=(
                        workflow_state["aggregated_facts"].serialize(format="turtle")
                        if workflow_state.get("aggregated_facts")
                        else ""
                    ),
                    ontology=(
                        workflow_state["current_ontology"].graph.serialize(
                            format="turtle"
                        )
                        if workflow_state.get("current_ontology")
                        else ""
                    ),
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

    return app


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), default=None)
@click.option("--head-chunks", type=int, default=None)
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
    tenant: str | None,
    project: str | None,
):
    """
    Main entry point for the OntoCast server/CLI.

    Backend selection is automatically inferred from available configuration:
    - Fuseki: If FUSEKI_URI and FUSEKI_AUTH are provided (preferred)
    - Neo4j: If NEO4J_URI and NEO4J_AUTH are provided (fallback)
    - Filesystem Triple Store: If ONTOCAST_WORKING_DIRECTORY and ONTOCAST_ONTOLOGY_DIRECTORY are provided
    - Filesystem Manager: If ONTOCAST_WORKING_DIRECTORY is provided (can be combined with other backends)

    No explicit backend configuration flags are needed - backends are automatically detected.

    """

    # Global configuration instance
    config = Config()

    # Validate LLM configuration
    config.validate_llm_config()

    if config.logging_level is not None:
        try:
            logger_conf = f"logging.{config.logging_level}.conf"
            logging.config.fileConfig(logger_conf, disable_existing_loggers=False)
            logger.debug("debug is on")
        except Exception as e:
            logger.error(f"could set logging level correctly {e}")

    if config.tool_config.path_config.working_directory is not None:
        config.tool_config.path_config.working_directory = pathlib.Path(
            config.tool_config.path_config.working_directory
        ).expanduser()
        config.tool_config.path_config.working_directory.mkdir(
            parents=True, exist_ok=True
        )
    else:
        raise ValueError(
            "Working directory must be provided via CLI argument or WORKING_DIRECTORY config"
        )

    if config.tool_config.path_config.ontology_directory is not None:
        config.tool_config.path_config.ontology_directory = pathlib.Path(
            config.tool_config.path_config.ontology_directory
        ).expanduser()

    # Create ToolBox with config
    tools: ToolBox = ToolBox(config)
    if tenant is not None or project is not None:
        t_res = (tenant or DEFAULT_TENANT).strip()
        p_res = (project or DEFAULT_PROJECT).strip()
        asyncio.run(tools.update_tenancy(t_res, p_res))
    asyncio.run(tools.initialize())

    workflow: CompiledStateGraph = create_agent_graph(tools)

    if input_path:
        input_path = input_path.expanduser()

        files = sorted(
            crawl_directories(
                input_path,
                suffixes=tuple([".json"] + list(tools.converter.supported_extensions)),
            )
        )

        recursion_limit = calculate_recursion_limit(
            head_chunks,
            config.server,
        )

        t_state = (tenant or DEFAULT_TENANT).strip()
        p_state = (project or DEFAULT_PROJECT).strip()

        async def process_files():
            for file_path in files:
                try:
                    state = AgentState(
                        files={file_path.as_posix(): file_path.read_bytes()},
                        max_visits=config.server.max_visits_per_node,
                        max_chunks=head_chunks,
                        render_mode=config.server.render_mode,
                        ontology_context_mode=config.server.ontology_context_mode,
                        tenant=t_state,
                        project=p_state,
                    )
                    async for _ in workflow.astream(
                        state,
                        stream_mode="values",
                        config=RunnableConfig(recursion_limit=recursion_limit),
                    ):
                        pass

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

        asyncio.run(process_files())
    else:
        app = create_app(
            tools=tools,
            server_config=config.server,
            head_chunks=head_chunks,
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
