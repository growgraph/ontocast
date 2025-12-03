"""OntoCast API server implementation.

This module provides a web server implementation for the OntoCast framework
using Robyn. It exposes REST API endpoints for processing documents and
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
    python -m ontocast.cli.serve --env-path .env

    # Process specific file
    python -m ontocast.cli.serve --env-path .env --input-path ./document.pdf

    # Process with chunk limit
    python -m ontocast.cli.serve --env-path .env --head-chunks 5
"""

import asyncio
import logging
import logging.config
import pathlib

import click
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ontocast.cli.util import crawl_directories
from ontocast.config import Config, ServerConfig
from ontocast.onto.state import AgentState
from ontocast.stategraph import create_agent_graph
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def calculate_recursion_limit(
    head_chunks: int | None,
    server_config: ServerConfig,
) -> int:
    """Calculate the recursion limit based on max_visits and head_chunks.

    Args:
        head_chunks: Optional maximum number of chunks to process

    Returns:
        int: Calculated recursion limit
    """
    if head_chunks is not None:
        # If we know the number of chunks, calculate exact limit
        return max(
            server_config.base_recursion_limit,
            server_config.max_visits * head_chunks * 10,
        )
    else:
        # If we don't know chunks, use a conservative estimate
        return max(
            server_config.base_recursion_limit,
            server_config.max_visits * server_config.estimated_chunks * 10,
        )


def create_app(
    tools: ToolBox,
    server_config: ServerConfig,
    head_chunks: int | None = None,
):
    from robyn import Headers, Request, Response, Robyn, jsonify

    app = Robyn(__file__)
    workflow: CompiledStateGraph = create_agent_graph(tools)
    recursion_limit = calculate_recursion_limit(
        head_chunks,
        server_config,
    )

    @app.get("/health")
    async def health_check():
        """MCP health check endpoint."""
        try:
            # Check if LLM is available
            if tools.llm is None:
                return Response(
                    status_code=503,
                    headers=Headers({"Content-Type": "application/json"}),
                    description=jsonify(
                        {"status": "unhealthy", "error": "LLM not initialized"}
                    ),
                )

            return Response(
                status_code=200,
                headers=Headers({"Content-Type": "application/json"}),
                description=jsonify(
                    {
                        "status": "healthy",
                        "version": "0.1.1",
                        "llm_provider": tools.llm_provider,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return Response(
                status_code=503,
                headers=Headers({"Content-Type": "application/json"}),
                description=jsonify({"status": "unhealthy", "error": str(e)}),
            )

    @app.get("/info")
    async def info():
        """MCP info endpoint."""
        return Response(
            status_code=200,
            headers=Headers({"Content-Type": "application/json"}),
            description=jsonify(
                {
                    "name": "ontocast",
                    "version": "0.1.1",
                    "description": "Agentic ontology assisted framework "
                    "for semantic triple extraction",
                    "capabilities": ["text-to-triples", "ontology-extraction"],
                    "input_types": ["text", "json", "pdf", "markdown"],
                    "output_types": ["turtle", "json"],
                }
            ),
        )

    @app.post("/flush")
    async def flush(request: Request):
        """Flush/clean data from the triple store.

        This endpoint deletes data from the configured triple store.
        For Fuseki, you can specify a dataset query parameter to clean a specific dataset,
        or omit it to clean all datasets. For Neo4j, this deletes all nodes (dataset parameter is ignored).

        Query Parameters:
            dataset (optional): For Fuseki only - name of the dataset to clean.
                If omitted, cleans all datasets (main and ontologies).

        Warning: This operation is irreversible and will delete all data.

        Returns:
            JSON response with status and message.

        Example:
            # Clean all datasets (Fuseki) or entire database (Neo4j)
            POST /flush

            # Clean specific Fuseki dataset
            POST /flush?dataset=my_dataset
        """
        try:
            if tools.triple_store_manager is None:
                return Response(
                    status_code=400,
                    headers=Headers({"Content-Type": "application/json"}),
                    description=jsonify(
                        {
                            "status": "error",
                            "error": "No triple store manager configured",
                        }
                    ),
                )

            # Extract dataset parameter (used by Fuseki, ignored by others)
            dataset = request.query_params.get("dataset", None)

            # All implementations accept the dataset parameter
            # Fuseki uses it, Neo4j and Filesystem ignore it with a warning
            await tools.triple_store_manager.clean(dataset=dataset)

            # Generate appropriate success message
            from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager

            if isinstance(tools.triple_store_manager, FusekiTripleStoreManager):
                if dataset:
                    message = f"Fuseki dataset '{dataset}' flushed successfully"
                else:
                    message = "Fuseki triple store flushed successfully (all datasets)"
            else:
                message = "Triple store flushed successfully"

            return Response(
                status_code=200,
                headers=Headers({"Content-Type": "application/json"}),
                description=jsonify(
                    {
                        "status": "success",
                        "message": message,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error flushing triple store: {str(e)}")
            return Response(
                status_code=500,
                headers=Headers({"Content-Type": "application/json"}),
                description=jsonify(
                    {
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                ),
            )

    @app.post("/process")
    async def process(request: Request):
        """MCP process endpoint."""
        workflow_state: dict | None = None
        try:
            content_type = request.headers.get("content-type")
            logger.debug(f"Content-Type: {content_type}")
            logger.debug(f"Request headers: {request.headers}")
            logger.debug(f"Request body: {request.body}")

            # Extract parameters from query parameters
            dataset = request.query_params.get("dataset", None)
            if dataset:
                logger.debug(f"Using dataset: {dataset}")

            # Extract skip_facts_rendering from query parameters
            skip_facts_rendering = request.query_params.get(
                "skip_facts_rendering", None
            )
            if skip_facts_rendering:
                logger.debug(f"Using skip_facts_rendering: {skip_facts_rendering}")

            # Extract skip_ontology_development from query parameters
            skip_ontology_development = request.query_params.get(
                "skip_ontology_development", None
            )
            if skip_ontology_development:
                logger.debug(
                    f"Using skip_ontology_development: {skip_ontology_development}"
                )

            # Extract user instructions from query parameters (available for both JSON and multipart)
            ontology_user_instruction = request.query_params.get(
                "ontology_user_instruction", ""
            )
            facts_user_instruction = request.query_params.get(
                "facts_user_instruction", ""
            )
            if ontology_user_instruction:
                logger.debug(
                    f"Query param - ontology_user_instruction: {ontology_user_instruction}"
                )
            if facts_user_instruction:
                logger.debug(
                    f"Query param - facts_user_instruction: {facts_user_instruction}"
                )

            if content_type and content_type.startswith("application/json"):
                data = request.body
                # Convert string to bytes if needed
                if isinstance(data, str):
                    bytes_data = data.encode("utf-8")
                else:
                    bytes_data = data
                logger.debug(
                    f"Parsed JSON data: {data}, bytes length: {len(bytes_data)}"
                )
                files = {"input.json": bytes_data}
                # User instructions already extracted from query params above
                # They can also be overridden by convert_document.py for JSON files
            elif content_type and content_type.startswith("multipart/form-data"):
                files = request.files
                logger.debug(f"Files: {files.keys()}")
                logger.debug(f"Files-types: {[(k, type(v)) for k, v in files.items()]}")

                # Check if form data contains user instructions (overrides query params)
                if hasattr(request, "form_data") and request.form_data:
                    form_ontology_instruction = request.form_data.get(
                        "ontology_user_instruction", ""
                    )
                    form_facts_instruction = request.form_data.get(
                        "facts_user_instruction", ""
                    )
                    if form_ontology_instruction:
                        ontology_user_instruction = form_ontology_instruction
                        logger.debug(
                            f"Form data - ontology_user_instruction: "
                            f"{ontology_user_instruction}"
                        )
                    if form_facts_instruction:
                        facts_user_instruction = form_facts_instruction
                        logger.debug(
                            f"Form data - facts_user_instruction: {facts_user_instruction}"
                        )
                if not files:
                    return Response(
                        status_code=400,
                        headers=Headers({"Content-Type": "application/json"}),
                        description=jsonify(
                            {
                                "status": "error",
                                "error": "No file provided",
                                "error_type": "ValidationError",
                            }
                        ),
                    )
            else:
                logger.debug(f"Unsupported content type: {content_type}")
                return Response(
                    status_code=400,
                    headers=Headers({"Content-Type": "application/json"}),
                    description=jsonify(
                        {
                            "status": "error",
                            "error": f"Unsupported content type: {content_type}",
                            "error_type": "ValidationError",
                        }
                    ),
                )

            # Update dataset if provided (efficient - no model reloading)
            if dataset:
                await tools.update_dataset(dataset)

            # Determine boolean flags from API params or server config
            def parse_bool_param(value, default):
                """Parse a boolean parameter from query string or use default.

                Args:
                    value: String value from query params (e.g., "true", "false", "1", "0")
                    default: Default boolean value from server config

                Returns:
                    bool: Parsed boolean value
                """
                if value is None:
                    return default
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    # Handle string representations of booleans
                    value_lower = value.lower().strip()
                    if value_lower in ("true", "1", "yes", "on"):
                        return True
                    if value_lower in ("false", "0", "no", "off"):
                        return False
                # If value is truthy but not a recognized boolean string, return default
                return default

            skip_facts_rendering_value: bool = parse_bool_param(
                skip_facts_rendering, server_config.skip_facts_rendering
            )
            skip_ontology_development_value: bool = parse_bool_param(
                skip_ontology_development, server_config.skip_ontology_development
            )

            initial_state = AgentState(
                files=files,
                max_visits=server_config.max_visits,
                max_chunks=head_chunks,
                skip_ontology_development=skip_ontology_development_value,
                skip_facts_rendering=skip_facts_rendering_value,
                ontology_max_triples=server_config.ontology_max_triples,
                dataset=dataset,
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

            # Extract budget tracker data if available
            budget_tracker_data = {}
            if workflow_state.get("budget_tracker"):
                budget_tracker = workflow_state["budget_tracker"]
                # Convert Pydantic model to dict using model_dump()
                budget_tracker_data = budget_tracker.model_dump()

            result = {
                "status": "success",
                "data": {
                    "facts": workflow_state["aggregated_facts"].serialize(
                        format="turtle"
                    )
                    if workflow_state.get("aggregated_facts")
                    else "",
                    "ontology": workflow_state["current_ontology"].graph.serialize(
                        format="turtle"
                    )
                    if workflow_state.get("current_ontology")
                    else "",
                },
                "metadata": {
                    "status": workflow_state["status"],
                    "chunks_processed": len(workflow_state.get("chunks_processed", [])),
                    "chunks_remaining": len(workflow_state.get("chunks", [])),
                    "budget": budget_tracker_data,
                },
            }

            return Response(
                status_code=200,
                headers=Headers({"Content-Type": "application/json"}),
                description=jsonify(result),
            )

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error("Error traceback:", exc_info=True)

            # Try to get error details from workflow_state if available
            error_details = None
            if workflow_state:
                error_details = {
                    "stage": workflow_state.get("failure_stage", "unknown"),
                    "reason": workflow_state.get("failure_reason", "unknown"),
                }

            return Response(
                status_code=500,
                headers=Headers({"Content-Type": "application/json"}),
                description=jsonify(
                    {
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_details": error_details,
                    }
                ),
            )

    return app


@click.command()
@click.option(
    "--env-file",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    default=".env",
    help="Path to .env file containing backend and configuration settings",
)
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), default=None)
@click.option("--head-chunks", type=int, default=None)
def run(
    env_file: pathlib.Path,
    input_path: pathlib.Path | None,
    head_chunks: int | None,
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

    _ = load_dotenv(dotenv_path=env_file.expanduser())
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

        async def process_files():
            for file_path in files:
                try:
                    state = AgentState(
                        files={file_path.as_posix(): file_path.read_bytes()},
                        max_visits=config.server.max_visits,
                        max_chunks=head_chunks,
                        skip_ontology_development=config.server.skip_ontology_development,
                        skip_facts_rendering=config.server.skip_facts_rendering,
                        dataset=config.tool_config.fuseki.dataset,
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
        logger.info(f"Starting Ontocast server on port {config.server.port}")
        app.start(port=config.server.port)


if __name__ == "__main__":
    run()
