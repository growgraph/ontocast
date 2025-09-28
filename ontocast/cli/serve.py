"""OntoCast API server implementation.

This module provides a web server implementation for the OntoCast framework
using Robyn. It exposes REST API endpoints for processing documents and
extracting semantic triples with ontology assistance.

The server supports:
- Health check endpoint (/health)
- Service information endpoint (/info)
- Document processing endpoint (/process)
- Multiple input formats (JSON, multipart/form-data)
- Streaming workflow execution
- Comprehensive error handling and logging

The server integrates with the OntoCast workflow graph to process documents
through the complete pipeline: chunking, ontology selection, fact extraction,
and aggregation.

Example:
    python -m ontocast.cli.serve --env-path .env --working-directory ./work
"""

import asyncio
import logging
import logging.config
import pathlib

import click
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from robyn import Headers, Request, Response, Robyn, jsonify

from ontocast.cli.util import crawl_directories
from ontocast.config import Config
from ontocast.onto import AgentState
from ontocast.stategraph import create_agent_graph
from ontocast.toolbox import ToolBox, init_toolbox

logger = logging.getLogger(__name__)


def calculate_recursion_limit(
    max_visits: int,
    head_chunks: int | None = None,
    base_recursion_limit: int = 1000,
    estimated_chunks: int = 30,
) -> int:
    """Calculate the recursion limit based on max_visits and head_chunks.

    Args:
        max_visits: Maximum number of visits allowed per node
        head_chunks: Optional maximum number of chunks to process
        base_recursion_limit: Base recursion limit from config
        estimated_chunks: Estimated number of chunks from config

    Returns:
        int: Calculated recursion limit
    """
    if head_chunks is not None:
        # If we know the number of chunks, calculate exact limit
        return max(base_recursion_limit, max_visits * head_chunks * 10)
    else:
        # If we don't know chunks, use a conservative estimate
        return max(base_recursion_limit, max_visits * estimated_chunks * 10)


def create_app(
    tools: ToolBox,
    head_chunks: int | None = None,
    max_visits: int = 3,
    skip_ontology_development=False,
    base_recursion_limit: int = 1000,
    estimated_chunks: int = 30,
):
    app = Robyn(__file__)
    workflow: CompiledStateGraph = create_agent_graph(tools)
    recursion_limit = calculate_recursion_limit(
        max_visits, head_chunks, base_recursion_limit, estimated_chunks
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
                    "llm_provider": tools.llm.provider,
                    "model_name": tools.llm.model,
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
            elif content_type and content_type.startswith("multipart/form-data"):
                files = request.files
                logger.debug(f"Files: {files.keys()}")
                logger.debug(f"Files-types: {[(k, type(v)) for k, v in files.items()]}")
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

            initial_state = AgentState(
                files=files,
                max_visits=max_visits,
                max_chunks=head_chunks,
                skip_ontology_development=skip_ontology_development,
            )

            async for chunk in workflow.astream(
                initial_state,
                stream_mode="values",
                config=RunnableConfig(recursion_limit=recursion_limit),
            ):
                workflow_state = chunk

            if workflow_state is None:
                raise ValueError("Workflow did not return a valid state")

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
    "--env-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    default=".env",
    help=(
        "Path to .env file. If NEO4J_URI and NEO4J_AUTH are set, "
        "neo4j will be used as triple store. If FUSEKI_URI and FUSEKI_AUTH are set, "
        "Fuseki will be used as triple store (preferred over Neo4j)."
    ),
)
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), default=None)
@click.option("--head-chunks", type=int, default=None)
def run(
    env_path: pathlib.Path,
    input_path: pathlib.Path | None,
    head_chunks: int | None,
):
    """
    Main entry point for the OntoCast server/CLI.
    If FUSEKI_URI and FUSEKI_AUTH are set in the environment,
        Fuseki will be used as the triple store backend (preferred).
    If NEO4J_URI and NEO4J_AUTH are set in the environment,
        Neo4j will be used as the triple store backend (if Fuseki not available).
    Otherwise, the filesystem backend is used.

    If --clean is set, the triple store (Neo4j or Fuseki) will be initialized as clean (all data deleted on startup).
    """

    _ = load_dotenv(dotenv_path=env_path.expanduser())
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

    # Use CLI arguments or fall back to config values
    if config.paths.working_directory is not None:
        config.paths.working_directory = pathlib.Path(
            config.paths.working_directory
        ).expanduser()
        config.paths.working_directory.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(
            "Working directory must be provided via CLI argument or WORKING_DIRECTORY config"
        )

    if config.paths.ontology_directory is not None:
        config.paths.ontology_directory = pathlib.Path(
            config.paths.ontology_directory
        ).expanduser()

    # Get triple store configuration
    triple_store_config = config.get_triple_store_config()

    # Get LLM configuration
    llm_config = config.get_llm_config()

    tools: ToolBox = ToolBox(
        working_directory=config.paths.working_directory,
        ontology_directory=config.paths.ontology_directory,
        **llm_config,
        **{k: v for k, v in triple_store_config.items() if k != "type"},
    )
    init_toolbox(tools)

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
            config.server.max_visits,
            head_chunks,
            config.server.recursion_limit,
            config.server.estimated_chunks,
        )

        async def process_files():
            for file_path in files:
                try:
                    state = AgentState(
                        files={file_path.as_posix(): file_path.read_bytes()},
                        max_visits=config.server.max_visits,
                        max_chunks=head_chunks,
                        skip_ontology_development=config.skip_ontology_development,
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
            tools,
            head_chunks,
            max_visits=config.server.max_visits,
            skip_ontology_development=config.skip_ontology_development,
            base_recursion_limit=config.server.recursion_limit,
            estimated_chunks=config.server.estimated_chunks,
        )
        server_config = config.get_server_config()
        logger.info(f"Starting Ontocast server on port {server_config['port']}")
        app.start(port=server_config["port"])


if __name__ == "__main__":
    run()
