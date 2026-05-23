"""OntoCast API server implementation.

This module provides a web server implementation for the OntoCast framework
using FastAPI/uvicorn. It exposes REST API endpoints for processing documents and
extracting semantic triples with ontology assistance.

The server supports:
- Health check endpoint (/health)
- Service information endpoint (/info)
- Document processing endpoint (/process)
- Unit processing endpoint (/process_unit)
- Ontology upload, replace, and delete (``/ontologies``; optional ``tenant`` /
  ``project`` query parameters, same semantics as ``/process``)
- Triple store flush endpoint (/flush)
- Multiple input formats (JSON, multipart/form-data)
- Streaming workflow execution
- Comprehensive error handling and logging

Optional query (or multipart) parameter ``strip_provenance`` on ``/process`` and
``/process_unit``: when true (``1``, ``true``, ``yes``, ``on``), returned Turtle
for facts and ontology artifacts omits reification/provenance scaffolding
(:class:`~ontocast.tool.triple_manager.core.TripleStoreManager.strip_provenance`).

Per-request ``max_visits`` (query, multipart field, or JSON body field) overrides
``max_visits_per_node`` from server config and limits render/critic loops in
``/process_unit`` (:func:`~ontocast.stategraph.unit_pipeline.run_unit_pipeline`)
and per-chunk loops in ``/process``.

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
from pydantic import BaseModel, ConfigDict, Field

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
from ontocast.api.tenancy_resolution import (
    apply_request_tenancy,
    resolve_tenant_project,
    stores_use_tenancy_partitions,
)
from ontocast.cli.http_responses import (
    invalid_max_visits_response,
    ontology_context_config_error_response,
)
from ontocast.cli.process_request import (
    build_agent_state_from_parsed,
    load_parsed_process_request,
)
from ontocast.cli.util import crawl_directories
from ontocast.config import Config, ServerConfig
from ontocast.onto.enum import (
    OntologyContextMode,
    RenderMode,
    Status,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import (
    OntologyContextConfigError,
    validate_ontology_context_mode,
)
from ontocast.onto.state import AgentState
from ontocast.onto.tenancy import DEFAULT_PROJECT, DEFAULT_TENANT
from ontocast.stategraph import create_agent_graph
from ontocast.stategraph.helpers import build_ontology_delta_graph
from ontocast.stategraph.unit_pipeline import DocumentConversionError, run_unit_pipeline
from ontocast.tool.agg.entity_aligner import EntityAligner
from ontocast.tool.agg.match_derivation import derive_pair_matches
from ontocast.tool.agg.match_models import (
    EntityCluster,
    EntityMatch,
    MatchRegime,
    TaggedGraph,
)
from ontocast.tool.agg.triple_evaluator import TripleSetEvaluator
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


class TaggedGraphInput(BaseModel):
    id: str
    graph: RDFGraph

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AlignEntitiesRequest(BaseModel):
    graphs: list[TaggedGraphInput]
    regime: MatchRegime = MatchRegime.ONTOLOGY_LOOSE
    similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"


class AlignEntitiesResponse(BaseModel):
    data: dict


class DeriveMatchesRequest(BaseModel):
    clusters: list[EntityCluster]
    predicted_graph_id: str
    gt_graph_id: str
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DeriveMatchesResponse(BaseModel):
    data: dict


class EvaluateMatchRequest(BaseModel):
    predicted_graph: RDFGraph
    gt_graph: RDFGraph
    entity_matches: list[EntityMatch]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EvaluateMatchResponse(BaseModel):
    data: dict


def get_supported_input_extensions(tools: ToolBox) -> tuple[str, ...]:
    """Return all input file suffixes handled by document conversion."""
    built_in_suffixes = {".json", ".jsonl", ".txt"}
    converter_suffixes = set(tools.converter.supported_extensions)
    return tuple(sorted(built_in_suffixes | converter_suffixes))


def turtle_from_graph(graph: RDFGraph, *, strip_provenance: bool) -> str:
    """Serialize ``graph`` to Turtle, optionally stripping reification/provenance."""
    out: RDFGraph = (
        TripleStoreManager.strip_provenance(graph) if strip_provenance else graph
    )
    return out.serialize_canonical_turtle()


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
    *,
    max_visits_per_node: int | None = None,
) -> int:
    """Calculate the recursion limit based on max visits and head chunks.

    Args:
        head_chunks: Optional maximum number of chunks to process
        server_config: Server configuration
        max_visits_per_node: Per-request override; defaults to server config

    Returns:
        int: Calculated recursion limit
    """
    visits = (
        max_visits_per_node
        if max_visits_per_node is not None
        else server_config.max_visits_per_node
    )
    if head_chunks is not None:
        # If we know the number of chunks, calculate exact limit
        return max(
            server_config.base_recursion_limit,
            visits * head_chunks * 10,
        )
    else:
        # If we don't know chunks, use a conservative estimate
        return max(
            server_config.base_recursion_limit,
            visits * server_config.estimated_chunks * 10,
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


def expand_input_to_states(
    file_path: pathlib.Path,
    *,
    config: Config,
    head_chunks: int | None,
    ontology_context_mode_value: OntologyContextMode,
    tenant: str | None,
    project: str | None,
) -> list[AgentState]:
    """Expand a local input file into one ``AgentState`` per logical record.

    Pre-processing step that lives outside the agent loop:
    - ``.jsonl`` files are fanned out into N states, one per non-empty line.
    - All other extensions produce exactly one state with the file bytes.
    """
    file_bytes = file_path.read_bytes()
    base_state_kwargs = {
        "max_visits": config.server.max_visits_per_node,
        "max_chunks": head_chunks,
        "render_mode": config.server.render_mode,
        "llm_graph_format": config.server.llm_graph_format,
        "ontology_context_mode": ontology_context_mode_value,
        "ontology_context_fixed_ontology_id": (
            config.server.ontology_context_fixed_ontology_id
        ),
        "tenant": tenant,
        "project": project,
    }

    if file_path.suffix.lower() != ".jsonl":
        return [
            AgentState(
                raw_input={file_path.as_posix(): file_bytes},
                **base_state_kwargs,
            )
        ]

    states: list[AgentState] = []
    for line_number, line in enumerate(
        file_bytes.decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        # Treat each JSONL line as an independent JSON document/state.
        virtual_path = f"{file_path.as_posix()}:{line_number}.json"
        states.append(
            AgentState(
                raw_input={virtual_path: line.encode("utf-8")},
                **base_state_kwargs,
            )
        )
    return states


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
            states = expand_input_to_states(
                file_path,
                config=config,
                head_chunks=head_chunks,
                ontology_context_mode_value=ontology_context_mode_value,
                tenant=tenant,
                project=project,
            )
            for state in states:
                if use_unit_pipeline:
                    try:
                        onto_result, facts_result = await run_unit_pipeline(
                            state, tools
                        )
                    except DocumentConversionError as exc:
                        logger.error("Error processing %s: %s", file_path, exc)
                        continue
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
    server startup. ``/process``, ``/process_unit``, and ``/ontologies`` use them
    when the request omits ``tenant`` / ``project`` query parameters.
    """

    app = FastAPI(title="ontocast", version=metadata.version("ontocast"))
    app.include_router(
        build_ontology_router(
            tools,
            active_tenant=active_tenant,
            active_project=active_project,
            server_config=server_config,
        )
    )

    workflow: CompiledStateGraph = create_agent_graph(tools)

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

    @app.post("/match/entities", response_model=AlignEntitiesResponse)
    async def align_entities(request: AlignEntitiesRequest):
        try:
            aligner = EntityAligner(
                embedding_model=request.embedding_model,
                similarity_threshold=request.similarity_threshold,
            )
            tagged_graphs = [
                TaggedGraph(id=item.id, graph=item.graph) for item in request.graphs
            ]
            result = aligner.align_graphs(tagged_graphs, regime=request.regime)
            return AlignEntitiesResponse(data=result.model_dump(mode="json"))
        except Exception as e:
            logger.error("Error aligning entities: %s", e)
            return JSONResponse(
                status_code=500,
                content=StatusErrorBody(
                    error=str(e),
                    error_type=type(e).__name__,
                ).model_dump(),
            )

    @app.post("/match/derive-matches", response_model=DeriveMatchesResponse)
    async def derive_matches(request: DeriveMatchesRequest):
        try:
            entity_matches = derive_pair_matches(
                request.clusters,
                request.predicted_graph_id,
                request.gt_graph_id,
                similarity_threshold=request.similarity_threshold,
            )
            return DeriveMatchesResponse(
                data={
                    "entity_matches": [
                        match.model_dump(mode="json") for match in entity_matches
                    ]
                }
            )
        except Exception as e:
            logger.error("Error deriving entity matches: %s", e)
            return JSONResponse(
                status_code=500,
                content=StatusErrorBody(
                    error=str(e),
                    error_type=type(e).__name__,
                ).model_dump(),
            )

    @app.post("/match/evaluate", response_model=EvaluateMatchResponse)
    async def evaluate_match(request: EvaluateMatchRequest):
        try:
            metrics = TripleSetEvaluator().evaluate(
                predicted_graph=request.predicted_graph,
                gt_graph=request.gt_graph,
                entity_matches=request.entity_matches,
            )
            return EvaluateMatchResponse(data=metrics.model_dump(mode="json"))
        except Exception as e:
            logger.error("Error evaluating RDF triple sets: %s", e)
            return JSONResponse(
                status_code=500,
                content=StatusErrorBody(
                    error=str(e),
                    error_type=type(e).__name__,
                ).model_dump(),
            )

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
            loaded = await load_parsed_process_request(
                request, server_config, log_label="process"
            )
            if isinstance(loaded, JSONResponse):
                return loaded

            resolved_tenant, resolved_project = await apply_request_tenancy(
                request,
                tools,
                active_tenant=active_tenant,
                active_project=active_project,
                initialize_vector_store=(
                    loaded.ontology_context_mode_value
                    == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
                ),
            )

            try:
                validate_ontology_context_mode(
                    loaded.ontology_context_mode_value, tools
                )
            except OntologyContextConfigError as e:
                return ontology_context_config_error_response(e)

            initial_state = build_agent_state_from_parsed(
                loaded,
                server_config=server_config,
                resolved_tenant=resolved_tenant,
                resolved_project=resolved_project,
                max_chunks=head_chunks,
            )
            request_recursion_limit = calculate_recursion_limit(
                head_chunks,
                server_config,
                max_visits_per_node=initial_state.max_visits,
            )

            async for chunk in workflow.astream(
                initial_state,
                stream_mode="values",
                config=RunnableConfig(recursion_limit=request_recursion_limit),
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

            ontology_artifact_payloads: list[dict] = []
            for artifact in ontology_artifacts:
                out_graph = (
                    TripleStoreManager.strip_provenance(artifact.graph)
                    if loaded.strip_provenance
                    else artifact.graph
                )
                ontology_artifact_payloads.append(
                    {
                        "iri": artifact.iri,
                        "ontology_id": artifact.ontology_id,
                        "title": artifact.title,
                        "triples": len(out_graph),
                        "ttl": out_graph.serialize_canonical_turtle(),
                    }
                )

            return ProcessOkResponse(
                data=ProcessResultData(
                    facts=(
                        turtle_from_graph(
                            workflow_state["aggregated_facts"],
                            strip_provenance=loaded.strip_provenance,
                        )
                        if workflow_state.get("aggregated_facts")
                        else ""
                    ),
                    ontology=None,
                    ontology_artifacts=ontology_artifact_payloads,
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
            if (
                isinstance(e, ValueError)
                and str(e) == "max_visits must be an integer >= 1"
            ):
                return invalid_max_visits_response()
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
        ``/process`` (including ``strip_provenance``).
        """
        try:
            loaded = await load_parsed_process_request(
                request, server_config, log_label="process_unit"
            )
            if isinstance(loaded, JSONResponse):
                return loaded

            resolved_tenant, resolved_project = await apply_request_tenancy(
                request,
                tools,
                active_tenant=active_tenant,
                active_project=active_project,
                initialize_vector_store=(
                    loaded.ontology_context_mode_value
                    == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
                ),
            )

            try:
                validate_ontology_context_mode(
                    loaded.ontology_context_mode_value, tools
                )
            except OntologyContextConfigError as e:
                return ontology_context_config_error_response(e)

            initial_state = build_agent_state_from_parsed(
                loaded,
                server_config=server_config,
                resolved_tenant=resolved_tenant,
                resolved_project=resolved_project,
                max_chunks=1,
            )

            try:
                onto_result, facts_result = await run_unit_pipeline(
                    initial_state, tools
                )
            except DocumentConversionError as exc:
                return JSONResponse(
                    status_code=422,
                    content=ProcessErrorResponse(
                        error=str(exc),
                        error_type="ConversionError",
                        error_details={"stage": exc.stage},
                    ).model_dump(),
                )
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
                    out_graph = (
                        TripleStoreManager.strip_provenance(delta_graph)
                        if loaded.strip_provenance
                        else delta_graph
                    )
                    ontology_artifacts = [
                        {
                            "iri": onto_result.assembly_anchor_iri or "",
                            "ontology_id": None,
                            "title": "Unit ontology artifact",
                            "triples": len(out_graph),
                            "ttl": out_graph.serialize_canonical_turtle(),
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
                facts_ttl = turtle_from_graph(
                    postprocessed_facts,
                    strip_provenance=loaded.strip_provenance,
                )

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
            if (
                isinstance(e, ValueError)
                and str(e) == "max_visits must be an integer >= 1"
            ):
                return invalid_max_visits_response()
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

    if (
        config.server.ontology_context_mode == OntologyContextMode.FIXED_SINGLE_ONTOLOGY
        and not config.server.ontology_context_fixed_ontology_id.strip()
    ):
        raise ValueError(
            "ontology_context_mode=fixed_single_ontology requires "
            "ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID in the environment (or server "
            "config field ontology_context_fixed_ontology_id)."
        )

    # Create ToolBox with config
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
                suffixes=get_supported_input_extensions(tools),
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
