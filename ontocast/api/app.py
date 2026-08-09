"""FastAPI application factory and route handlers."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from starlette.exceptions import HTTPException as StarletteHTTPException

from ontocast._version import __version__
from ontocast.api.match_models import (
    AlignEntitiesRequest,
    AlignEntitiesResponse,
    DeriveMatchesRequest,
    DeriveMatchesResponse,
    EvaluateMatchRequest,
    EvaluateMatchResponse,
)
from ontocast.api.ontologies import build_ontology_router
from ontocast.api.parse import RequestParamError
from ontocast.api.process_helpers import (
    calculate_recursion_limit,
    get_supported_input_extensions,
    select_unit_facts_ontology_graph,
    turtle_from_graph,
    validate_unit_pipeline_facts,
)
from ontocast.api.process_request import (
    build_agent_state_from_parsed,
    load_parsed_process_request,
)
from ontocast.api.responses import (
    document_conversion_error_response,
    ontology_context_config_error_response,
    request_param_error_response,
)
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
from ontocast.api.tenancy_resolution import apply_request_tenancy
from ontocast.config import ServerConfig
from ontocast.onto.enum import OntologyContextMode, RenderMode, Status
from ontocast.onto.retrieval_capabilities import (
    OntologyContextConfigError,
    validate_ontology_context_mode,
)
from ontocast.onto.tenancy import DEFAULT_PROJECT, DEFAULT_TENANT
from ontocast.stategraph import create_agent_graph
from ontocast.stategraph.helpers import build_ontology_delta_graph
from ontocast.stategraph.unit_pipeline import DocumentConversionError, run_unit_pipeline
from ontocast.tool.agg.match_derivation import derive_pair_matches
from ontocast.tool.agg.match_models import TaggedGraph
from ontocast.tool.agg.triple_evaluator import TripleSetEvaluator
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


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

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Release backend connections when the server stops.

        ``ToolBox.aclose`` also closes every per-tenant ToolBox spawned through
        ``for_scope``, so this covers the whole registry.
        """
        yield
        await tools.aclose()

    app = FastAPI(title="ontocast", version=__version__, lifespan=lifespan)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException):
        """Render every HTTPException in the same shape as the other routes.

        The ``/ontologies`` routes raise ``HTTPException``, which FastAPI
        renders as ``{"detail": ...}`` -- a third error shape alongside
        ``StatusErrorBody`` and ``ProcessErrorResponse``, so a client could not
        write one error handler. Normalizing here covers every route, including
        framework-generated 404s and 405s.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content=StatusErrorBody(
                error=str(exc.detail),
                error_type="HTTPError",
            ).model_dump(),
            headers=getattr(exc, "headers", None),
        )

    app.include_router(
        build_ontology_router(
            tools,
            active_tenant=active_tenant,
            active_project=active_project,
            server_config=server_config,
        )
    )

    workflow: CompiledStateGraph = create_agent_graph(tools, name="ontocast")

    def workflow_for(scoped: ToolBox) -> CompiledStateGraph:
        """Return the compiled graph bound to ``scoped``.

        Nodes are ``partial(fn, tools=tools)`` and ``make_*_node(tools)``
        closures, so a graph belongs to exactly one ToolBox and a scoped one
        needs its own. Compilation is in-memory topology work with no I/O, so
        caching it per scope costs far less than the ToolBox it belongs to.
        """
        if scoped is tools:
            return workflow
        scope = scoped.scope
        if scope is None:
            return create_agent_graph(scoped, name="ontocast")
        return tools.ensure_tenancy_registry().graph_for(
            scope, lambda: create_agent_graph(scoped, name="ontocast")
        )

    process_semaphore: asyncio.Semaphore | None = None
    if server_config.max_concurrent_processes is not None:
        process_semaphore = asyncio.Semaphore(server_config.max_concurrent_processes)

    @app.get(
        "/health",
        response_model=HealthOkResponse,
        responses={503: {"model": HealthErrorResponse}},
        summary="Liveness probe",
    )
    async def health_check():
        """Report whether the LLM tool was constructed.

        This is a liveness signal, not a readiness one: it does not reach the
        LLM provider, the triple store, or the vector store.
        """
        try:
            if tools.llm is None:
                return JSONResponse(
                    status_code=503,
                    content=HealthErrorResponse(
                        error="LLM not initialized"
                    ).model_dump(),
                )
            return HealthOkResponse(
                llm_provider=tools.llm_provider, version=__version__
            )
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return JSONResponse(
                status_code=503,
                content=HealthErrorResponse(error=str(e)).model_dump(),
            )

    @app.get("/info", response_model=InfoResponse, summary="Server capabilities")
    async def info():
        llm_cache = None
        if tools.llm is not None:
            # Async variant: the disk stats walk every cache file, which is tens
            # of thousands of stat() calls on a warm cache and must not run on
            # the event loop.
            llm_cache = await tools.llm.aget_cache_stats()
        return InfoResponse(
            version=__version__,
            llm_cache=llm_cache,
            max_concurrent_processes=server_config.max_concurrent_processes,
            # Computed, not hardcoded: without the doc-processing extra the
            # server cannot accept PDFs, and advertising them anyway made
            # /info unusable for capability probing.
            input_types=sorted(
                ext.lstrip(".") for ext in get_supported_input_extensions(tools)
            ),
        )

    @app.post("/match/entities", response_model=AlignEntitiesResponse)
    async def align_entities(request: AlignEntitiesRequest):
        try:
            aligner = tools.get_entity_aligner(
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

    @app.post(
        "/flush",
        response_model=FlushOkResponse,
        responses={400: {"model": StatusErrorBody}, 500: {"model": StatusErrorBody}},
        summary="Delete stored facts and ontologies for a tenancy scope",
    )
    async def flush(
        tenant: str | None = Query(
            default=None,
            description="Tenancy partition to flush. Defaults to the server's active tenant.",
        ),
        project: str | None = Query(
            default=None,
            description="Project partition to flush. Defaults to the server's active project.",
        ),
    ):
        """Destructive: drops the target partition's facts, ontologies, and vectors."""
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

    @app.post(
        "/process",
        response_model=ProcessOkResponse,
        responses={
            400: {"model": StatusErrorBody},
            409: {"model": StatusErrorBody},
            422: {"model": StatusErrorBody},
            500: {"model": ProcessErrorResponse},
        },
        summary="Extract ontology and facts from a document",
    )
    async def process(request: Request):
        """Run the full chunked pipeline over an uploaded document.

        Accepts multipart form data (``file=@doc.pdf``) or a JSON body. Request
        parameters are documented in the API user guide; they are read from the
        query string, form fields, or JSON body interchangeably.
        """
        workflow_state: dict | None = None
        if process_semaphore is not None:
            await process_semaphore.acquire()
        try:
            loaded = await load_parsed_process_request(
                request, server_config, log_label="process"
            )
            if isinstance(loaded, JSONResponse):
                return loaded

            # Use `scoped_tools` from here on: it is bound to this request's
            # tenant/project partition, and may not be the startup ToolBox.
            (
                scoped_tools,
                resolved_tenant,
                resolved_project,
            ) = await apply_request_tenancy(
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
                    loaded.ontology_context_mode_value, scoped_tools
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

            async for chunk in workflow_for(scoped_tools).astream(
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

            unit_failures = [
                failure.model_dump(mode="json")
                for failure in workflow_state.get("unit_failures", [])
            ]
            facts_repairs = {
                unit_index: [record.model_dump(mode="json") for record in records]
                for unit_index, records in workflow_state.get(
                    "facts_repairs_applied", {}
                ).items()
            }
            validation_findings = [
                finding.model_dump(mode="json")
                for finding in workflow_state.get("facts_validation_findings", [])
            ]
            gate_repairs = [
                record.model_dump(mode="json")
                for record in workflow_state.get("facts_gate_repairs", [])
            ]

            if workflow_state["status"] == Status.FAILED:
                # Every unit failed, or conversion did. Returning 200 here made
                # a total failure look identical to a document with nothing to
                # extract.
                return JSONResponse(
                    status_code=422,
                    content=ProcessErrorResponse(
                        error="Extraction produced no output for any content unit",
                        error_type="PipelineError",
                        error_code="no_units_extracted",
                        error_details={
                            "stage": workflow_state.get("failure_stage"),
                            "reason": workflow_state.get("failure_reason"),
                            "unit_failures": unit_failures,
                        },
                    ).model_dump(),
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
                    facts_repairs=facts_repairs,
                    failed_units=unit_failures,
                    improvement_suggestions=list(
                        workflow_state.get("improvements_suggestions", [])
                    ),
                    facts_conformance=dict(
                        workflow_state.get("facts_conformance", {}) or {}
                    ),
                    facts_validation_findings=validation_findings,
                    facts_gate_repairs=gate_repairs,
                ),
            )

        except RequestParamError as e:
            # Malformed input is the client's error, not ours.
            return request_param_error_response(e)
        except DocumentConversionError as e:
            return document_conversion_error_response(e, e.stage)
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
        finally:
            if process_semaphore is not None:
                process_semaphore.release()

    @app.post(
        "/process_unit",
        response_model=ProcessOkResponse,
        responses={
            400: {"model": StatusErrorBody},
            409: {"model": StatusErrorBody},
            422: {"model": StatusErrorBody},
            500: {"model": ProcessErrorResponse},
        },
        summary="Extract from a single small document without chunking",
    )
    async def process_unit(request: Request):
        """Process the whole input as one content unit.

        Skips chunking, section tagging, summarization, and normalization, so
        ``max_chunks`` and the section-selection parameters have no effect
        here. The post-aggregation validation gate (invariant findings, SHACL,
        LLM-free autofix) does run, minus the un-merge repair, which is
        meaningless for a single unit. Use ``/process`` for anything larger
        than a single passage.
        """
        if process_semaphore is not None:
            await process_semaphore.acquire()
        try:
            loaded = await load_parsed_process_request(
                request, server_config, log_label="process_unit"
            )
            if isinstance(loaded, JSONResponse):
                return loaded

            # Use `scoped_tools` from here on: it is bound to this request's
            # tenant/project partition, and may not be the startup ToolBox.
            (
                scoped_tools,
                resolved_tenant,
                resolved_project,
            ) = await apply_request_tenancy(
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
                    loaded.ontology_context_mode_value, scoped_tools
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
                    initial_state, scoped_tools
                )
            except DocumentConversionError as exc:
                return document_conversion_error_response(exc, exc.stage)
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
                # Single-unit responses expose the insert complement; deletes
                # are catalog-apply concerns and this path never writes the
                # catalog.
                delta_graph = build_ontology_delta_graph(onto_result).inserts
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
                ontology_graph = select_unit_facts_ontology_graph(
                    onto_result, facts_result
                )
                document_metadata = dict(initial_state.document_metadata)
                if (
                    initial_state.source_url
                    and "source_url" not in document_metadata
                    and "source_uri" not in document_metadata
                ):
                    document_metadata["source_url"] = initial_state.source_url
                postprocessed_facts = scoped_tools.aggregator.postprocess_facts_units(
                    units=[facts_result.content_unit],
                    ontology_graph=ontology_graph,
                    doc_iri=initial_state.doc_iri,
                    document_metadata=document_metadata,
                    doc_namespace=initial_state.doc_namespace,
                )
                # Same gate the document pipeline reaches at VALIDATE_FACTS:
                # invariant findings, SHACL, and the LLM-free autofix, so the
                # served graph and conformance report match the CLI unit path.
                initial_state.aggregated_facts = postprocessed_facts.graph
                await asyncio.to_thread(
                    validate_unit_pipeline_facts,
                    initial_state,
                    ontology_graph,
                    scoped_tools,
                )
                facts_ttl = turtle_from_graph(
                    initial_state.aggregated_facts,
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
                    facts_conformance=dict(initial_state.facts_conformance or {}),
                    facts_validation_findings=[
                        finding.model_dump(mode="json")
                        for finding in initial_state.facts_validation_findings
                    ],
                    facts_gate_repairs=[
                        record.model_dump(mode="json")
                        for record in initial_state.facts_gate_repairs
                    ],
                ),
            )

        except RequestParamError as e:
            return request_param_error_response(e)
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
        finally:
            if process_semaphore is not None:
                process_semaphore.release()

    return app
