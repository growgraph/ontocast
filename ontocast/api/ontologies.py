"""Ontology upload, replace, and delete routes."""

from io import BytesIO
from urllib.parse import unquote

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ontocast.api.schemas import OntologyDeleteResponse, OntologyMutationResponse
from ontocast.api.tenancy_resolution import apply_request_tenancy
from ontocast.config import ServerConfig
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.toolbox import ToolBox


def build_ontology_router(
    tools: ToolBox,
    *,
    active_tenant: str,
    active_project: str,
    server_config: ServerConfig,
) -> APIRouter:
    router = APIRouter(prefix="/ontologies", tags=["ontologies"])

    init_vec = (
        server_config.ontology_context_mode
        == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
    )

    async def apply_ontology_tenancy(request: Request) -> None:
        await apply_request_tenancy(
            request,
            tools,
            active_tenant=active_tenant,
            active_project=active_project,
            initialize_vector_store=init_vec,
        )

    @router.post(
        "",
        response_model=OntologyMutationResponse,
        summary="Upload an ontology (Turtle)",
    )
    async def upload_ontology(
        request: Request,
        file: UploadFile = File(...),
    ) -> OntologyMutationResponse:
        await apply_ontology_tenancy(request)
        ttl = await file.read()
        try:
            o = await tools.ingest_ontology_ttl(ttl, filename=file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return OntologyMutationResponse(
            iri=o.iri,
            ontology_id=o.ontology_id,
            version=o.version,
            hash=o.hash,
        )

    @router.put(
        "/{ontology_iri:path}",
        response_model=OntologyMutationResponse,
        summary="Replace an ontology by IRI (path segment, URL-encoded)",
    )
    async def replace_ontology(
        request: Request,
        ontology_iri: str,
        file: UploadFile = File(...),
    ) -> OntologyMutationResponse:
        await apply_ontology_tenancy(request)
        expected = unquote(ontology_iri)
        ttl = await file.read()
        try:
            graph = RDFGraph()
            graph.parse(BytesIO(ttl), format="turtle")
            parsed = Ontology(graph=graph)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Turtle: {e}") from e
        if not parsed.iri or parsed.iri != expected:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Ontology IRI {parsed.iri!r} does not match path {expected!r}"
                ),
            )
        try:
            await tools.delete_ontology_by_iri(expected)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            o = await tools.ingest_ontology_ttl(ttl, filename=file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return OntologyMutationResponse(
            iri=o.iri,
            ontology_id=o.ontology_id,
            version=o.version,
            hash=o.hash,
        )

    @router.delete(
        "/{ontology_iri:path}",
        response_model=OntologyDeleteResponse,
        summary="Remove an ontology by IRI",
    )
    async def delete_ontology_route(
        request: Request,
        ontology_iri: str,
    ) -> OntologyDeleteResponse:
        await apply_ontology_tenancy(request)
        iri = unquote(ontology_iri)
        try:
            await tools.delete_ontology_by_iri(iri)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return OntologyDeleteResponse(iri=iri)

    return router
