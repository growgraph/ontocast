"""Ontology upload, replace, and delete routes."""

from io import BytesIO
from urllib.parse import unquote

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ontocast.api.schemas import OntologyDeleteResponse, OntologyMutationResponse
from ontocast.api.tenancy_resolution import make_scoped_toolbox_resolver
from ontocast.config import ServerConfig
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

    apply_ontology_tenancy = make_scoped_toolbox_resolver(
        tools,
        active_tenant=active_tenant,
        active_project=active_project,
        server_config=server_config,
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
        scoped = await apply_ontology_tenancy(request)
        ttl = await file.read()
        try:
            o = await scoped.ingest_ontology_ttl(ttl, filename=file.filename)
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
        scoped = await apply_ontology_tenancy(request)
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
            await scoped.delete_ontology_by_iri(expected)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            o = await scoped.ingest_ontology_ttl(ttl, filename=file.filename)
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
        scoped = await apply_ontology_tenancy(request)
        iri = unquote(ontology_iri)
        try:
            await scoped.delete_ontology_by_iri(iri)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return OntologyDeleteResponse(iri=iri)

    return router
