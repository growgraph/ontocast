"""SHACL shapes upload, list, and delete routes.

Mirrors :mod:`ontocast.api.ontologies`, against the shapes partition rather
than the ontologies dataset. A document is addressed by the ``owl:Ontology``
IRI it declares, so uploading the same document twice replaces it.
"""

from urllib.parse import unquote

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ontocast.api.schemas import (
    ShapesDeleteResponse,
    ShapesListResponse,
    ShapesMutationResponse,
)
from ontocast.api.tenancy_resolution import make_scoped_toolbox_resolver
from ontocast.config import ServerConfig
from ontocast.toolbox import ToolBox


def build_shapes_router(
    tools: ToolBox,
    *,
    active_tenant: str,
    active_project: str,
    server_config: ServerConfig,
) -> APIRouter:
    """Build the ``/shapes`` router bound to ``tools``' tenancy registry."""
    router = APIRouter(prefix="/shapes", tags=["shapes"])

    apply_shapes_tenancy = make_scoped_toolbox_resolver(
        tools,
        active_tenant=active_tenant,
        active_project=active_project,
        server_config=server_config,
    )

    @router.get(
        "",
        response_model=ShapesListResponse,
        summary="List stored SHACL shapes documents",
    )
    async def list_shapes(request: Request) -> ShapesListResponse:
        scoped = await apply_shapes_tenancy(request)
        graph = scoped.shapes_catalog.graph()
        return ShapesListResponse(
            graph_uris=await scoped.shapes_catalog.list_graph_uris(),
            triples=len(graph) if graph is not None else 0,
        )

    @router.post(
        "",
        response_model=ShapesMutationResponse,
        summary="Upload a SHACL shapes document (Turtle)",
    )
    async def upload_shapes(
        request: Request,
        file: UploadFile = File(...),
    ) -> ShapesMutationResponse:
        scoped = await apply_shapes_tenancy(request)
        ttl = await file.read()
        try:
            graph_uri = await scoped.ingest_shapes_ttl(ttl, filename=file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        graph = scoped.shapes_catalog.graph()
        return ShapesMutationResponse(
            graph_uri=graph_uri,
            triples=len(graph) if graph is not None else 0,
        )

    @router.delete(
        "/{graph_uri:path}",
        response_model=ShapesDeleteResponse,
        summary="Remove a SHACL shapes document by graph URI",
    )
    async def delete_shapes(
        request: Request,
        graph_uri: str,
    ) -> ShapesDeleteResponse:
        scoped = await apply_shapes_tenancy(request)
        target = unquote(graph_uri)
        try:
            await scoped.delete_shapes_by_uri(target)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return ShapesDeleteResponse(graph_uri=target)

    return router
