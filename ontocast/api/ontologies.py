"""Ontology upload, replace, and delete routes."""

from io import BytesIO
from urllib.parse import unquote

from fastapi import APIRouter, File, HTTPException, UploadFile

from ontocast.api.schemas import OntologyDeleteResponse, OntologyMutationResponse
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.toolbox import ToolBox


def build_ontology_router(tools: ToolBox) -> APIRouter:
    router = APIRouter(prefix="/ontologies", tags=["ontologies"])

    @router.post(
        "",
        response_model=OntologyMutationResponse,
        summary="Upload an ontology (Turtle)",
    )
    async def upload_ontology(file: UploadFile = File(...)) -> OntologyMutationResponse:
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
        ontology_iri: str,
        file: UploadFile = File(...),
    ) -> OntologyMutationResponse:
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
        ontology_iri: str,
    ) -> OntologyDeleteResponse:
        iri = unquote(ontology_iri)
        try:
            await tools.delete_ontology_by_iri(iri)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return OntologyDeleteResponse(iri=iri)

    return router
