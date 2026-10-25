"""Ontology list, version history, upload, version ops, and delete routes."""

from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import unquote

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from ontocast.api.schemas import (
    OntologyDeleteResponse,
    OntologyDiffResponse,
    OntologyListResponse,
    OntologyMutationResponse,
    OntologySummary,
    OntologyVersionDeleteResponse,
    OntologyVersionsResponse,
    OntologyVersionSummary,
)
from ontocast.api.tenancy_resolution import make_scoped_toolbox_resolver
from ontocast.config import ServerConfig
from ontocast.onto.ontology import Ontology, OntologyHeader
from ontocast.onto.ontology_header import DivergedOntologyError, heads_by_iri
from ontocast.toolbox import ToolBox


def _version_summary(header: OntologyHeader) -> OntologyVersionSummary:
    assert header.version_iri and header.version and header.created_at
    assert header.modified_at
    return OntologyVersionSummary(
        version_iri=header.version_iri,
        version=header.version,
        prior_version_iris=list(header.prior_version_iris),
        created_at=header.created_at,
        modified_at=header.modified_at,
        removed=header.removed,
    )


def _version_summary_from_ontology(o: Ontology) -> OntologyVersionSummary:
    return _version_summary(OntologyHeader.from_ontology(o))


def _mutation_response(o: Ontology) -> OntologyMutationResponse:
    assert o.ontology_id and o.version and o.version_iri
    return OntologyMutationResponse(
        iri=o.iri,
        ontology_id=o.ontology_id,
        version=o.version,
        version_iri=o.version_iri,
    )


def _resolve_iri(scoped: ToolBox, ontology_id: str) -> str:
    iri = scoped.ontology_manager.resolve_ontology_ref(unquote(ontology_id))
    if iri is None:
        raise HTTPException(status_code=404, detail="Unknown ontology")
    return iri


def _version_iri(iri: str, key: str) -> str:
    return f"{iri}#{unquote(key)}"


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

    @router.get(
        "",
        response_model=OntologyListResponse,
        summary="List every catalog ontology with its head version(s)",
    )
    async def list_ontologies(request: Request) -> OntologyListResponse:
        scoped = await apply_ontology_tenancy(request)
        headers = await scoped.ontology_manager.aget_catalog_headers()
        versions_by_iri: dict[str, list[OntologyHeader]] = defaultdict(list)
        for header in headers:
            versions_by_iri[header.iri].append(header)
        heads = heads_by_iri(headers)
        summaries: list[OntologySummary] = []
        for iri in sorted(versions_by_iri):
            group = heads.get(iri, [])
            newest = group[0] if group else versions_by_iri[iri][0]
            assert newest.ontology_id and newest.title
            summaries.append(
                OntologySummary(
                    iri=iri,
                    ontology_id=newest.ontology_id,
                    title=newest.title,
                    description=newest.description,
                    versions=len(versions_by_iri[iri]),
                    heads=[_version_summary(h) for h in group],
                    diverged=len(group) > 1,
                )
            )
        return OntologyListResponse(ontologies=summaries)

    @router.get(
        "/{ontology_id}/versions",
        response_model=OntologyVersionsResponse,
        summary="The version history of one ontology, newest first",
    )
    async def list_versions(
        request: Request, ontology_id: str
    ) -> OntologyVersionsResponse:
        scoped = await apply_ontology_tenancy(request)
        iri = _resolve_iri(scoped, ontology_id)
        headers = [
            h
            for h in await scoped.ontology_manager.aget_catalog_headers()
            if h.iri == iri
        ]
        headers.sort(
            key=lambda h: h.modified_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return OntologyVersionsResponse(
            iri=iri,
            ontology_id=(headers[0].ontology_id if headers else None)
            or unquote(ontology_id),
            versions=[_version_summary(h) for h in headers],
        )

    @router.get(
        "/{ontology_id}/versions/{key}",
        summary="One version as Turtle (a tombstone is header-only)",
        response_class=Response,
    )
    async def get_version(request: Request, ontology_id: str, key: str) -> Response:
        scoped = await apply_ontology_tenancy(request)
        iri = _resolve_iri(scoped, ontology_id)
        version_iri = _version_iri(iri, key)
        graph = await scoped.aget_version_graph(version_iri)
        if graph is None:
            raise HTTPException(status_code=404, detail="Unknown version")
        return Response(
            content=graph.serialize(format="turtle"),
            media_type="text/turtle",
        )

    @router.delete(
        "/{ontology_id}/versions/{key}",
        response_model=OntologyVersionDeleteResponse,
        summary="Tombstone one version; heads fall back through removed versions",
    )
    async def delete_version(
        request: Request, ontology_id: str, key: str
    ) -> OntologyVersionDeleteResponse:
        scoped = await apply_ontology_tenancy(request)
        iri = _resolve_iri(scoped, ontology_id)
        version_iri = _version_iri(iri, key)
        try:
            heads = await scoped.remove_ontology_version(version_iri)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return OntologyVersionDeleteResponse(
            version_iri=version_iri,
            heads=[_version_summary_from_ontology(h) for h in heads],
        )

    @router.get(
        "/{ontology_id}/diff",
        response_model=OntologyDiffResponse,
        summary="Content triples added and removed between two versions",
    )
    async def diff_versions_route(
        request: Request,
        ontology_id: str,
        from_key: str = Query(..., alias="from"),
        to_key: str = Query(..., alias="to"),
    ) -> OntologyDiffResponse:
        scoped = await apply_ontology_tenancy(request)
        iri = _resolve_iri(scoped, ontology_id)
        from_iri = _version_iri(iri, from_key)
        to_iri = _version_iri(iri, to_key)
        try:
            diff = await scoped.diff_ontology_versions(from_iri, to_iri)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return OntologyDiffResponse(
            from_iri=diff.from_iri,
            to_iri=diff.to_iri,
            added=diff.added,
            removed=diff.removed,
            added_count=diff.added_count,
            removed_count=diff.removed_count,
        )

    @router.post(
        "",
        response_model=OntologyMutationResponse,
        summary="Upload an ontology (Turtle) as the root version of a new ontology",
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
            status = 409 if "already exists" in str(e) else 400
            raise HTTPException(status_code=status, detail=str(e)) from e
        assert o is not None
        return _mutation_response(o)

    @router.put(
        "/{ontology_iri:path}",
        response_model=OntologyMutationResponse,
        summary="Upload a new version of an ontology (prior = current head)",
    )
    async def put_ontology_version(
        request: Request,
        ontology_iri: str,
        file: UploadFile = File(...),
    ) -> OntologyMutationResponse:
        scoped = await apply_ontology_tenancy(request)
        expected = unquote(ontology_iri)
        ttl = await file.read()
        try:
            o = await scoped.ingest_ontology_ttl(
                ttl, filename=file.filename, as_new_version=True
            )
        except DivergedOntologyError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if o is None:
            head = scoped.ontology_manager.get_head_by_iri(expected)
            if head is None or head.iri != expected:
                raise HTTPException(
                    status_code=400,
                    detail=f"unchanged upload for {expected!r} but no head is held",
                )
            return _mutation_response(head)
        if o.iri != expected:
            versions = scoped.ontology_manager.get_ontology_versions_by_iri(o.iri)
            if len(versions) <= 1:
                await scoped.delete_ontology_by_iri(o.iri)
            elif o.version_iri:
                await scoped.remove_ontology_version(o.version_iri)
            raise HTTPException(
                status_code=400,
                detail=f"Ontology IRI {o.iri!r} does not match path {expected!r}",
            )
        return _mutation_response(o)

    @router.delete(
        "/{ontology_iri:path}",
        response_model=OntologyDeleteResponse,
        summary="Remove an ontology by IRI, every version included",
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
