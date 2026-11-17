"""Cross-document identity routes: which stored entities are one."""

from fastapi import APIRouter, HTTPException, Query, Request

from ontocast.api.tenancy_resolution import make_scoped_toolbox_resolver
from ontocast.config import ServerConfig
from ontocast.link import EntityCluster, cluster_of
from ontocast.toolbox import ToolBox


def build_entities_router(
    tools: ToolBox,
    *,
    active_tenant: str,
    active_project: str,
    server_config: ServerConfig,
) -> APIRouter:
    router = APIRouter(prefix="/entities", tags=["entities"])

    apply_tenancy = make_scoped_toolbox_resolver(
        tools,
        active_tenant=active_tenant,
        active_project=active_project,
        server_config=server_config,
    )

    @router.get(
        "/cluster",
        response_model=EntityCluster,
        summary="The stored entities that are one with an entity, across documents",
    )
    async def entity_cluster(
        request: Request,
        iri: str = Query(..., description="IRI of an entity in a stored document."),
    ) -> EntityCluster:
        scoped = await apply_tenancy(request)
        store = scoped.triple_store_manager
        if store is None or not store.supports_sparql_select():
            raise HTTPException(
                status_code=501, detail="Clusters need a SPARQL triple store"
            )
        cluster = await cluster_of(
            store, scoped.ontology_manager.identity_declarations(), iri
        )
        if cluster is None:
            raise HTTPException(
                status_code=404, detail="No stored document holds this entity"
            )
        return cluster

    return router
