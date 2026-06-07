"""Shared HTTP request tenancy resolution for API routes."""

from starlette.requests import Request

from ontocast.onto.tenancy import DEFAULT_PROJECT, DEFAULT_TENANT
from ontocast.toolbox import ToolBox


def stores_use_tenancy_partitions(tools: ToolBox) -> bool:
    """True when triple store and/or Qdrant should be retargeted for tenant/project."""
    if tools.vector_store is not None:
        return True
    triple = tools.triple_store_manager
    if triple is None:
        return False
    supports = getattr(triple, "supports_tenancy_partition", None)
    if supports is None:
        return False
    return supports()


def resolve_tenant_project(tenant: str | None, project: str | None) -> tuple[str, str]:
    t = (tenant or DEFAULT_TENANT).strip()
    p = (project or DEFAULT_PROJECT).strip()
    if not t or not p:
        raise ValueError("tenant and project must be non-empty after resolution")
    return t, p


def request_has_tenancy_query_params(request: Request) -> bool:
    return "tenant" in request.query_params or "project" in request.query_params


async def apply_request_tenancy(
    request: Request,
    tools: ToolBox,
    *,
    active_tenant: str,
    active_project: str,
    initialize_vector_store: bool,
) -> tuple[str, str]:
    """Resolve tenant/project and retarget partitioned stores when the client set QS.

    Mirrors ``/process``: if ``tenant`` or ``project`` appears in the query string,
    resolve with defaults and call :meth:`ToolBox.update_tenancy_with_vector_mode`
    when Fuseki/Qdrant partitions are in use. Otherwise return ``active_*`` from
    server startup without retargeting.
    """
    if not request_has_tenancy_query_params(request):
        return active_tenant, active_project
    request_tenant = request.query_params.get("tenant", None)
    request_project = request.query_params.get("project", None)
    resolved_tenant, resolved_project = resolve_tenant_project(
        request_tenant, request_project
    )
    if stores_use_tenancy_partitions(tools):
        await tools.update_tenancy_with_vector_mode(
            resolved_tenant,
            resolved_project,
            initialize_vector_store=initialize_vector_store,
            fail_on_vector_store_error=False,
        )
    return resolved_tenant, resolved_project
