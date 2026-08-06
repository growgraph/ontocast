"""Shared HTTP request tenancy resolution for API routes."""

from starlette.requests import Request

from ontocast.onto.enum import OntologyContextMode
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
) -> tuple[ToolBox, str, str]:
    """Resolve tenant/project and return the ToolBox that serves that partition.

    If ``tenant`` or ``project`` appears in the query string, resolve with
    defaults and hand back a ToolBox bound to that scope; otherwise return the
    startup ToolBox and the ``active_*`` scope unchanged.

    This used to retarget the *shared* ToolBox in place, so every request
    mutated process-wide state -- dataset names, collection names, the ontology
    catalog -- behind a single lock that serialized all multi-tenant traffic.
    Now each scope owns a ToolBox (over a deep-copied ``Config``, sharing the
    expensive runtime), so isolation is structural and different tenants run
    concurrently.

    Returns:
        ``(tools, tenant, project)`` -- **use the returned ToolBox**, not the one
        passed in, for everything downstream of this call.
    """
    if not request_has_tenancy_query_params(request):
        return tools, active_tenant, active_project
    request_tenant = request.query_params.get("tenant", None)
    request_project = request.query_params.get("project", None)
    resolved_tenant, resolved_project = resolve_tenant_project(
        request_tenant, request_project
    )
    if not stores_use_tenancy_partitions(tools):
        return tools, resolved_tenant, resolved_project

    scoped = await tools.for_scope(
        resolved_tenant,
        resolved_project,
        # `initialize_vector_store` already encodes "this request runs in the
        # vector-search context mode"; pass the mode itself, since that is what
        # ToolBox.should_initialize_vector_store checks.
        ontology_context_mode=(
            OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
            if initialize_vector_store
            else None
        ),
        fail_on_vector_store_error=False,
    )
    return scoped, resolved_tenant, resolved_project
