"""``ontocast link`` commands: keep the resolution graph in step with the documents.

The resolution graph holds one identity links graph per document graph
(``<graph>/identity-links``). ``build`` recomputes all of them, ``add``
updates the given documents' links, and ``remove`` drops them. None of these
makes a model call, though the toolbox they start still needs the LLM
settings ``ontocast serve`` needs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import click

from ontocast.link import ResolutionReport, add_documents, build, remove_documents
from ontocast.toolbox import ToolBox

_tenant_option = click.option("--tenant", type=str, default=None, help="Tenant id.")
_project_option = click.option("--project", type=str, default=None, help="Project id.")
_graph_option = click.option(
    "--graph",
    "graphs",
    multiple=True,
    required=True,
    help="A document's facts graph IRI; repeat for several.",
)


@click.group("link")
def link() -> None:
    """Cross-document identity: build, update or trim the resolution graph."""


def _run(
    tenant: str | None,
    project: str | None,
    operation: Callable[[ToolBox, str, frozenset[str]], Awaitable[ResolutionReport]],
) -> ResolutionReport:
    from ontocast.cli import server as server_module

    async def _go() -> ResolutionReport:
        runtime = await server_module._abootstrap_tools(
            tenant=tenant,
            project=project,
            wipe_vector_store=None,
        )
        tools = runtime.tools
        store = tools.triple_store_manager
        if store is None or not store.supports_sparql_select():
            raise click.ClickException(
                "the resolution graph needs a SPARQL triple store"
            )
        return await operation(
            tools,
            runtime.config.tool_config.domain.current_domain,
            tools.get_atomic_tools().value_classes(),
        )

    return asyncio.run(_go())


def _echo(report: ResolutionReport) -> None:
    click.echo(
        f"{report.documents} document(s) linked, {report.links} link(s), "
        f"{report.removed} links graph(s) dropped"
    )


@link.command("build")
@_tenant_option
@_project_option
def build_command(tenant: str | None, project: str | None) -> None:
    """Recompute every document's links; drop links of documents that are gone."""

    async def operation(tools: ToolBox, domain: str, values: frozenset[str]):
        assert tools.triple_store_manager is not None
        return await build(
            tools.triple_store_manager,
            tools.ontology_manager.identity_declarations(),
            domain=domain,
            value_classes=values,
        )

    _echo(_run(tenant, project, operation))


@link.command("add")
@_graph_option
@_tenant_option
@_project_option
def add_command(
    graphs: tuple[str, ...], tenant: str | None, project: str | None
) -> None:
    """Write the links of the given documents (an update of the resolution graph)."""

    async def operation(tools: ToolBox, domain: str, values: frozenset[str]):
        assert tools.triple_store_manager is not None
        return await add_documents(
            tools.triple_store_manager,
            graphs,
            tools.ontology_manager.identity_declarations(),
            domain=domain,
            value_classes=values,
        )

    _echo(_run(tenant, project, operation))


@link.command("remove")
@_graph_option
@_tenant_option
@_project_option
def remove_command(
    graphs: tuple[str, ...], tenant: str | None, project: str | None
) -> None:
    """Drop the links of the given documents from the resolution graph."""

    async def operation(tools: ToolBox, domain: str, values: frozenset[str]):
        assert tools.triple_store_manager is not None
        return await remove_documents(tools.triple_store_manager, graphs)

    _echo(_run(tenant, project, operation))
