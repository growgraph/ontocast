"""``ontocast check``: model-free check of a facts graph against a catalog."""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import Iterable
from typing import NoReturn

import click

from ontocast.check import FactsCheckReport, check_facts
from ontocast.onto.rdfgraph import RDFGraph

_RDF_SUFFIXES = frozenset(RDFGraph.SUFFIX_FORMATS)


def _fail(message: str) -> NoReturn:
    """Print an error and exit with code 2: the check could not run."""
    click.echo(f"error: {message}", err=True)
    raise SystemExit(2)


def _load(paths: Iterable[pathlib.Path]) -> RDFGraph:
    """Merge RDF files into one graph; an unreadable file exits with code 2."""
    merged = RDFGraph()
    for path in paths:
        if path.suffix.lower() not in _RDF_SUFFIXES:
            _fail(
                f"cannot read {path}: unknown RDF file suffix {path.suffix!r}; "
                f"use one of {', '.join(sorted(_RDF_SUFFIXES))}"
            )
        try:
            merged += RDFGraph.from_file(path)
        except Exception as exc:  # parsers raise many exception types
            _fail(f"cannot read {path}: {exc}")
    return merged


def _dir_files(directory: pathlib.Path | None) -> list[pathlib.Path]:
    if directory is None:
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _RDF_SUFFIXES
    )


def _catalog_from_tenant(
    tenant: str | None, project: str | None
) -> tuple[RDFGraph, RDFGraph | None]:
    """The catalog and shapes the store holds; a failed start exits with code 2."""
    from ontocast.cli import server as server_module

    async def _go() -> tuple[RDFGraph, RDFGraph | None]:
        runtime = await server_module._abootstrap_tools(
            tenant=tenant,
            project=project,
            wipe_vector_store=False,
            ontology_dir="",
            shapes_dir="",
            summarize_ontologies=False,
        )
        try:
            return await runtime.tools.catalog_graphs()
        finally:
            await runtime.tools.aclose()

    try:
        return asyncio.run(_go())
    except Exception as exc:  # store, settings and tenancy failures alike
        _fail(f"cannot load the catalog of the tenant and project: {exc}")


def _print(report: FactsCheckReport) -> None:
    totals = {"error": 0, "warning": 0}
    for by_severity in report.counts.values():
        for severity, n in by_severity.items():
            totals[severity] = totals.get(severity, 0) + n
    metrics = report.metrics
    click.echo(
        f"{metrics.triples} triples, {metrics.instances} instances: "
        f"{totals['error']} error(s), {totals['warning']} warning(s)"
    )
    if report.catalog_empty:
        click.echo("note: the catalog is empty; catalog checks were skipped")
    for kind, by_severity in sorted(report.counts.items()):
        parts = ", ".join(f"{n} {sev}" for sev, n in sorted(by_severity.items()))
        click.echo(f"{kind}: {parts}")
    for f in report.findings:
        where = f" [{f.subject}]" if f.subject else ""
        click.echo(f"{f.severity} {f.kind.value}{where}: {f.message}")
    hidden = report.findings_total - len(report.findings)
    if hidden > 0:
        click.echo(f"... {hidden} more finding(s) not shown")


@click.command("check")
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--ontology-dir",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    default=None,
    help="Directory of catalog ontology files (all RDF files in it).",
)
@click.option(
    "--shapes-dir",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    default=None,
    help="Directory of SHACL shapes files.",
)
@click.option("--tenant", type=str, default=None, help="Tenant id.")
@click.option("--project", type=str, default=None, help="Project id.")
@click.option(
    "--vocabulary",
    type=click.Choice(["strict", "open"]),
    default="strict",
    show_default=True,
    help="strict: a term outside the catalog is an error; open: a warning.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the report as JSON.")
def check(
    files: tuple[pathlib.Path, ...],
    ontology_dir: pathlib.Path | None,
    shapes_dir: pathlib.Path | None,
    tenant: str | None,
    project: str | None,
    vocabulary: str,
    as_json: bool,
) -> None:
    """Check FILES (merged into one facts graph) without a model call.

    Exit status: 0 without error findings, 1 with any, 2 when the check cannot
    run (a file cannot be read or holds no triples, or the catalog cannot be
    loaded). Without --tenant/--project the catalog is read from
    --ontology-dir and --shapes-dir; either may be omitted.
    """
    from_store = tenant is not None or project is not None
    if from_store and (ontology_dir is not None or shapes_dir is not None):
        raise click.UsageError(
            "--tenant/--project read the catalog from the store; "
            "do not combine them with --ontology-dir or --shapes-dir."
        )
    graph = _load(files)
    if not len(graph):
        _fail("the files hold no triples")
    if from_store:
        catalog, shapes = _catalog_from_tenant(tenant, project)
    else:
        catalog = _load(_dir_files(ontology_dir))
        shapes_graph = _load(_dir_files(shapes_dir))
        shapes = shapes_graph if len(shapes_graph) else None
    report = check_facts(
        graph,
        catalog,
        shapes,
        vocabulary="open" if vocabulary == "open" else "strict",
    )
    if as_json:
        click.echo(report.model_dump_json(indent=2))
    else:
        _print(report)
    errors = sum(by.get("error", 0) for by in report.counts.values())
    raise SystemExit(1 if errors else 0)
