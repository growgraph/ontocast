"""``ontocast config`` commands for auditing configuration files."""

from __future__ import annotations

from pathlib import Path

import click

from ontocast.cli.env_files import read_env_file
from ontocast.config.env_audit import Verdict, audit_assignments


@click.group("config")
def config() -> None:
    """Audit OntoCast configuration files."""


@config.command("check")
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Report only unknown and invalid variables.",
)
def check(files: tuple[Path, ...], quiet: bool) -> None:
    """Classify each variable in FILES against the current defaults.

    \b
    unknown    no setting reads it (renamed, removed, or not OntoCast's)
    invalid    the setting rejects the value
    redundant  equals the current default; the line can be deleted
    pinned     overrides the current default

    Each file is judged on its own, against the code defaults, never the
    ambient environment. Only variable names are printed. Exits 1 when any
    file has an unknown or invalid variable.
    """
    failed = False
    for path in files:
        report = audit_assignments(read_env_file(path))
        failed = failed or not report.ok
        click.echo(str(path))
        for finding in report.findings:
            if quiet and finding.verdict in (Verdict.REDUNDANT, Verdict.PINNED):
                continue
            line = f"  {finding.verdict.value:<10} {finding.name}"
            if finding.detail:
                line += f": {finding.detail}"
            click.echo(line)
        for error in report.config_errors:
            click.echo(f"  {'config':<10} {error}")
    if failed:
        raise SystemExit(1)
