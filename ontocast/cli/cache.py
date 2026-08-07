"""``ontocast cache`` commands for inspecting and trimming the on-disk cache.

The cache bounds itself automatically (see
:meth:`~ontocast.tool.cache.Cacher.prune`); these commands exist for the cases
automation deliberately stays out of: inspecting what is stored, forcing a trim
with a different ceiling, and clearing directories left behind by an older cache
layout.
"""

from __future__ import annotations

import click

from ontocast.config import Config
from ontocast.tool.cache import KNOWN_CACHE_SUBDIRS, Cacher, PruneReport


def _format_bytes(value: int) -> str:
    """Render a byte count in the largest unit that keeps it readable."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _build_cacher() -> Cacher:
    """Cacher pointed at the configured cache directory, with pruning inert.

    Automatic pruning is disabled here so that a read-only command such as
    ``stats`` never deletes anything as a side effect; the ``prune`` command
    passes its ceilings explicitly.
    """
    return Cacher(config=Config(), max_bytes=0, ttl_days=None)


def _echo_report(report: PruneReport) -> None:
    if not report.changed:
        click.echo("Nothing to remove.")
    else:
        click.echo(
            f"Removed {report.files_removed} entries, "
            f"freed {_format_bytes(report.bytes_reclaimed)}."
        )
    click.echo(f"Cache now {_format_bytes(report.bytes_remaining)}.")


@click.group("cache")
def cache() -> None:
    """Inspect and trim the OntoCast on-disk cache."""


@cache.command("stats")
def stats() -> None:
    """Show cache size, broken down by tool."""
    cacher = _build_cacher()
    summary = cacher.cache_stats()
    click.echo(f"Cache directory: {cacher.cache_dir}")
    click.echo(
        f"Total: {summary.total_files} entries, "
        f"{_format_bytes(summary.total_size_bytes)}"
    )
    for name in sorted(summary.subdirectories):
        entry = summary.subdirectories[name]
        marker = "" if name in KNOWN_CACHE_SUBDIRS else "  (orphaned)"
        click.echo(
            f"  {name}: {entry.files} entries, "
            f"{_format_bytes(entry.size_bytes)}{marker}"
        )


@cache.command("prune")
@click.option(
    "--max-bytes",
    type=int,
    default=None,
    help="Size ceiling in bytes. Defaults to ONTOCAST_CACHE_MAX_BYTES.",
)
@click.option(
    "--ttl-days",
    type=int,
    default=None,
    help="Drop entries unused for this many days.",
)
@click.option(
    "--orphaned",
    is_flag=True,
    help="Instead of trimming by size, remove subdirectories no current tool writes to.",
)
def prune(max_bytes: int | None, ttl_days: int | None, orphaned: bool) -> None:
    """Trim the cache by size and age, or clear orphaned subdirectories."""
    cacher = _build_cacher()
    if orphaned:
        _echo_report(cacher.prune_orphaned_subdirs(KNOWN_CACHE_SUBDIRS))
        return

    config_paths = Config().get_tool_config().path_config
    _echo_report(
        cacher.prune(
            max_bytes=(
                config_paths.cache_max_bytes if max_bytes is None else max_bytes
            ),
            ttl_days=(config_paths.cache_ttl_days if ttl_days is None else ttl_days),
        )
    )


@cache.command("clear")
@click.option(
    "--subdir",
    default=None,
    help="Clear only this tool's cache (e.g. llm). Omit to clear everything.",
)
@click.confirmation_option(prompt="Delete cached entries?")
def clear(subdir: str | None) -> None:
    """Delete cached entries."""
    cacher = _build_cacher()
    before = cacher.cache_stats().total_size_bytes
    cacher.clear(subdirectory=subdir)
    after = cacher.cache_stats().total_size_bytes
    target = subdir or "all subdirectories"
    click.echo(f"Cleared {target}; freed {_format_bytes(before - after)}.")
