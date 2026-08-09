"""Console-script entry points with a helpful error on a base install.

The CLI and HTTP server need click, uvicorn, FastAPI and the document stack,
none of which ship in the light core. Without this shim a base install fails
with a bare ``ModuleNotFoundError: No module named 'click'``, which says
nothing about the extra that fixes it.

Every console script in ``pyproject.toml`` routes through :func:`_run`.
"""

from __future__ import annotations

import sys
from typing import Any, NoReturn

_HINT = """\
The `{script}` command needs OntoCast's server and CLI dependencies, which are
not part of the base install.

    pip install "ontocast[server]"

The base install is deliberately light so that OntoCast can be embedded in
another application. See the "Using OntoCast from your own agent" guide for
what each extra adds.

(missing module: {missing})\
"""


def _fail(script: str, missing: str) -> NoReturn:
    print(_HINT.format(script=script, missing=missing), file=sys.stderr)
    raise SystemExit(1)


def _run(script: str, module: str, attribute: str) -> Any:
    """Import and call a CLI entry point, or explain which extra is missing."""
    import importlib

    try:
        target = getattr(importlib.import_module(module), attribute)
    except ImportError as exc:
        _fail(script, exc.name or str(exc))
    return target()


def ontocast() -> Any:
    """Entry point for the `ontocast` command."""
    return _run("ontocast", "ontocast.cli.server", "cli")


def plot_graph() -> Any:
    """Entry point for the `plot-graph` command."""
    return _run("plot-graph", "ontocast.cli.plot_graph", "main")


def match_graphs() -> Any:
    """Entry point for the `match-graphs` command."""
    return _run("match-graphs", "ontocast.cli.match_graphs", "main")


def pdfs_to_markdown() -> Any:
    """Entry point for the `pdfs-to-markdown` command."""
    return _run("pdfs-to-markdown", "ontocast.cli.pdfs_to_markdown", "main")


def test_api() -> Any:
    """Entry point for the `test-api` command."""
    return _run("test-api", "ontocast.cli.test_api", "main")
