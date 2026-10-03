"""Load layered dotenv files into the environment for the ``ontocast`` CLI.

Files apply in the order given and a later file overrides an earlier one, so a
stack such as provider credentials, then project settings, reads as one
configuration. A variable already exported in the shell is never overridden,
which keeps a one-off ``VAR=value ontocast ...`` effective on top of the stack.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import dotenv_values

from ontocast.config.env_audit import declared_env_names

logger = logging.getLogger(__name__)


def read_env_file(path: Path) -> dict[str, str | None]:
    """Parse one dotenv file; ``None`` marks a line with no ``=``."""
    return dict(dotenv_values(path))


def merge_env_files(paths: Sequence[Path]) -> dict[str, str | None]:
    """Merge dotenv files in order, a later file winning on a shared name."""
    merged: dict[str, str | None] = {}
    for path in paths:
        merged.update(read_env_file(path))
    return merged


def apply_env_files(paths: Sequence[Path]) -> None:
    """Export the merged files into ``os.environ`` beneath the shell's own values.

    Variables no setting reads are still exported -- libraries read some of
    them, such as ``HF_TOKEN`` -- but are named in a warning, since a renamed or
    removed OntoCast setting looks exactly like one. Values are never logged.
    """
    merged = merge_env_files(paths)
    declared = declared_env_names()
    unknown = sorted(name for name in merged if name.upper() not in declared)
    if unknown:
        logger.warning(
            "--env-file sets variables no OntoCast setting reads (exported anyway; "
            "a renamed or removed setting if you meant one): %s",
            ", ".join(unknown),
        )
    for name, value in merged.items():
        if value is None or name in os.environ:
            continue
        os.environ[name] = value
