"""Every test fixture must live under ``test/``.

The sdist ships ``/test`` and an allowlist of root files -- see
``[tool.hatch.build.targets.sdist]``. A test that resolves a path outside
``test/`` therefore cannot run from a published sdist, and until this guard
existed four of them did: three reached into a 1.1 MB ``data/`` corpus the
sdist deliberately excluded, and one reached into a *sibling repository* in the
author's workspace. Three of the four skipped when the path was absent, so the
coverage did not fail on the machines that lacked it -- it silently vanished.

The escape is always the same idiom -- walking up from ``__file__`` past
``test/`` -- so that is what is checked, exactly rather than heuristically.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TEST_ROOT = Path(__file__).resolve().parent

#: ``parents[N]``, counted from the file's own directory: a file at
#: ``test/sub/x.py`` reaches ``test/`` at 1 and the repo root at 2.
_PARENTS_INDEX = re.compile(r"parents\[\s*(\d+)\s*\]")

#: Files permitted to walk out of ``test/``, and the repo-root entries each may
#: reach. Every one of these reads a *declaration* -- settings, source, docs --
#: not a corpus; that is the line the allowlist draws.
ALLOWED_ESCAPES: dict[str, set[str]] = {
    "test_env_example_coverage.py": {
        ".env.example",
        ".env.example.minimal",
        "README.md",
        "docs",
    },
    "test_retrieval_metric_keys.py": {"ontocast"},
    "test_repo_isolation.py": {"pyproject.toml"},
}

#: Allowlisted targets the sdist does *not* ship. A reader of one of these must
#: skip when it is absent, so an installed-sdist run reports "skipped", never a
#: FileNotFoundError. Enforced by :func:`test_unshipped_reads_are_guarded`.
NOT_IN_SDIST = {"docs"}


def _escape_depths(source: str) -> list[int]:
    """Depths above its own directory that this source walks to.

    ``parents[2]`` and ``Path(__file__).parent.parent`` both score 1 -- the
    first ``.parent`` is the file's directory, which ``parents[0]`` also names,
    so the two spellings are normalised onto one scale before comparison.
    """
    depths = [int(match.group(1)) for match in _PARENTS_INDEX.finditer(source)]
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute) or node.attr != "parent":
            continue
        # Score the outermost link of a `.parent.parent` chain only.
        if isinstance(node.value, ast.Attribute) and node.value.attr == "parent":
            continue
        depth, current = 0, node
        while isinstance(current, ast.Attribute) and current.attr == "parent":
            depth += 1
            current = current.value
        if "__file__" in ast.dump(current):
            depths.append(depth - 1)
    return depths


def _budget(path: Path) -> int:
    """How far up from ``path``'s directory ``test/`` sits."""
    return len(path.resolve().relative_to(TEST_ROOT).parts) - 1


def _sources() -> list[Path]:
    return sorted(TEST_ROOT.rglob("*.py"))


def test_no_test_file_resolves_a_path_outside_test() -> None:
    offenders: list[str] = []
    for path in _sources():
        source = path.read_text(encoding="utf-8")
        budget = _budget(path)
        worst = max(_escape_depths(source), default=-1)
        if worst <= budget:
            continue
        allowed = ALLOWED_ESCAPES.get(path.name)
        if allowed is None:
            offenders.append(
                f"{path.relative_to(TEST_ROOT.parent)} walks {worst} level(s) up "
                f"from its directory, past test/ at {budget}. Move the fixture "
                "under test/, or add an entry to ALLOWED_ESCAPES saying what "
                "declaration file it reads and why."
            )
        elif not any(f'"{name}' in source for name in allowed):
            offenders.append(
                f"{path.relative_to(TEST_ROOT.parent)} is allowlisted for "
                f"{sorted(allowed)} but names none of them"
            )
    assert offenders == [], "\n".join(offenders)


def test_allowlist_names_only_files_the_sdist_ships() -> None:
    """An allowlisted escape is useless if the sdist drops what it reads."""
    include = (TEST_ROOT.parent / "pyproject.toml").read_text(encoding="utf-8")
    shipped = set(re.findall(r'^\s*"/([^"]+)",?\s*$', include, re.M))
    assert shipped, "could not read the sdist include list from pyproject.toml"
    for name, targets in ALLOWED_ESCAPES.items():
        for target in targets - NOT_IN_SDIST:
            assert target in shipped, (
                f"{name} is allowed to read {target}, which the sdist include "
                f"list does not ship (ships: {sorted(shipped)}). Either ship "
                "it, or add it to NOT_IN_SDIST and make the reader skip."
            )


def test_unshipped_reads_are_guarded() -> None:
    """A reader of an unshipped path must skip, not raise, when it is absent."""
    for name, targets in ALLOWED_ESCAPES.items():
        if not targets & NOT_IN_SDIST:
            continue
        source = (TEST_ROOT / name).read_text(encoding="utf-8")
        assert "is_file()" in source or "exists()" in source, (
            f"{name} reads {sorted(targets & NOT_IN_SDIST)}, which the sdist "
            "does not ship, but has no existence check -- from an installed "
            "sdist it would fail rather than skip"
        )
