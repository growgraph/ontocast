"""Seed-directory resolution for ``ontocast serve`` and ``ontocast process``.

A directory the operator named is an assertion, and the three ways it can fail
are not the same fault. These pin which of them stops a run.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile

import click
import pytest

from ontocast.cli.server import _prepare_path_config, _resolve_seed_directory
from ontocast.config import Config, FactsValidationConfig, PathConfig, ToolConfig

pytestmark = pytest.mark.unit


def _config(ontology_directory=None, shapes_dir=None) -> Config:
    return Config(
        tool_config=ToolConfig(
            path_config=PathConfig(ontology_directory=ontology_directory),
            facts_validation=FactsValidationConfig(shapes_dir=shapes_dir),
        )
    )


# --- _resolve_seed_directory ------------------------------------------------


def test_an_unnamed_directory_is_not_a_fault() -> None:
    assert (
        _resolve_seed_directory(
            None,
            source="ONTOCAST_ONTOLOGY_DIRECTORY",
            kind="ontology",
            missing_is_fatal=True,
        )
        is None
    )


def test_a_named_directory_that_does_not_resolve_stops_a_batch_run() -> None:
    """The failure that costs the most to diagnose anywhere else.

    Nothing downstream can tell a mistyped path from "deliberately none", so
    the catalog just comes out empty and every symptom surfaces minutes later
    inside retrieval, naming the vector index rather than the typo.
    """
    with pytest.raises(click.UsageError) as excinfo:
        _resolve_seed_directory(
            "/nonexistent/ontologies",
            source="--ontology-dir",
            kind="ontology",
            missing_is_fatal=True,
        )

    message = str(excinfo.value)
    assert "--ontology-dir" in message, "the error must name what asserted the path"
    assert "/nonexistent/ontologies" in message


def test_a_named_directory_that_does_not_resolve_only_warns_a_server(caplog) -> None:
    """A long-lived server may have the directory appear under it later."""
    with caplog.at_level(logging.WARNING):
        _resolve_seed_directory(
            "/nonexistent/ontologies",
            source="ONTOCAST_ONTOLOGY_DIRECTORY",
            kind="ontology",
            missing_is_fatal=False,
        )

    assert "/nonexistent/ontologies" in caplog.text


def test_a_resolvable_directory_is_expanded_and_logged(caplog) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with caplog.at_level(logging.INFO):
            resolved = _resolve_seed_directory(
                tmp, source="--ontology-dir", kind="ontology", missing_is_fatal=True
            )

        assert resolved == pathlib.Path(tmp)
        assert str(pathlib.Path(tmp).absolute()) in caplog.text


# --- _prepare_path_config: the three states of each flag --------------------


def test_an_omitted_flag_leaves_the_environment_in_force() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = _config(ontology_directory=tmp, shapes_dir=tmp)

        _prepare_path_config(config)

        assert config.tool_config.path_config.ontology_directory == pathlib.Path(tmp)
        assert config.tool_config.facts_validation.shapes_dir == tmp


def test_a_flag_overrides_the_environment() -> None:
    with tempfile.TemporaryDirectory() as env_dir, tempfile.TemporaryDirectory() as cli:
        config = _config(ontology_directory=env_dir, shapes_dir=env_dir)

        _prepare_path_config(config, ontology_dir=cli, shapes_dir=cli)

        assert config.tool_config.path_config.ontology_directory == pathlib.Path(cli)
        assert config.tool_config.facts_validation.shapes_dir == cli


def test_an_empty_flag_clears_a_configured_directory() -> None:
    """How a run declares "no seed ontologies, infer them".

    This is why the flags are plain strings rather than ``click.Path``:
    ``pathlib.Path("")`` is ``.``, so an empty value would otherwise seed the
    working directory.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = _config(ontology_directory=tmp, shapes_dir=tmp)

        _prepare_path_config(config, ontology_dir="", shapes_dir="")

        assert config.tool_config.path_config.ontology_directory is None
        assert config.tool_config.facts_validation.shapes_dir is None


def test_clearing_a_directory_is_never_the_missing_path_error() -> None:
    """An explicit "none" is an assertion that resolves, not one that fails."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _config(ontology_directory=tmp)

        _prepare_path_config(config, ontology_dir="", missing_is_fatal=True)

        assert config.tool_config.path_config.ontology_directory is None


def test_a_batch_run_rejects_a_mistyped_flag() -> None:
    config = _config()

    with pytest.raises(click.UsageError):
        _prepare_path_config(
            config, ontology_dir="/nonexistent/ontologies", missing_is_fatal=True
        )
