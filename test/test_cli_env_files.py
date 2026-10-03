"""Layered settings files: ``ontocast --env-file`` and ``ontocast config check``."""

import logging
import os
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest
from click.testing import CliRunner
from dotenv import dotenv_values

from ontocast.cli.env_files import apply_env_files, merge_env_files
from ontocast.cli.server import cli
from ontocast.config.env_audit import Verdict, audit_assignments
from ontocast.config.settings import WebSearchConfig

pytestmark = pytest.mark.unit


@pytest.fixture
def restored_environ() -> Iterator[None]:
    """`apply_env_files` writes ``os.environ`` directly; undo it after the test."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _verdicts(text: str) -> dict[str, Verdict]:
    report = audit_assignments(dict(dotenv_values(stream=StringIO(text))))
    return {f.name: f.verdict for f in report.findings}


def test_later_file_wins(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.conf", "CHUNK_MIN_SIZE=1000\nPARALLEL_WORKERS=2\n")
    second = _write(tmp_path / "second.conf", "CHUNK_MIN_SIZE=2000\n")

    merged = merge_env_files([first, second])

    assert merged == {"CHUNK_MIN_SIZE": "2000", "PARALLEL_WORKERS": "2"}


def test_shell_export_wins_over_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restored_environ: None
) -> None:
    monkeypatch.setenv("CHUNK_MIN_SIZE", "4321")
    monkeypatch.delenv("PARALLEL_WORKERS", raising=False)
    conf = _write(tmp_path / "p.conf", "CHUNK_MIN_SIZE=1000\nPARALLEL_WORKERS=2\n")

    apply_env_files([conf])

    assert os.environ["CHUNK_MIN_SIZE"] == "4321"
    assert os.environ["PARALLEL_WORKERS"] == "2"


def test_unknown_variable_is_named_but_its_value_is_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restored_environ: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("ONTOCAST_NO_SUCH_KNOB", raising=False)
    conf = _write(tmp_path / "p.conf", "ONTOCAST_NO_SUCH_KNOB=s3cr3t-value\n")

    with caplog.at_level(logging.WARNING, logger="ontocast.cli.env_files"):
        apply_env_files([conf])

    assert "ONTOCAST_NO_SUCH_KNOB" in caplog.text
    assert "s3cr3t-value" not in caplog.text
    # Exported anyway: libraries read variables OntoCast does not declare.
    assert os.environ["ONTOCAST_NO_SUCH_KNOB"] == "s3cr3t-value"


def test_group_option_loads_files_before_the_subcommand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restored_environ: None
) -> None:
    monkeypatch.delenv("CHUNK_MIN_SIZE", raising=False)
    conf = _write(tmp_path / "p.conf", "CHUNK_MIN_SIZE=1234\n")

    result = CliRunner().invoke(cli, ["--env-file", str(conf), "cache", "--help"])

    assert result.exit_code == 0, result.output
    assert os.environ["CHUNK_MIN_SIZE"] == "1234"


def test_audit_classifies_each_verdict() -> None:
    verdicts = _verdicts(
        "CONVERTER_PROFILE=auto\n"
        "CHUNK_MAX_SIZE=99999\n"
        "WEB_SEARCH_TOP_K=not-a-number\n"
        "ONTOCAST_NO_SUCH_KNOB=1\n"
    )

    assert verdicts == {
        "CONVERTER_PROFILE": Verdict.REDUNDANT,
        "CHUNK_MAX_SIZE": Verdict.PINNED,
        "WEB_SEARCH_TOP_K": Verdict.INVALID,
        "ONTOCAST_NO_SUCH_KNOB": Verdict.UNKNOWN,
    }


def test_audit_ignores_the_ambient_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value is judged against the code default, not the caller's shell."""
    monkeypatch.setenv("CONVERTER_PROFILE", "ocr")

    assert _verdicts("CONVERTER_PROFILE=auto\n") == {
        "CONVERTER_PROFILE": Verdict.REDUNDANT
    }
    assert os.environ["CONVERTER_PROFILE"] == "ocr"


def test_audit_reports_removed_value_without_echoing_input() -> None:
    report = audit_assignments({"CONVERTER_PROFILE": "born_digital"})

    (finding,) = report.findings
    assert finding.verdict is Verdict.INVALID
    assert "removed" in finding.detail
    assert not report.ok


def test_audit_reports_values_rejected_only_in_combination() -> None:
    report = audit_assignments(
        {
            "LLM_PROVIDER": "google",
            "LLM_REASONING_EFFORT": "low",
            "LLM_THINKING_BUDGET": "512",
        }
    )

    assert {f.verdict for f in report.findings} == {Verdict.INVALID}
    assert all("combination" in f.detail for f in report.findings)


def test_audit_reports_cross_group_errors() -> None:
    report = audit_assignments(
        {"QDRANT_URI": "http://localhost:6333", "LANCEDB_ENABLED": "true"}
    )

    assert {f.verdict for f in report.findings} == {Verdict.PINNED}
    assert report.config_errors
    assert not report.ok


def test_config_check_exit_status(tmp_path: Path) -> None:
    clean = _write(tmp_path / "clean.conf", "CHUNK_MAX_SIZE=99999\n")
    stale = _write(tmp_path / "stale.conf", "CHUNK_BREAKPOINT_THRESHOLD_TYPE=x\n")
    runner = CliRunner()

    ok = runner.invoke(cli, ["config", "check", str(clean)])
    bad = runner.invoke(cli, ["config", "check", str(clean), str(stale)])

    assert ok.exit_code == 0, ok.output
    assert "pinned     CHUNK_MAX_SIZE" in ok.output
    assert bad.exit_code == 1
    assert "unknown    CHUNK_BREAKPOINT_THRESHOLD_TYPE" in bad.output


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", []),
        ("a.com, B.org", ["a.com", "b.org"]),
        ('["a.com"]', ["a.com"]),
    ],
)
def test_web_search_domains_parse_from_environment(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]
) -> None:
    monkeypatch.setenv("WEB_SEARCH_ALLOWED_DOMAINS", raw)

    assert WebSearchConfig().allowed_domains == expected
