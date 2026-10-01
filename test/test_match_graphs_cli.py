"""``match-graphs`` resolves unset aligner options from ``AggregationConfig``."""

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from ontocast.cli import match_graphs
from ontocast.config.settings import AggregationConfig

_TTL = "<https://example.org/a> <https://example.org/r> <https://example.org/b> .\n"


def _invoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str
) -> dict[str, Any]:
    gt = tmp_path / "gt.ttl"
    predicted = tmp_path / "predicted.ttl"
    gt.write_text(_TTL)
    predicted.write_text(_TTL)
    captured: dict[str, Any] = {}

    def _fake_run_match(_gt, _predicted, **kwargs):
        captured.update(kwargs)
        metrics = dict.fromkeys(
            ("precision", "recall", "f1", "entity_f1", "fact_f1"), 0.0
        )
        return {"metrics": metrics, "entity_match_count": 0}, []

    monkeypatch.setattr(match_graphs, "_run_match", _fake_run_match)
    result = CliRunner().invoke(
        match_graphs.main,
        ["--gt", str(gt), "--predicted", str(predicted), "--no-verbose", *extra],
    )
    assert result.exit_code == 0, result.output
    return captured


def test_unset_options_come_from_aggregation_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGG_SIMILARITY_THRESHOLD", "0.65")
    captured = _invoke(tmp_path, monkeypatch)
    assert captured["similarity_threshold"] == 0.65
    assert captured["embedding_model"] == AggregationConfig().embedding_model


def test_explicit_options_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGG_SIMILARITY_THRESHOLD", "0.65")
    captured = _invoke(
        tmp_path,
        monkeypatch,
        "--similarity-threshold",
        "0.5",
        "--embedding-model",
        "some/model",
    )
    assert captured["similarity_threshold"] == 0.5
    assert captured["embedding_model"] == "some/model"


def test_undefined_metrics_print_as_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gt = tmp_path / "gt.ttl"
    predicted = tmp_path / "predicted.ttl"
    gt.write_text(_TTL)
    predicted.write_text("")

    def _fake_run_match(_gt, _predicted, **_kwargs):
        metrics = dict.fromkeys(("precision", "entity_f1", "fact_f1", "f1"), None)
        metrics["recall"] = 0.0
        return {"metrics": metrics, "entity_match_count": 0}, []

    monkeypatch.setattr(match_graphs, "_run_match", _fake_run_match)
    result = CliRunner().invoke(
        match_graphs.main,
        ["--gt", str(gt), "--predicted", str(predicted), "--no-verbose"],
    )
    assert result.exit_code == 0, result.output
    assert "P=n/a R=0.0000" in result.output
