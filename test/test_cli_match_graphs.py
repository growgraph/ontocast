from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from ontocast.cli import match_graphs


def test_match_graphs_cli_outputs_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    left_path = tmp_path / "left.ttl"
    right_path = tmp_path / "right.ttl"
    left_path.write_text(
        "@prefix ex: <https://left.example/> . ex:a <https://p/rel> ex:b .",
        encoding="utf-8",
    )
    right_path.write_text(
        "@prefix ex: <https://right.example/> . ex:a <https://p/rel> ex:b .",
        encoding="utf-8",
    )

    class _FakeMatcher:
        def __init__(self, embedding_model: str, similarity_threshold: float) -> None:
            self.embedding_model = embedding_model
            self.similarity_threshold = similarity_threshold

        def match(self, **_kwargs):
            class _Result:
                def model_dump(self, mode: str = "python") -> dict:
                    return {
                        "regime": "ontology_loose",
                        "ground_truth_side": "right",
                        "entity_matches": [{"left_entity": "a", "right_entity": "b"}],
                        "metrics": {
                            "precision": 0.5,
                            "recall": 1.0,
                            "f1": 0.6666666667,
                        },
                    }

            return _Result()

    monkeypatch.setattr(match_graphs, "TripleSetMatcher", _FakeMatcher)
    runner = CliRunner()
    result = runner.invoke(
        match_graphs.main,
        [
            "--left",
            str(left_path),
            "--right",
            str(right_path),
            "--json-output",
        ],
    )
    assert result.exit_code == 0
    assert "Metrics: P=0.5000 R=1.0000 F1=0.6667" in result.output
