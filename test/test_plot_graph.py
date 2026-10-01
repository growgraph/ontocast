"""``plot-graph`` renders the topology without any provider configured."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from ontocast.cli import plot_graph


def test_plot_graph_needs_no_llm_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    for name in ("LLM_API_KEY", "OPENAI_API_KEY", "FUSEKI_URI", "QDRANT_URI"):
        monkeypatch.delenv(name, raising=False)

    result = CliRunner().invoke(
        plot_graph.main, ["--output-dir", str(tmp_path), "--format", "svg"]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "graph.mmd").read_text().strip()
