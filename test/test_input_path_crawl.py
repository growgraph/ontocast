"""Input-path resolution for the file-crawling CLIs.

Issue #53: a mistyped or single-file ``--input-path`` printed a line to stdout,
processed nothing, and exited 0 -- indistinguishable from a successful run.
"""

import pathlib

import pytest
from click.testing import CliRunner

from ontocast.util.files import crawl_directories

pytestmark = pytest.mark.unit


def test_single_file_is_accepted(tmp_path: pathlib.Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4")

    assert crawl_directories(paper, suffixes=(".pdf", ".json")) == [paper]


def test_single_file_with_unsupported_suffix_raises(tmp_path: pathlib.Path) -> None:
    other = tmp_path / "notes.md"
    other.write_text("text", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported input file"):
        crawl_directories(other, suffixes=(".pdf", ".json"))


def test_single_file_honours_prefix(tmp_path: pathlib.Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4")

    assert crawl_directories(paper, suffixes=(".pdf",), prefix="paper") == [paper]
    # An explicitly named file excluded by the prefix filter is a contradictory
    # invocation -- silently returning [] is the exact no-op issue #53 removed.
    with pytest.raises(ValueError, match="does not match prefix"):
        crawl_directories(paper, suffixes=(".pdf",), prefix="other")


def test_suffix_match_is_case_insensitive(tmp_path: pathlib.Path) -> None:
    upper = tmp_path / "report.PDF"
    upper.write_bytes(b"%PDF-1.4")

    assert crawl_directories(upper, suffixes=(".pdf",)) == [upper]
    assert crawl_directories(tmp_path, suffixes=(".pdf",)) == [upper]


def test_missing_path_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="neither a file nor a directory"):
        crawl_directories(tmp_path / "absent.pdf", suffixes=(".pdf",))


def test_directory_is_crawled_recursively(tmp_path: pathlib.Path) -> None:
    (tmp_path / "nested").mkdir()
    first = tmp_path / "a.pdf"
    second = tmp_path / "nested" / "b.pdf"
    first.write_bytes(b"%PDF-1.4")
    second.write_bytes(b"%PDF-1.4")
    (tmp_path / "skip.md").write_text("text", encoding="utf-8")

    assert sorted(crawl_directories(tmp_path, suffixes=(".pdf",))) == [first, second]


def test_empty_directory_returns_no_files(tmp_path: pathlib.Path) -> None:
    """A directory that matches nothing is not an error -- callers decide."""
    assert crawl_directories(tmp_path, suffixes=(".pdf",)) == []


def test_cli_reports_a_bad_input_path_as_a_parameter_error(
    tmp_path: pathlib.Path,
) -> None:
    """The ValueError must reach the user as a non-zero exit, not a stdout line."""
    from ontocast.cli.pdfs_to_markdown import main

    result = CliRunner().invoke(
        main,
        [
            "--input-path",
            str(tmp_path / "absent.pdf"),
            "--output-path",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "neither a file nor a directory" in result.output
    # param_hint is what tells the user *which* path was wrong.
    assert "--input-path" in result.output


def test_cli_accepts_a_single_file(tmp_path: pathlib.Path, monkeypatch) -> None:
    from ontocast.cli import pdfs_to_markdown

    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4")
    processed: list[pathlib.Path] = []
    monkeypatch.setattr(pdfs_to_markdown, "process", lambda out, f: processed.append(f))

    result = CliRunner().invoke(
        pdfs_to_markdown.main,
        ["--input-path", str(paper), "--output-path", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert processed == [paper]
