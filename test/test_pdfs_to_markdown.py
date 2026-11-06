"""``pdfs-to-markdown`` writes files OntoCast reads back as documents."""

import pathlib

import pytest

from ontocast.cli import pdfs_to_markdown
from ontocast.config.settings import DEFAULT_CONVERTER_EXTENSIONS

pytestmark = pytest.mark.unit

# What export_to_markdown() writes for section headers: level two.
MARKDOWN = "## Methods\n\nThe samples were annealed.\n\n## Results\n\nThe films held.\n"


def test_process_writes_markdown_named_after_the_pdf(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pdfs_to_markdown, "pdf2markdown", lambda f: MARKDOWN)
    out = tmp_path / "out" / "nested"

    written = pdfs_to_markdown.process(out, tmp_path / "paper.pdf")

    assert written == out / "paper.md"
    assert written.read_text(encoding="utf-8") == MARKDOWN


def test_output_is_a_default_input_and_keeps_its_headings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("docling")
    from ontocast.tool.converter import ConverterTool

    assert ".md" in DEFAULT_CONVERTER_EXTENSIONS
    monkeypatch.setattr(pdfs_to_markdown, "pdf2markdown", lambda f: MARKDOWN)
    written = pdfs_to_markdown.process(tmp_path, tmp_path / "paper.pdf")

    doc = ConverterTool()(written)

    headings = [item.text for item in doc.texts if item.label == "section_header"]
    assert headings == ["Methods", "Results"]
