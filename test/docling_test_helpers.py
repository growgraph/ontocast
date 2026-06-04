"""Test helpers for building DoclingDocument fixtures."""

from docling_core.types.doc import DocItemLabel, DoclingDocument

from ontocast.onto.docling_helpers import plain_text_to_docling_doc


def plain_doc(text: str, name: str = "test") -> DoclingDocument:
    """Wrap plain text as a single-paragraph DoclingDocument."""
    return plain_text_to_docling_doc(text, name)


def doc_from_markdown_lines(text: str, name: str = "test") -> DoclingDocument:
    """Build a DoclingDocument from markdown-style heading/body lines."""
    doc = DoclingDocument(name=name)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            doc.add_text(label=DocItemLabel.SECTION_HEADER, text=heading)
        else:
            doc.add_text(label=DocItemLabel.PARAGRAPH, text=stripped)
    return doc
