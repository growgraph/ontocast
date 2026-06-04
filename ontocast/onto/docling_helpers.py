"""Helpers for constructing DoclingDocument instances in the OntoCast pipeline."""

from docling_core.types.doc import DocItemLabel, DoclingDocument


def plain_text_to_docling_doc(text: str, doc_name: str) -> DoclingDocument:
    """Wrap plain text as a single-paragraph DoclingDocument."""
    doc = DoclingDocument(name=doc_name)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text=text)
    return doc
