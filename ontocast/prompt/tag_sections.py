"""Prompt templates for section heading classification."""

from langchain_core.prompts import ChatPromptTemplate

HEADING_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify section headings from structured text into normalised labels. "
            "Use only these labels: {allowed_labels}. "
            "If a heading does not match any section type, set label to null. "
            "{format_instructions}",
        ),
        (
            "human",
            "{document_context}Classify each heading (one per line):\n{headings}",
        ),
    ]
)


def document_type_context(document_type: str | None) -> str:
    """Optional human-message prefix when the caller supplies a document type hint."""
    if document_type is None:
        return ""
    stripped = document_type.strip()
    if not stripped:
        return ""
    return f"Optional context — the source material is described as: {stripped}.\n\n"
