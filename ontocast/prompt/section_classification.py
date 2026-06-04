"""Prompt templates for section classification during chunk prepare."""

from langchain_core.prompts import ChatPromptTemplate

CHUNK_SECTION_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify a short excerpt from a structured document into one "
            "normalised section label. Use only these labels: {allowed_labels}. "
            "If the excerpt does not clearly belong to any section type, set label "
            "to null. {format_instructions}",
        ),
        (
            "human",
            "{document_context}Excerpt:\n{fragment}",
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
