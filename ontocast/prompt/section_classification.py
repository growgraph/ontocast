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


CHUNK_SECTION_BATCH_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify excerpts from one structured document into "
            "normalised section labels. Use only these labels: "
            "{allowed_labels}. The excerpts are given in document order and "
            "are numbered; return one assignment per number, reusing the same "
            "index. Judge each excerpt on structural cues — whether it reads "
            "as framing, procedure, reported measurements, interpretation, "
            "boilerplate or a reference list — not on its subject matter. Set "
            "label to null for any excerpt that does not clearly belong to a "
            "section type. {format_instructions}",
        ),
        (
            "human",
            "{document_context}Excerpts:\n{items}",
        ),
    ]
)


def format_batch_items(items: list[tuple[int, str]]) -> str:
    """Render ``(index, fragment)`` pairs for the batched classification prompt."""
    return "\n\n".join(f"[{index}] {fragment}" for index, fragment in items)


def document_type_context(document_type: str | None) -> str:
    """Optional human-message prefix when the caller supplies a document type hint."""
    if document_type is None:
        return ""
    stripped = document_type.strip()
    if not stripped:
        return ""
    return f"Optional context — the source material is described as: {stripped}.\n\n"
