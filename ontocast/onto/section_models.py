"""Pydantic models for document section spans."""

from ontocast.config.section_labels import (
    canonical_labels,
    get_default_section_schema,
)
from ontocast.onto.model import BasePydanticModel

# Backward-compatible alias for prompts and imports.
CANONICAL_SECTION_LABELS: tuple[str, ...] = canonical_labels(
    get_default_section_schema()
)


class SectionSpan(BasePydanticModel):
    """Character span of a document section with a normalised label."""

    label: str
    start: int
    end: int


__all__ = [
    "CANONICAL_SECTION_LABELS",
    "SectionSpan",
]
