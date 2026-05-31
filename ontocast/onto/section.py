"""Document section span models and helpers for structured-document preprocessing."""

from __future__ import annotations

import re

from pydantic import Field

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.model import BasePydanticModel

# Optional numeric or Roman-numeral section prefix (e.g. "1.", "II.", "2.1)").
_SECTION_PREFIX = r"(?:\d+|[IVXivx]+)(?:\.\d+)*[.)]\s*"

# Chapter / Section / Part prefix (e.g. "Chapter 3:", "Section II —").
_STRUCTURAL_PREFIX = re.compile(
    r"^(?:chapter|section|part)\s+(?:\d+|[IVXivx]+)(?:\.\d+)*[.:)\-–—]?\s*",
    re.I,
)

# Normalised section labels and heading line patterns (after stripping markdown #).
_SECTION_HEADING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "abstract",
        re.compile(rf"^(?:{_SECTION_PREFIX})?abstract\s*$", re.I),
    ),
    (
        "abstract",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:executive\s+summary|synopsis)\s*$",
            re.I,
        ),
    ),
    (
        "introduction",
        re.compile(rf"^(?:{_SECTION_PREFIX})?introduction\s*$", re.I),
    ),
    (
        "introduction",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:overview|preamble|foreword|preface|motivation)\s*$",
            re.I,
        ),
    ),
    (
        "related_work",
        re.compile(rf"^(?:{_SECTION_PREFIX})?related\s+work\s*$", re.I),
    ),
    (
        "related_work",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:related\s+(?:literature|approaches?|research|studies)|prior\s+(?:work|art)|literature\s+(?:review|survey)|state\s+of\s+the\s+art|survey)\s*$",
            re.I,
        ),
    ),
    (
        "background",
        re.compile(rf"^(?:{_SECTION_PREFIX})?background\s*$", re.I),
    ),
    (
        "methods",
        re.compile(rf"^(?:{_SECTION_PREFIX})?methods?\s*$", re.I),
    ),
    (
        "methods",
        re.compile(rf"^(?:{_SECTION_PREFIX})?methodology\s*$", re.I),
    ),
    (
        "methods",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:materials?\s+and\s+)?(?:methods?|methodology|approach|experimental\s+setup|proposed\s+(?:method|model|framework|system)|implementation|design|procedure|protocol|framework|architecture)\s*$",
            re.I,
        ),
    ),
    (
        "results",
        re.compile(rf"^(?:{_SECTION_PREFIX})?results?\s*$", re.I),
    ),
    (
        "results",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:experimental\s+)?(?:results?|findings?|evaluation|experiments?|ablation(?:\s+stud(?:y|ies))?|outcomes?|performance|benchmarks?)\s*$",
            re.I,
        ),
    ),
    (
        "discussion",
        re.compile(rf"^(?:{_SECTION_PREFIX})?discussion\s*$", re.I),
    ),
    (
        "discussion",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:analysis|interpretation|observations?)\s*$",
            re.I,
        ),
    ),
    (
        "conclusion",
        re.compile(rf"^(?:{_SECTION_PREFIX})?conclusions?\s*$", re.I),
    ),
    (
        "conclusion",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:conclusions?|concluding\s+remarks?|summary|final\s+remarks?|final\s+thoughts|takeaways?|wrap[\s-]?up)\s*$",
            re.I,
        ),
    ),
    (
        "future_work",
        re.compile(rf"^(?:{_SECTION_PREFIX})?future\s+work\s*$", re.I),
    ),
    (
        "future_work",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:future\s+(?:work|directions?|research)|next\s+steps?|roadmap|outlook)\s*$",
            re.I,
        ),
    ),
    (
        "limitations",
        re.compile(rf"^(?:{_SECTION_PREFIX})?limitations?\s*$", re.I),
    ),
    (
        "acknowledgements",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?acknowledg(?:e?ments?|ments?)\s*$",
            re.I,
        ),
    ),
    (
        "data",
        re.compile(rf"^(?:{_SECTION_PREFIX})?data(?:set|sets)?\s*$", re.I),
    ),
    (
        "data",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:corpus|data\s+(?:collection|description))\s*$",
            re.I,
        ),
    ),
    (
        "appendix",
        re.compile(rf"^(?:{_SECTION_PREFIX})?appendi(?:x|ces)\s*$", re.I),
    ),
    (
        "appendix",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:supplementary\s+material|supplement)\s*$",
            re.I,
        ),
    ),
    (
        "references",
        re.compile(rf"^(?:{_SECTION_PREFIX})?references\s*$", re.I),
    ),
    (
        "references",
        re.compile(
            rf"^(?:{_SECTION_PREFIX})?(?:bibliography|works\s+cited)\s*$",
            re.I,
        ),
    ),
)

CANONICAL_SECTION_LABELS: tuple[str, ...] = (
    "abstract",
    "introduction",
    "related_work",
    "background",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "future_work",
    "limitations",
    "acknowledgements",
    "data",
    "appendix",
    "references",
)

_MAX_HEADING_LINE_LEN = 120
_MAX_HEADING_WORDS = 12


class SectionSpan(BasePydanticModel):
    """Character span of a document section with a normalised label."""

    label: str = Field(
        description="Normalised section label, e.g. results, future_work"
    )
    start: int = Field(description="Start character offset in input_text (inclusive)")
    end: int = Field(description="End character offset in input_text (exclusive)")


def _normalise_heading_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        stripped = stripped.lstrip("#").strip()
    stripped = _STRUCTURAL_PREFIX.sub("", stripped).strip()
    return stripped


def _looks_like_heading_line(raw_line: str, normalised: str) -> bool:
    """Heuristic: short title-like line that may be an unmatched section heading."""
    if not normalised or len(normalised) > _MAX_HEADING_LINE_LEN:
        return False
    stripped = raw_line.strip()
    if stripped.startswith("#"):
        return True
    if re.match(rf"^(?:{_SECTION_PREFIX})", normalised):
        return True
    words = normalised.split()
    if len(words) > _MAX_HEADING_WORDS:
        return False
    if re.search(
        r"\b(the|we|our|this|that|is|are|was|were|have|has|had|with|from|for)\b",
        normalised,
        re.I,
    ):
        return False
    if normalised.endswith((".", ",", ";", ":")) and len(words) > 3:
        return False
    if normalised.isupper() and len(words) <= _MAX_HEADING_WORDS:
        return True
    if len(words) <= 8 and normalised == normalised.title():
        return True
    return False


def _match_section_label(heading_line: str) -> str | None:
    normalised = _normalise_heading_line(heading_line)
    if not normalised or len(normalised) > _MAX_HEADING_LINE_LEN:
        return None
    for label, pattern in _SECTION_HEADING_PATTERNS:
        if pattern.match(normalised):
            return label
    return None


def normalise_user_section_label(raw: str) -> str | None:
    """Map a free-text user-supplied section name to a canonical label.

    Pipeline: clean → check canonical → try regex match → None.
    Passes '*' through unchanged.
    """
    if raw.strip() == "*":
        return "*"
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in CANONICAL_SECTION_LABELS:
        return cleaned
    matched = _match_section_label(raw)
    if matched is not None:
        return matched
    return None


def _build_spans_from_heading_starts(
    text: str, heading_starts: list[tuple[int, str]]
) -> list[SectionSpan]:
    if not heading_starts:
        return []
    sorted_starts = sorted(heading_starts, key=lambda item: item[0])
    spans: list[SectionSpan] = []
    for index, (start, label) in enumerate(sorted_starts):
        end = (
            sorted_starts[index + 1][0] if index + 1 < len(sorted_starts) else len(text)
        )
        if end > start:
            spans.append(SectionSpan(label=label, start=start, end=end))
    return spans


def iter_heading_lines(text: str) -> list[tuple[int, str, str | None]]:
    """Return (offset, normalised_heading, regex_label_or_none) for candidate headings."""
    candidates: list[tuple[int, str, str | None]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        normalised = _normalise_heading_line(line)
        if _looks_like_heading_line(line, normalised):
            label = _match_section_label(line)
            candidates.append((offset, normalised, label))
        offset += len(line)
    return candidates


def detect_section_spans(text: str) -> list[SectionSpan]:
    """Detect academic-style section headings and return character spans."""
    if not text:
        return []

    heading_starts: list[tuple[int, str]] = []
    for offset, _normalised, label in iter_heading_lines(text):
        if label is not None:
            heading_starts.append((offset, label))
    return _build_spans_from_heading_starts(text, heading_starts)


def build_section_spans_from_labels(
    text: str, labeled_headings: list[tuple[int, str]]
) -> list[SectionSpan]:
    """Build section spans from explicit (offset, label) pairs."""
    return _build_spans_from_heading_starts(text, labeled_headings)


def _chunk_char_range(
    chunk_text: str, document_text: str, search_from: int
) -> tuple[int, int]:
    """Locate chunk in document_text; return (start, end) or (0, 0) if not found."""
    if not chunk_text:
        return 0, 0
    position = document_text.find(chunk_text, search_from)
    if position < 0:
        position = document_text.find(chunk_text)
    if position < 0:
        return 0, 0
    return position, position + len(chunk_text)


def resolve_section_label(
    chunk_text: str,
    document_text: str,
    spans: list[SectionSpan],
    search_from: int = 0,
) -> tuple[str | None, int]:
    """Return section label with max overlap; second value is next search offset."""
    start, end = _chunk_char_range(chunk_text, document_text, search_from)
    if end <= start or not spans:
        return None, end

    best_label: str | None = None
    best_overlap = 0
    for span in spans:
        overlap_start = max(start, span.start)
        overlap_end = min(end, span.end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = span.label
    return best_label, end


def assign_section_labels(
    units: list[ContentUnit],
    document_text: str,
    spans: list[SectionSpan],
) -> None:
    """Set section_label on each content unit from section spans."""
    search_from = 0
    for unit in units:
        label, search_from = resolve_section_label(
            unit.text, document_text, spans, search_from
        )
        unit.section_label = label


def filter_units_by_target_sections(
    units: list[ContentUnit],
    target_sections: list[str] | None,
) -> list[ContentUnit]:
    """Drop units whose section_label is not in target_sections."""
    if target_sections is None:
        return units
    allowed = {
        section.strip().lower() for section in target_sections if section.strip()
    }
    if not allowed:
        return units
    return [
        unit
        for unit in units
        if unit.section_label is not None and unit.section_label.lower() in allowed
    ]


def should_summarize_unit(
    unit: ContentUnit,
    summarize_sections: list[str] | None,
) -> bool:
    """Whether a unit should be passed through the summarization node."""
    if summarize_sections is None:
        return False
    if not summarize_sections or "*" in summarize_sections:
        return True
    if unit.section_label is None:
        return False
    allowed = {section.strip().lower() for section in summarize_sections}
    return unit.section_label.lower() in allowed
