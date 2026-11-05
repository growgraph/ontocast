"""Load versioned section-label schemas from YAML in this package."""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from functools import lru_cache
from importlib import resources
from importlib.resources.abc import Traversable

import yaml
from pydantic import BaseModel, Field

_MAX_HEADING_LINE_LEN = 120
_STRUCTURAL_PREFIX = re.compile(
    r"^(?:chapter|section|part)\s+(?:\d+|[IVXivx]+)(?:\.\d+)*\s*[:.\-–—)]\s*",
    re.I,
)

# Publishers decorate headings with glyphs and bullets ("■ REFERENCES",
# "*sı Supporting Information"), and docling carries them through to the
# markdown export. Strip any leading run of non-word characters, keeping "("
# and "[" so bracketed numbering survives for the numbering pass.
_LEADING_DECORATION = re.compile(r"^[^\w(\[]+")
_TRAILING_DECORATION = re.compile(r"[^\w)\]]+$")
_MARKDOWN_EMPHASIS = re.compile(r"[*_`]{1,3}")

# Bare digit numbering is stripped unconditionally ("2.1 Synthesis of ...",
# "1 Introduction"). Single letters and roman numerals require a trailing
# separator, or "I Introduction" and "A Framework" would lose their first word.
_DIGIT_NUMBERING = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+")
_ALPHA_NUMBERING = re.compile(r"^(?:[A-Za-z]|[IVXLivxl]+)[.)]\s+")

# Superscript/footnote artefacts left by two-column PDF extraction, e.g. the
# "sı" in "*sı Supporting Information".
_ARTEFACT_TOKEN = re.compile(r"^(?:sı|si|s)\s+(?=[A-Za-z])", re.I)


# The catalog holds more patterns than ``re``'s own cache, so schema detection,
# which matches every schema, would otherwise recompile them on every heading.
@lru_cache(maxsize=None)
def _compile_heading_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


@lru_cache(maxsize=None)
def _compile_keyword(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.I)


class RouteClass(StrEnum):
    """What a section of this label holds, for routing before extraction.

    ``content`` sections are extracted. ``boilerplate`` sections are dropped
    by default when the label is strong and the text states no measurement.
    ``non_content`` sections (front or back matter: author blocks, notes,
    availability and licence statements) are flagged, or dropped, per
    ``CHUNK_NON_CONTENT_MODE`` under the same conditions. ``citations``
    sections are reference lists, routed by ``CHUNK_BIBLIOGRAPHY_MODE`` when
    their content agrees. The class never decides a drop on its own.
    """

    CONTENT = "content"
    BOILERPLATE = "boilerplate"
    NON_CONTENT = "non_content"
    CITATIONS = "citations"


class SectionLabelDef(BaseModel):
    """One canonical section label, its heading patterns and recall keywords.

    ``heading_patterns`` are high-precision anchored regexes. ``keywords`` are
    the recall tier: whole-word phrases that identify the label inside a
    compound or decorated heading ("Results and Discussion", "Experimental
    Section") which the anchored patterns cannot match.
    """

    id: str
    route_class: RouteClass = RouteClass.CONTENT
    terminal: bool = Field(
        default=False,
        description=(
            "A flat list with no subsections (a reference list): descriptive "
            "headings after it never inherit the label, and a keyword names it "
            "only as a whole conjunct of a heading ('Notes and References'), "
            "never inside a longer phrase ('Comparison with literature "
            "references')."
        ),
    )
    heading_patterns: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    order: int | None = Field(
        default=None,
        description=(
            "Canonical position of this section in a well-formed document of "
            "this type. Used only to refuse label fills that would run "
            "backwards; absent means the label is not order-constrained."
        ),
    )

    @property
    def compiled_keywords(self) -> tuple[re.Pattern[str], ...]:
        return tuple(_compile_keyword(keyword) for keyword in self.keywords)


class SectionLabelSchema(BaseModel):
    """Domain-specific section label vocabulary for one document type.

    The catalog of schemas is a **partition**: every document belongs to exactly
    one cell, with ``general`` as the residual. ``document_profile`` is what
    makes that partition checkable -- if two profiles could describe the same
    document, the cells overlap and one of them is wrong. It is also the text
    the content-based detector embeds, so it describes the *document type*,
    unlike ``description``, which describes this schema's headings.
    """

    schema_version: str
    id: str
    description: str = ""
    document_profile: str = Field(
        default="",
        description=(
            "One sentence describing the kind of document this schema covers, "
            "written to be true of no other schema in the catalog. Empty means "
            "the schema is not a detection candidate."
        ),
    )
    parent: str | None = Field(
        default=None,
        description=(
            "Reserved for a future document-type hierarchy (e.g. a thesis as a "
            "sub-type of academic). Unused today: the catalog is flat, and "
            "sub-types would blur the cells a flat detector must separate."
        ),
    )
    labels: list[SectionLabelDef]
    ordered: bool = Field(
        default=False,
        description=(
            "Whether this document type has a canonical section order, making "
            "the per-label 'order' values meaningful for fill guarding."
        ),
    )

    @property
    def compiled_patterns(self) -> tuple[tuple[str, re.Pattern[str]], ...]:
        compiled: list[tuple[str, re.Pattern[str]]] = []
        for label_def in self.labels:
            for pattern in label_def.heading_patterns:
                compiled.append((label_def.id, _compile_heading_pattern(pattern)))
        return tuple(compiled)


class SchemaManifestEntry(BaseModel):
    id: str
    file: str


class SectionLabelManifest(BaseModel):
    catalog_version: str
    default_schema: str
    schemas: list[SchemaManifestEntry]
    document_type_hints: dict[str, str] = Field(default_factory=dict)


def _labels_dir() -> Traversable:
    return resources.files(__package__)


def normalise_heading_line(line: str) -> str:
    """Reduce a raw heading line to its bare section name.

    Strips markdown syntax, publisher decoration glyphs, extraction artefacts
    and section numbering, so that "## ■ REFERENCES" and "2.1 Synthesis of
    films" reach the matchers as "REFERENCES" and "Synthesis of films".
    """
    stripped = unicodedata.normalize("NFKC", line).strip()
    if stripped.startswith("#"):
        stripped = stripped.lstrip("#").strip()
    stripped = _MARKDOWN_EMPHASIS.sub("", stripped).strip()
    stripped = _LEADING_DECORATION.sub("", stripped)
    stripped = _TRAILING_DECORATION.sub("", stripped)
    stripped = _ARTEFACT_TOKEN.sub("", stripped)
    stripped = _STRUCTURAL_PREFIX.sub("", stripped).strip()
    stripped = _DIGIT_NUMBERING.sub("", stripped)
    stripped = _ALPHA_NUMBERING.sub("", stripped)
    return stripped.strip()


def match_heading_line(line: str, schema: SectionLabelSchema) -> str | None:
    """Match a heading against the schema's anchored patterns (high precision).

    Deliberately exact: this function also gates segment coalescing, where a
    fuzzy match on a body first line would join distinct sections. Recall lives
    in :func:`match_heading_keywords`.
    """
    normalised = normalise_heading_line(line)
    if not normalised or len(normalised) > _MAX_HEADING_LINE_LEN:
        return None
    for label, pattern in schema.compiled_patterns:
        if pattern.match(normalised):
            return label
    return None


# Coordination splits a compound heading into conjuncts ("Results and
# Discussion"); the first names the section.
_CONJUNCTION = re.compile(r"\s+(?:and|&)\s+|\s*/\s*|\s*,\s*", re.I)
# A prepositional phrase follows the head noun ("Summary of Results").
_PREPOSITION = re.compile(
    r"\s+(?:of|for|in|on|with|from|to|by|via|under|at|using|over|versus|vs\.?)\s+",
    re.I,
)
# "Head: subtitle" and "Head - subtitle" headings.
_SUBTITLE = re.compile(r"\s*(?::|\s[-–—]\s)\s*")


def heading_head(normalised: str) -> str | None:
    """The head of a subtitled heading ("Appendix A: Derivation ..." -> "Appendix A").

    Returns:
        The part before the first colon or spaced dash, or ``None`` when the
        heading has no subtitle.
    """
    parts = _SUBTITLE.split(normalised, maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return None
    return parts[0].strip()


def _conjuncts(normalised: str) -> list[str]:
    return [part.strip() for part in _CONJUNCTION.split(normalised) if part.strip()]


def _head_phrase(normalised: str) -> str:
    """The first conjunct, cut before its first prepositional phrase."""
    conjuncts = _conjuncts(normalised)
    first = conjuncts[0] if conjuncts else normalised
    return _PREPOSITION.split(first, maxsplit=1)[0].strip()


def match_heading_keywords(line: str, schema: SectionLabelSchema) -> str | None:
    """Match a heading by keyword, for compound and non-canonical headings.

    Resolution follows the heading's head noun, in order:

    1. a terminal label whose keyword is a whole conjunct ("Notes and
       References" is references); a terminal label matches no other way;
    2. a keyword ending the head phrase -- the first conjunct cut before its
       first preposition ("Characterization Results" is results, "Summary of
       Results" a summary, "Results and Discussion" results);
    3. any keyword in the head phrase, rightmost first ("Experimental
       Section" is methods);
    4. the earliest keyword anywhere in the heading.

    Ties go to the longer keyword, then to schema order. English noun-phrase
    structure is assumed, as the keywords themselves are English.

    Args:
        line: Raw heading line.
        schema: Active section label schema.

    Returns:
        The label, or ``None`` when no keyword matches.
    """
    normalised = normalise_heading_line(line)
    if not normalised or len(normalised) > _MAX_HEADING_LINE_LEN:
        return None

    conjuncts = _conjuncts(normalised)
    for label_def in schema.labels:
        if label_def.terminal and any(
            pattern.fullmatch(conjunct)
            for pattern in label_def.compiled_keywords
            for conjunct in conjuncts
        ):
            return label_def.id

    head = _head_phrase(normalised)
    ending: tuple[int, int, str] | None = None
    inside: tuple[int, int, int, str] | None = None
    anywhere: tuple[int, int, int, str] | None = None
    for index, label_def in enumerate(schema.labels):
        if label_def.terminal:
            continue
        for pattern in label_def.compiled_keywords:
            in_head = pattern.search(head)
            if in_head is not None:
                length = in_head.end() - in_head.start()
                if in_head.end() == len(head):
                    candidate = (-length, index, label_def.id)
                    if ending is None or candidate < ending:
                        ending = candidate
                inside_candidate = (-in_head.end(), -length, index, label_def.id)
                if inside is None or inside_candidate < inside:
                    inside = inside_candidate
            found = pattern.search(normalised)
            if found is not None:
                any_candidate = (
                    found.start(),
                    -(found.end() - found.start()),
                    index,
                    label_def.id,
                )
                if anywhere is None or any_candidate < anywhere:
                    anywhere = any_candidate
    if ending is not None:
        return ending[2]
    if inside is not None:
        return inside[3]
    if anywhere is not None:
        return anywhere[3]
    return None


def match_heading_name(line: str, schema: SectionLabelSchema) -> str | None:
    """Match a heading that consists of one of a label's keywords and nothing else.

    "Author Contributions" *is* a name of the acknowledgements section, as
    exactly as an anchored pattern; "Funding Mechanisms" only contains one.
    """
    normalised = normalise_heading_line(line)
    if not normalised or len(normalised) > _MAX_HEADING_LINE_LEN:
        return None
    for label_def in schema.labels:
        for pattern in label_def.compiled_keywords:
            if pattern.fullmatch(normalised):
                return label_def.id
    return None


def resolve_heading_label(
    line: str, schema: SectionLabelSchema
) -> tuple[str, str] | None:
    """Resolve a heading to a label via patterns, then keywords.

    A heading that is a keyword and nothing else resolves as a pattern.

    Returns:
        ``(label, source)`` where source is ``"heading_pattern"`` or
        ``"heading_keyword"``; ``None`` when the heading is unrecognised.
    """
    exact = match_heading_line(line, schema) or match_heading_name(line, schema)
    if exact is not None:
        return exact, "heading_pattern"
    keyword = match_heading_keywords(line, schema)
    if keyword is not None:
        return keyword, "heading_keyword"
    return None


def label_order(label: str, schema: SectionLabelSchema) -> int | None:
    """Canonical position of a label in this schema, when order-constrained."""
    if not schema.ordered:
        return None
    for label_def in schema.labels:
        if label_def.id == label:
            return label_def.order
    return None


@lru_cache(maxsize=1)
def load_manifest() -> SectionLabelManifest:
    path = _labels_dir() / "manifest.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SectionLabelManifest.model_validate(raw)


@lru_cache(maxsize=16)
def load_section_label_schema(schema_id: str) -> SectionLabelSchema:
    manifest = load_manifest()
    entry = next((item for item in manifest.schemas if item.id == schema_id), None)
    if entry is None:
        known = ", ".join(item.id for item in manifest.schemas)
        raise ValueError(f"Unknown section schema {schema_id!r}; known: {known}")
    path = _labels_dir() / entry.file
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    schema = SectionLabelSchema.model_validate(raw)
    if schema.id != schema_id:
        raise ValueError(
            f"Schema file {entry.file} has id {schema.id!r}, expected {schema_id!r}"
        )
    return schema


@lru_cache(maxsize=1)
def _hint_matchers() -> tuple[tuple[re.Pattern[str], str], ...]:
    """Hint needles as word-boundary patterns, most specific first.

    Bare substring matching mis-fires on short needles: ``epo`` matches
    "r*epo*rt", ``paper`` matches "news*paper*", and ``novel`` matches "*novel*
    materials study" -- sending an academic paper to the fiction schema. Word
    boundaries make each needle match only whole words.

    Longest needle first so the most specific hint wins ("quarterly report"
    over "report"-length needles) rather than whichever the YAML happens to
    list first.
    """
    matchers = []
    for needle, schema_id in load_manifest().document_type_hints.items():
        cleaned = needle.strip().lower()
        if not cleaned:
            continue
        matchers.append((cleaned, schema_id))
    matchers.sort(key=lambda item: (-len(item[0]), item[0]))
    return tuple(
        (re.compile(rf"\b{re.escape(needle)}\b"), schema_id)
        for needle, schema_id in matchers
    )


def schema_id_from_hint(document_type_hint: str | None) -> str | None:
    """Schema a free-text document-type hint maps to, or ``None`` if it maps to none.

    Distinct from :func:`resolve_section_schema_id`, which cannot express "the
    caller told us nothing": it returns the manifest default both for an
    unmatched hint and for no hint at all. Automatic detection must run in
    exactly those cases, so it needs this finer answer.
    """
    if not document_type_hint or not document_type_hint.strip():
        return None
    hint_lower = document_type_hint.strip().lower()
    for pattern, schema_id in _hint_matchers():
        if pattern.search(hint_lower):
            return schema_id
    return None


def resolve_section_schema_id(
    *,
    section_schema_id: str | None = None,
    document_type_hint: str | None = None,
) -> str:
    """Pick schema: explicit id, then hint substring match, then manifest default."""
    manifest = load_manifest()
    if section_schema_id and section_schema_id.strip():
        schema_id = section_schema_id.strip().lower()
        load_section_label_schema(schema_id)
        return schema_id

    from_hint = schema_id_from_hint(document_type_hint)
    if from_hint is not None:
        return from_hint

    return manifest.default_schema


def get_default_section_schema() -> SectionLabelSchema:
    manifest = load_manifest()
    return load_section_label_schema(manifest.default_schema)


def canonical_labels(schema: SectionLabelSchema) -> tuple[str, ...]:
    return tuple(label_def.id for label_def in schema.labels)


def label_route_class(label: str | None, schema: SectionLabelSchema) -> RouteClass:
    """Route class of a label in this schema; ``content`` for unknown labels."""
    if label is None:
        return RouteClass.CONTENT
    wanted = label.strip().lower()
    for label_def in schema.labels:
        if label_def.id == wanted:
            return label_def.route_class
    return RouteClass.CONTENT


def label_is_terminal(label: str | None, schema: SectionLabelSchema) -> bool:
    """Whether a label names a flat list that no subsection can belong to."""
    if label is None:
        return False
    return any(
        label_def.terminal for label_def in schema.labels if label_def.id == label
    )


def labels_of_class(schema: SectionLabelSchema, route_class: RouteClass) -> list[str]:
    """Label ids of one route class, in schema order."""
    return [
        label_def.id
        for label_def in schema.labels
        if label_def.route_class is route_class
    ]


@lru_cache(maxsize=1)
def all_known_label_ids() -> frozenset[str]:
    manifest = load_manifest()
    ids: set[str] = set()
    for entry in manifest.schemas:
        schema = load_section_label_schema(entry.id)
        ids.update(canonical_labels(schema))
    return frozenset(ids)


def normalise_llm_label(raw: str | None, schema: SectionLabelSchema) -> str | None:
    if raw is None:
        return None
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    allowed = set(canonical_labels(schema))
    if cleaned in allowed:
        return cleaned
    return None


def normalise_user_section_label(
    raw: str,
    *,
    schema_id: str | None = None,
) -> str | None:
    """Map user-supplied section name to a canonical label."""
    if raw.strip() == "*":
        return "*"

    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in all_known_label_ids():
        return cleaned

    resolved_id = resolve_section_schema_id(
        section_schema_id=schema_id,
        document_type_hint=None,
    )
    schema = load_section_label_schema(resolved_id)
    if cleaned in canonical_labels(schema):
        return cleaned

    matched = match_heading_line(raw, schema)
    if matched is not None:
        return matched

    for entry in load_manifest().schemas:
        other = load_section_label_schema(entry.id)
        matched = match_heading_line(raw, other)
        if matched is not None:
            return matched

    return None


def clear_section_label_caches() -> None:
    """Clear loader caches (for tests)."""
    load_manifest.cache_clear()
    load_section_label_schema.cache_clear()
    all_known_label_ids.cache_clear()
    _hint_matchers.cache_clear()
