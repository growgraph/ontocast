"""Load versioned section-label schemas from YAML in this package."""

from __future__ import annotations

import re
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


class SectionLabelDef(BaseModel):
    """One canonical section label and heading regex patterns."""

    id: str
    heading_patterns: list[str] = Field(default_factory=list)


class SectionLabelSchema(BaseModel):
    """Domain-specific section label vocabulary."""

    schema_version: str
    id: str
    description: str = ""
    labels: list[SectionLabelDef]

    @property
    def compiled_patterns(self) -> tuple[tuple[str, re.Pattern[str]], ...]:
        compiled: list[tuple[str, re.Pattern[str]]] = []
        for label_def in self.labels:
            for pattern in label_def.heading_patterns:
                compiled.append((label_def.id, re.compile(pattern, re.I)))
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
    stripped = line.strip()
    if stripped.startswith("#"):
        stripped = stripped.lstrip("#").strip()
    stripped = _STRUCTURAL_PREFIX.sub("", stripped).strip()
    return stripped


def match_heading_line(line: str, schema: SectionLabelSchema) -> str | None:
    normalised = normalise_heading_line(line)
    if not normalised or len(normalised) > _MAX_HEADING_LINE_LEN:
        return None
    for label, pattern in schema.compiled_patterns:
        if pattern.match(normalised):
            return label
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

    if document_type_hint and document_type_hint.strip():
        hint_lower = document_type_hint.strip().lower()
        for needle, schema_id in manifest.document_type_hints.items():
            if needle.lower() in hint_lower:
                return schema_id

    return manifest.default_schema


def get_default_section_schema() -> SectionLabelSchema:
    manifest = load_manifest()
    return load_section_label_schema(manifest.default_schema)


def canonical_labels(schema: SectionLabelSchema) -> tuple[str, ...]:
    return tuple(label_def.id for label_def in schema.labels)


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
