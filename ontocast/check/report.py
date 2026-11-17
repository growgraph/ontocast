"""Report models for the facts check."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class FactsCheckKind(StrEnum):
    """Kind of a finding; the value is the snake_case member name."""

    NON_CATALOG_VOCABULARY = "non_catalog_vocabulary"
    UNKNOWN_TERM = "unknown_term"
    DOMAIN_VIOLATION = "domain_violation"
    UNTYPED_VALUE = "untyped_value"
    INVALID_LITERAL = "invalid_literal"
    TEMPORAL_RANGE_MISMATCH = "temporal_range_mismatch"
    FUNCTIONAL_VIOLATION = "functional_violation"
    SHACL = "shacl"
    UNTYPED_INSTANCE = "untyped_instance"
    ISOLATED_INSTANCE = "isolated_instance"
    SILENT_INSTANCE = "silent_instance"
    DANGLING_REFERENCE = "dangling_reference"
    TWIN_LABELS = "twin_labels"
    CATALOG_SUBJECT = "catalog_subject"
    SCHEMA_IN_FACTS = "schema_in_facts"


class FactsCheckFinding(BaseModel):
    """One problem found in a facts graph."""

    kind: FactsCheckKind
    severity: Literal["error", "warning"]
    message: str
    subject: str | None = None
    predicate: str | None = None
    value: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    source_shape: str | None = None


class FactsCheckMetrics(BaseModel):
    """Size and shape figures of the checked graph."""

    triples: int
    instances: int
    isolated: int
    untyped: int
    twin_labels: int


class FactsCheckReport(BaseModel):
    """Outcome of a facts check.

    `counts` maps kind value to severity to number of findings and is complete
    even when `findings` is truncated; `findings_total` is the untruncated count.
    """

    findings: list[FactsCheckFinding]
    findings_total: int
    counts: dict[str, dict[str, int]]
    metrics: FactsCheckMetrics
    shacl_evaluated: bool | None
    catalog_empty: bool
