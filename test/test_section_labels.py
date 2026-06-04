"""Tests for YAML section label catalog loading."""

import pytest

from ontocast.config.section_labels import (
    clear_section_label_caches,
    load_manifest,
    load_section_label_schema,
    match_heading_line,
    normalise_llm_label,
    normalise_user_section_label,
    resolve_section_schema_id,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_section_label_caches()
    yield
    clear_section_label_caches()


def test_manifest_loads_all_schemas() -> None:
    manifest = load_manifest()
    assert manifest.catalog_version == "1.0"
    assert manifest.default_schema == "academic"
    schema_ids = {entry.id for entry in manifest.schemas}
    assert schema_ids == {
        "academic",
        "financial",
        "legal",
        "clinical",
        "manual",
        "fiction",
        "general",
    }


@pytest.mark.parametrize(
    ("schema_id", "heading", "expected"),
    [
        ("academic", "Introduction", "introduction"),
        ("academic", "Abstract.", "abstract"),
        ("academic", "ABSTRACT", "abstract"),
        ("academic", "II. Results", "results"),
        ("financial", "Risk Factors", "risk_factors"),
        ("financial", "Item 7", "md_and_a"),
        ("legal", "Definitions", "definitions"),
        ("clinical", "Adverse Events", "adverse_events"),
        ("manual", "Troubleshooting", "troubleshooting"),
        ("fiction", "Chapter 3", "chapter"),
        ("general", "Executive Summary", "summary"),
    ],
)
def test_match_heading_line_per_domain(
    schema_id: str, heading: str, expected: str
) -> None:
    schema = load_section_label_schema(schema_id)
    assert match_heading_line(heading, schema) == expected


def test_resolve_section_schema_id_explicit() -> None:
    assert (
        resolve_section_schema_id(
            section_schema_id="financial", document_type_hint=None
        )
        == "financial"
    )


def test_resolve_section_schema_id_from_hint() -> None:
    assert (
        resolve_section_schema_id(
            section_schema_id=None,
            document_type_hint="SEC 10-Q filing",
        )
        == "financial"
    )
    assert (
        resolve_section_schema_id(
            section_schema_id=None,
            document_type_hint="clinical trial protocol",
        )
        == "clinical"
    )


def test_resolve_section_schema_id_defaults_to_academic() -> None:
    assert (
        resolve_section_schema_id(section_schema_id=None, document_type_hint=None)
        == "academic"
    )


def test_normalise_user_section_label_cross_schema() -> None:
    assert normalise_user_section_label("risk_factors") == "risk_factors"
    assert normalise_user_section_label("md_and_a", schema_id="financial") == "md_and_a"


def test_normalise_llm_label_validates_against_schema() -> None:
    schema = load_section_label_schema("academic")
    assert normalise_llm_label("Results", schema) == "results"
    assert normalise_llm_label("not_a_section", schema) is None
