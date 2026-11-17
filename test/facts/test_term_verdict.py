"""``term_verdict`` decides whether a term is acceptable, without messages."""

import pytest

from ontocast.tool.facts_validation.term_verdict import (
    TermVerdictKind,
    term_verdict,
)
from ontocast.tool.facts_validation.terms import ValidationPolicy

pytestmark = pytest.mark.unit

ONTO = "http://onto.test/o#"
CATALOG = {f"{ONTO}Pump", f"{ONTO}hasPart"}
STANDARD = ValidationPolicy().standard_namespaces()


def _verdict(text: str, *, vocabulary="strict", catalog=None, **kwargs):
    return term_verdict(
        text,
        catalog_terms=CATALOG if catalog is None else catalog,
        known_terms=kwargs.get("known_terms", set()),
        fallback_terms=kwargs.get("fallback_terms", set()),
        declared_namespaces=kwargs.get("declared_namespaces", {ONTO}),
        catalog_namespaces=kwargs.get("catalog_namespaces", {ONTO}),
        standard_namespaces=STANDARD,
        vocabulary=vocabulary,
    )


def test_placeholder_namespace():
    verdict = _verdict("http://example.org/x")
    assert verdict is not None
    assert verdict.kind is TermVerdictKind.PLACEHOLDER_NAMESPACE
    assert verdict.term == "http://example.org/x"


def test_well_known_missing_term():
    verdict = _verdict("http://www.w3.org/ns/sosa/hasValue")
    assert verdict is not None
    assert verdict.kind is TermVerdictKind.WELL_KNOWN_MISSING
    assert verdict.vocabulary_title


def test_well_known_existing_term_passes():
    assert _verdict("http://www.w3.org/ns/sosa/hasResult") is None


def test_catalog_term_passes():
    assert _verdict(f"{ONTO}Pump") is None


def test_closed_namespace_near_miss():
    verdict = _verdict(f"{ONTO}Pumpp")
    assert verdict is not None
    assert verdict.kind is TermVerdictKind.CLOSED_NAMESPACE_MISSING


def test_strict_rejects_foreign_namespace():
    verdict = _verdict("http://other.test/ns#thing", vocabulary="strict")
    assert verdict is not None
    assert verdict.kind is TermVerdictKind.NON_CATALOG_STRICT


def test_open_accepts_foreign_namespace():
    assert _verdict("http://other.test/ns#thing", vocabulary="open") is None


def test_empty_catalog_skips_catalog_verdicts():
    empty = set()
    assert _verdict(f"{ONTO}Pumpp", catalog=empty) is None
    assert _verdict("http://other.test/ns#thing", catalog=empty) is None
    placeholder = _verdict("http://example.org/x", catalog=empty)
    assert placeholder is not None
    assert placeholder.kind is TermVerdictKind.PLACEHOLDER_NAMESPACE
    missing = _verdict("http://www.w3.org/ns/sosa/hasValue", catalog=empty)
    assert missing is not None
    assert missing.kind is TermVerdictKind.WELL_KNOWN_MISSING
