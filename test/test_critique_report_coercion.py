"""Free-text coercion on LLM report models.

Providers routinely answer a single-string field with a bulleted list. Rejecting
the whole report over that costs a retry, so the string fields coerce.
"""

import pytest
from pydantic import ValidationError

from ontocast.onto.model import (
    ExternalEvidenceRequest,
    FactsCritiqueReport,
    OntologyCritiqueReport,
    Suggestions,
)

REPORT_CLASSES = [OntologyCritiqueReport, FactsCritiqueReport]


@pytest.mark.parametrize("report_cls", REPORT_CLASSES)
def test_systemic_critique_summary_accepts_a_list(report_cls) -> None:
    """The exact payload from issue #50 — gpt-5-nano returns a list of bullets."""
    report = report_cls(
        success=True,
        score=72,
        systemic_critique_summary=[
            "Strengths observed: the ontology covers core IIoT concepts.",
            "Key gaps: time-series value representation is not coherent.",
        ],
    )

    assert report.systemic_critique_summary == (
        "Strengths observed: the ontology covers core IIoT concepts.\n"
        "Key gaps: time-series value representation is not coherent."
    )


@pytest.mark.parametrize("report_cls", REPORT_CLASSES)
def test_systemic_critique_summary_passes_strings_through(report_cls) -> None:
    report = report_cls(
        success=False, score=10, systemic_critique_summary="One flat summary."
    )
    assert report.systemic_critique_summary == "One flat summary."


@pytest.mark.parametrize("report_cls", REPORT_CLASSES)
def test_systemic_critique_summary_none_becomes_empty(report_cls) -> None:
    report = report_cls(success=True, score=99, systemic_critique_summary=None)
    assert report.systemic_critique_summary == ""


@pytest.mark.parametrize("report_cls", REPORT_CLASSES)
def test_systemic_critique_summary_drops_blank_entries(report_cls) -> None:
    report = report_cls(
        success=True,
        score=50,
        systemic_critique_summary=["  first  ", "", "   ", "second"],
    )
    assert report.systemic_critique_summary == "first\nsecond"


def test_suggestions_summary_accepts_a_list() -> None:
    assert (
        Suggestions(systemic_critique_summary=["a", "b"]).systemic_critique_summary
        == "a\nb"
    )


def test_external_evidence_rationale_accepts_a_list() -> None:
    request = ExternalEvidenceRequest(
        initiate_search=True, rationale=["needs a unit vocabulary", "no SI coverage"]
    )
    assert request.rationale == "needs a unit vocabulary\nno SI coverage"


def test_score_still_rejects_out_of_range_values() -> None:
    """Coercion is for shape, not for constraints — bounds still apply."""
    with pytest.raises(ValidationError):
        OntologyCritiqueReport(success=True, score=150)
