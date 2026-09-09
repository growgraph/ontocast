"""Free-text coercion on LLM report models.

Providers routinely answer a single-string field with a bulleted list. Rejecting
the whole report over that costs a retry, so the string fields coerce.
"""

import pytest
from pydantic import ValidationError

from ontocast.onto.model import (
    ExternalEvidencePlan,
    ExternalEvidenceRequest,
    FactsCritiqueReport,
    OntologyCritiqueReport,
    Suggestions,
    TripleFix,
)

pytestmark = pytest.mark.unit

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


def test_external_evidence_plan_rationale_accepts_a_list() -> None:
    """The plan's rationale coerces like the request's, not unlike it."""
    plan = ExternalEvidencePlan(
        should_search=True, rationale=["term is ambiguous", "no catalog match"]
    )
    assert plan.rationale == "term is ambiguous\nno catalog match"


def test_triple_fix_required_free_text_accepts_a_list() -> None:
    """``text_fragment``/``explanation`` are required, so a list used to raise.

    Both are prose the model writes, and both are the shape providers bullet.
    Rejecting them discarded every fix in the report, not just the one field.
    """
    fix = TripleFix(
        text_fragment=["The sample was annealed at 350 C", "for two hours."],
        action="ADD",
        severity="important",
        explanation=["Missing datatype.", "Temporal literal needs xsd:date."],
    )

    assert fix.text_fragment == "The sample was annealed at 350 C\nfor two hours."
    assert fix.explanation == "Missing datatype.\nTemporal literal needs xsd:date."


def test_triple_fix_graph_syntax_fields_are_not_coerced() -> None:
    """Graph-payload fields stay strict — joining a list would corrupt them."""
    with pytest.raises(ValidationError):
        # Via model_validate: the point is an ill-typed payload, which the
        # constructor signature correctly refuses to express.
        TripleFix.model_validate(
            {
                "text_fragment": "quote",
                "action": "REPLACE",
                "severity": "minor",
                "explanation": "why",
                "correct_value": ["ex:a ex:b ex:c .", "ex:d ex:e ex:f ."],
            }
        )


def test_score_still_rejects_out_of_range_values() -> None:
    """Coercion is for shape, not for constraints — bounds still apply."""
    with pytest.raises(ValidationError):
        OntologyCritiqueReport(success=True, score=150)
