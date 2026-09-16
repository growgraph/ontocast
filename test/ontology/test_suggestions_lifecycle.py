"""The ontology path had the facts loop's stale-suggestion leak, uncaught.

``state.suggestions`` is written by the critic and read by the next render.
Neither ``render_ontology_update`` nor ``criticise_ontology``'s accept branch
cleared it, so once the critic rejected a unit its suggestions rode along into
every later render of that unit.

The facts loop shipped the same defect and its CHANGELOG entry records where it
landed: the two-visit arm lost graph connectivity and gained validation errors
while the one-visit arm did not. The ontology path is worse off in one respect --
it has no finding-driven repair pass, so nothing downstream would have noticed.

Also pinned here: ``update_ontology()`` returning whether it actually applied.
It used to return ``None`` and swallow a budget rejection, so a render whose
entire output was discarded still reported SUCCESS with an unchanged graph.
"""

import importlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDFS, Literal, URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.model import (
    OntologyCritiqueReport,
    OntologyUnitFinding,
    OntologyUnitFindingKind,
    Suggestions,
    TripleFix,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import FactsAcceptancePolicy
from test.snapshot_helpers import empty_snapshot

criticise_ontology_module = importlib.import_module("ontocast.agent.criticise_ontology")

pytestmark = pytest.mark.unit

_CLASS = URIRef("https://example.com/onto#Sample")


def _graph_of(*triples) -> RDFGraph:
    graph = RDFGraph()
    for triple in triples:
        graph.add(triple)
    return graph


def _unit_state(**kwargs) -> UnitOntologyState:
    graph = RDFGraph()
    graph.add((_CLASS, RDFS.label, Literal("Sample")))
    unit = ContentUnit(
        text="a sample of perovskite",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=RDFGraph(),
    )
    state = UnitOntologyState(
        content_unit=unit,
        ontology_snapshot=empty_snapshot(),
        **kwargs,
    )
    state.working_graph = graph
    state.status = Status.SUCCESS
    return state


async def _llm_tool(_budget_tracker) -> SimpleNamespace:
    return SimpleNamespace()


def _tools() -> AtomicToolBox:
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            get_llm_tool=_llm_tool,
            web_grounding_enabled_for_node=lambda _node: False,
            ontology_acceptance_policy=FactsAcceptancePolicy(
                blocking_finding_kinds=frozenset(
                    {
                        "foreign_delete",
                        "foreign_namespace",
                        "subclass_cycle",
                        "role_confusion",
                    }
                )
            ),
        ),
    )


def _critique(*, success: bool, score: float) -> OntologyCritiqueReport:
    return OntologyCritiqueReport(
        success=success,
        score=score,
        actionable_ontology_fixes=[
            TripleFix(
                text_fragment="a sample of perovskite",
                action="ADD",
                severity="important",
                explanation="declare rdfs:comment on the new class",
            )
        ],
        systemic_critique_summary="labels are thin",
    )


def _stub(monkeypatch: pytest.MonkeyPatch, critique: OntologyCritiqueReport) -> None:
    async def fake_call_llm_with_retry(*args, **kwargs):
        return critique

    monkeypatch.setattr(
        criticise_ontology_module, "call_llm_with_retry", fake_call_llm_with_retry
    )


@pytest.mark.anyio
async def test_a_low_score_alone_no_longer_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate is the deterministic findings, not the model's self-assessment.

    The incumbent rule rejected anything scoring 90 or below, while the prompt's
    own rubric calls 70-89 "Good" -- so it rejected ontologies its instructions
    considered good, and did it on a number the model was never shown a scale
    for. The verdict is still recorded for comparison.
    """
    _stub(monkeypatch, _critique(success=False, score=40))
    state = await criticise_ontology_module.criticise_ontology(_unit_state(), _tools())

    assert state.status == Status.SUCCESS
    assert state.attempt_log[-1].incumbent_accepted is False
    assert state.attempt_log[-1].score == 40


@pytest.mark.anyio
async def test_a_blocking_finding_rejects_whatever_the_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destructive delta is a defect the critic's optimism cannot override."""
    state = _unit_state()
    state.deterministic_findings = [
        OntologyUnitFinding(
            kind=OntologyUnitFindingKind.FOREIGN_DELETE,
            mandatory=True,
            message="deletes catalog statements",
        )
    ]

    _stub(monkeypatch, _critique(success=True, score=95))
    state = await criticise_ontology_module.criticise_ontology(state, _tools())

    assert state.status == Status.FAILED
    assert state.attempt_log[-1].accept_reason == "mandatory_findings"
    assert state.attempt_log[-1].incumbent_accepted is True


@pytest.mark.anyio
async def test_an_accepting_critic_keeps_its_own_fixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance says the unit may leave, not that the critique is worthless.

    Clearing here discarded every fix attached to an accepted render -- which,
    since a REMOVE can never by itself cause a rejection, was most of them. The
    patch pass reads exactly this field.
    """
    state = _unit_state()

    _stub(monkeypatch, _critique(success=True, score=95))
    state = await criticise_ontology_module.criticise_ontology(state, _tools())

    assert state.status == Status.SUCCESS
    assert state.suggestions.actionable_fixes
    assert state.suggestions.systemic_critique_summary == "labels are thin"


def test_update_ontology_reports_that_it_applied() -> None:
    state = _unit_state()
    state.ontology_updates.append(
        GraphUpdate(
            triple_operations=[
                TripleOp(
                    type="insert",
                    graph=_graph_of(
                        (_CLASS, RDFS.comment, Literal("A material sample"))
                    ),
                )
            ]
        )
    )

    assert state.update_ontology() is True
    assert (_CLASS, RDFS.comment, None) in state.working_graph
    assert state.ontology_updates == []


def test_update_ontology_reports_a_budget_rejection_instead_of_swallowing_it() -> None:
    """A discarded update must be distinguishable from an applied one.

    It used to return ``None`` either way, so ``render_ontology_update`` reported
    SUCCESS on a render whose entire output the ONTOLOGY_MAX_TRIPLES backstop had
    thrown away -- and the working graph still held the pre-render content.
    """
    state = _unit_state(ontology_max_triples=1)
    before = len(state.working_graph)
    state.ontology_updates.append(
        GraphUpdate(
            triple_operations=[
                TripleOp(
                    type="insert",
                    graph=_graph_of(
                        (_CLASS, RDFS.comment, Literal("A material sample")),
                        (_CLASS, RDFS.seeAlso, URIRef("https://example.com/x")),
                    ),
                )
            ]
        )
    )

    assert state.update_ontology() is False
    assert len(state.working_graph) == before, "the working graph must be untouched"
    assert state.ontology_updates, "rejected updates stay pending, not applied"


def test_no_suggestions_renders_no_improvement_instruction() -> None:
    """The reset must silence the improvement block, not merely empty a field."""
    from ontocast.agent.common import render_suggestions_prompt
    from ontocast.onto.enum import WorkflowNode

    assert render_suggestions_prompt(Suggestions(), WorkflowNode.TEXT_TO_ONTOLOGY) == ""
