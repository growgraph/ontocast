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
from ontocast.onto.model import OntologyCritiqueReport, Suggestions, TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.tool.atomic import AtomicToolBox
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
async def test_a_rejecting_critic_still_hands_over_its_fixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mechanism must keep working: a rejection reaches the next render."""
    _stub(monkeypatch, _critique(success=False, score=40))
    state = await criticise_ontology_module.criticise_ontology(_unit_state(), _tools())

    assert state.status == Status.FAILED
    assert state.suggestions.actionable_fixes


@pytest.mark.anyio
async def test_an_accepting_critic_leaves_no_stale_suggestions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepting after an evidence search must not carry the rejection forward."""
    state = _unit_state()

    _stub(monkeypatch, _critique(success=False, score=40))
    state = await criticise_ontology_module.criticise_ontology(state, _tools())
    assert state.suggestions.actionable_fixes

    _stub(monkeypatch, _critique(success=True, score=95))
    state = await criticise_ontology_module.criticise_ontology(state, _tools())

    assert state.status == Status.SUCCESS
    assert not state.suggestions.actionable_fixes, (
        "an accepting critic has no outstanding requests; a later render must "
        "not inherit the rejected attempt's suggestions"
    )
    assert not state.suggestions.systemic_critique_summary


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
