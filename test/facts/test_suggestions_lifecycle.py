"""Critic suggestions must not outlive the render they were raised against.

``state.suggestions`` is written by the critic and read by the next render.
Nothing used to clear it, so once the critic rejected a unit its suggestions
were carried into every later render of that unit -- including the
finding-driven repair render, which then ran under two contradictory contracts
in one prompt: the improvement template's "Critic's suggestions are advisory
... proactively identify and fix additional problems not mentioned in the
critique", and the findings block's "apply every item, rewrite in place, never
delete".

The leak is unreachable at MAX_VISITS=1, where the critic never runs, and that
is exactly the arm split observed in practice: the two-visit arm lost
graph connectivity and gained validation errors while the one-visit arm did not.

Two writers had to be fixed, because the loop can reach the repair with stale
suggestions by two different routes:

* a render consumes them (``render_facts_update``), and
* the critic *accepts* on a later attempt of the same render, after an
  external-evidence search, with no render in between to consume them.
"""

import importlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import Literal, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.model import FactsCritiqueReport, Suggestions, TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.tool.atomic import AtomicToolBox
from test.snapshot_helpers import empty_snapshot

criticise_facts_module = importlib.import_module("ontocast.agent.criticise_facts")

pytestmark = pytest.mark.unit


def _unit_state() -> UnitFactsState:
    graph = RDFGraph()
    subject = URIRef(f"{DEFAULT_IRI}sample_1")
    graph.add(
        (
            subject,
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
            Literal("sample"),
        )
    )
    unit = ContentUnit(
        text="a shift of 96 meV",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=graph,
    )
    state = UnitFactsState(content_unit=unit, ontology_snapshot=empty_snapshot())
    state.status = Status.SUCCESS
    return state


async def _llm_tool(_budget_tracker) -> SimpleNamespace:
    """Stand-in for the awaited LLM tool; the response is stubbed downstream."""
    return SimpleNamespace()


def _tools() -> AtomicToolBox:
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            get_llm_tool=_llm_tool,
            web_grounding_enabled_for_node=lambda _node: False,
            object_property_literal_check=True,
            property_alias_min_ratio=0.85,
            code_predicates=(),
            citation_vocabulary={},
            quantity_fallback_vocabulary=None,
            additional_standard_namespaces=(),
            validation_policy=None,
            acceptance_policy=None,
        ),
    )


def _critique(
    *, success: bool, score: float, severity: str = "critical"
) -> FactsCritiqueReport:
    """Build a critique whose *fixes* decide the verdict.

    Acceptance now reads the fix severities, not the score, so a test that
    wants a rejection has to supply a blocking fix rather than a low number.
    """
    return FactsCritiqueReport(
        success=success,
        score=score,
        actionable_triple_fixes=[
            TripleFix(
                action="REPLACE",
                severity=severity,
                explanation="use the canonical scalar property",
                text_fragment="a shift of 96 meV",
            )
        ],
        systemic_critique_summary="tighten the value encoding",
    )


def _stub_critique(monkeypatch: pytest.MonkeyPatch, critique: FactsCritiqueReport):
    async def fake_call_llm_with_retry(*args, **kwargs):
        return critique

    monkeypatch.setattr(
        criticise_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )


@pytest.mark.anyio
async def test_a_rejecting_critic_records_suggestions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mechanism must keep working: a rejection still reaches the render."""
    _stub_critique(monkeypatch, _critique(success=False, score=55))
    state = await criticise_facts_module.criticise_facts(_unit_state(), _tools())

    assert state.status == Status.FAILED
    assert state.suggestions.actionable_fixes, (
        "a rejecting critic must hand its fixes to the next render"
    )


@pytest.mark.anyio
async def test_an_accepting_critic_leaves_no_suggestions_for_the_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exit path that has no render between rejection and repair.

    A critic that rejects, requests an evidence search, and then accepts on the
    retry reaches ``_run_finding_driven_repair`` directly. Without this reset
    the repair render inherits the rejected attempt's suggestions.
    """
    state = _unit_state()

    _stub_critique(monkeypatch, _critique(success=False, score=55))
    state = await criticise_facts_module.criticise_facts(state, _tools())
    assert state.suggestions.actionable_fixes

    _stub_critique(monkeypatch, _critique(success=True, score=98, severity="minor"))
    state = await criticise_facts_module.criticise_facts(state, _tools())

    assert state.status == Status.SUCCESS
    assert not state.suggestions.actionable_fixes, (
        "an accepting critic has no outstanding requests; the repair render "
        "must not inherit the rejected attempt's suggestions"
    )
    assert not state.suggestions.systemic_critique_summary


def test_empty_suggestions_render_no_improvement_instruction() -> None:
    """The reset must actually silence the improvement block, not just empty it.

    ``render_suggestions_prompt`` returns the whole "advisory ... think
    independently ... proactively fix additional problems" template whenever
    either field is non-empty, so a reset that left one populated would keep
    the contradiction alive.
    """
    from ontocast.agent.common import render_suggestions_prompt
    from ontocast.onto.enum import WorkflowNode

    rendered = render_suggestions_prompt(Suggestions(), WorkflowNode.TEXT_TO_FACTS)
    assert rendered == ""
