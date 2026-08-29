"""The ontology critic's ledger, recorded before its gate is recalibrated.

The facts gate was replaced on numbers (28/34 rejected, median score 79). For
the ontology critic no numbers exist -- it has never run on recorded data
(every arm ran ``render_mode: facts``) -- so the loop is instrumented first:
each critic call appends a :class:`LoopAttempt`, the deterministic delta
findings are collected and injected into the critic prompt (shadow mode), and
the gate itself stays ``success or score > 90`` until a sampling run yields a
distribution to replace it from.
"""

import importlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode, OntologyContextMode, Status
from ontocast.onto.model import (
    OntologyCritiqueReport,
    OntologyUnitFinding,
    OntologyUnitFindingKind,
    TripleFix,
)
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.stategraph import atomic as unit_loops
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.toolbox import ToolBox

criticise_ontology_module = importlib.import_module("ontocast.agent.criticise_ontology")

pytestmark = pytest.mark.unit

ONTO = "https://example.com/onto#"
_CLASS = URIRef(f"{ONTO}Sample")


def _snapshot_graph() -> RDFGraph:
    graph = RDFGraph()
    graph.bind("onto", ONTO)
    graph.add((_CLASS, RDF.type, OWL.Class))
    graph.add((_CLASS, RDFS.label, Literal("Sample")))
    return graph


def _unit_state() -> UnitOntologyState:
    unit = ContentUnit(
        text="a sample of perovskite",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=RDFGraph(),
    )
    state = UnitOntologyState(
        content_unit=unit,
        ontology_snapshot=OntologySnapshot(
            graph=_snapshot_graph(), source_iris=[f"{ONTO.rstrip('#')}"]
        ),
    )
    state.working_graph = _snapshot_graph()
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


def _critique(
    *, success: bool, score: float, fixes: list[TripleFix] | None = None
) -> OntologyCritiqueReport:
    default_fixes = [
        TripleFix(
            text_fragment="a sample of perovskite",
            action="REPLACE",
            severity="critical",
            explanation="rework the hierarchy",
            incorrect_value="onto:Sample rdfs:label 'Sample' .",
        ),
        TripleFix(
            text_fragment="a sample of perovskite",
            action="ADD",
            severity="important",
            explanation="declare a comment",
        ),
    ]
    return OntologyCritiqueReport(
        success=success,
        score=score,
        actionable_ontology_fixes=default_fixes if fixes is None else fixes,
        systemic_critique_summary="hierarchy is thin",
    )


def _stub(
    monkeypatch: pytest.MonkeyPatch,
    critique: OntologyCritiqueReport,
    captured: dict | None = None,
) -> None:
    async def fake_call_llm_with_retry(*args, **kwargs):
        if captured is not None:
            captured.update(kwargs.get("prompt_kwargs", {}))
        return critique

    monkeypatch.setattr(
        criticise_ontology_module, "call_llm_with_retry", fake_call_llm_with_retry
    )


@pytest.mark.anyio
async def test_a_rejection_is_recorded_with_its_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, _critique(success=False, score=72))
    state = _unit_state()
    state.deterministic_findings = [
        OntologyUnitFinding(
            kind=OntologyUnitFindingKind.MISSING_LABEL,
            message="new term has no label",
        )
    ]

    state = await criticise_ontology_module.criticise_ontology(state, _tools())

    assert state.status == Status.FAILED
    [attempt] = state.attempt_log
    assert attempt.kind == "critic"
    assert attempt.score == 72
    assert attempt.success is False
    assert attempt.accept_reason == "incumbent_rejected"
    assert attempt.severity_counts == {"critical": 1, "important": 1}
    assert attempt.n_actionable_fixes == 2
    assert attempt.n_deterministic_findings == 1
    assert attempt.n_mandatory_findings == 1
    # The critical fix rewrites onto:Sample, which the snapshot declares and
    # the (empty) delta never touched -- the critic litigating catalog content.
    assert attempt.n_fixes_targeting_snapshot == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("success", "score", "reason"),
    [(True, 40, "incumbent_success"), (False, 95, "incumbent_score")],
)
async def test_an_acceptance_records_which_clause_fired(
    monkeypatch: pytest.MonkeyPatch, success: bool, score: float, reason: str
) -> None:
    _stub(monkeypatch, _critique(success=success, score=score))
    state = await criticise_ontology_module.criticise_ontology(_unit_state(), _tools())

    assert state.status == Status.SUCCESS
    [attempt] = state.attempt_log
    assert attempt.success is True
    assert attempt.accept_reason == reason


@pytest.mark.anyio
async def test_findings_are_injected_into_the_critic_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shadow mode: the critic sees the deterministic findings block."""
    captured: dict = {}
    _stub(monkeypatch, _critique(success=True, score=95), captured)
    state = _unit_state()
    state.deterministic_findings = [
        OntologyUnitFinding(
            kind=OntologyUnitFindingKind.FOREIGN_DELETE,
            message="this update deletes catalog content",
        ),
        OntologyUnitFinding(
            kind=OntologyUnitFindingKind.LABEL_COLLISION,
            mandatory=False,
            message="label duplicates onto:Sample",
        ),
    ]

    await criticise_ontology_module.criticise_ontology(state, _tools())

    chapter = captured["ontology_chapter"]
    assert "## MANDATORY fixes" in chapter
    assert "this update deletes catalog content" in chapter
    assert "## Advisory findings" in chapter
    assert "label duplicates onto:Sample" in chapter


@pytest.mark.anyio
async def test_loop_collects_findings_even_when_no_critic_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At MAX_VISITS=1 the critic is skipped; the residual must exist anyway."""
    foreign = URIRef("https://elsewhere.example/vocab#Widget")

    async def fake_render(state: UnitOntologyState, tools, **kwargs):
        insert = RDFGraph()
        insert.add((foreign, RDF.type, OWL.Class))
        state.ontology_updates = [
            GraphUpdate(triple_operations=[TripleOp(type="insert", graph=insert)])
        ]
        state.status = Status.SUCCESS
        return state

    async def fake_resolve(_state, _tools, _unit):
        return UnitOntologyContext(
            snapshot=OntologySnapshot(
                graph=_snapshot_graph(), source_iris=["https://example.com/onto"]
            ),
            writable_iris=["https://example.com/onto"],
            confidence=1.0,
        )

    monkeypatch.setattr(unit_loops, "render_ontology", fake_render)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", fake_resolve)

    state = _unit_state()
    toolbox = cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox, SimpleNamespace(validation_policy=None)
            ),
            ontology_manager=OntologyManager(),
        ),
    )
    document_state = AgentState(
        ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY
    )

    result = await unit_loops.ontology_loop(
        state, toolbox, UnitLoopContext.from_agent_state(document_state)
    )

    assert result.status == Status.SUCCESS
    kinds = {finding.kind for finding in result.deterministic_findings}
    assert OntologyUnitFindingKind.FOREIGN_NAMESPACE in kinds
    assert OntologyUnitFindingKind.MISSING_LABEL in kinds
    assert not any(a.kind == "critic" for a in result.attempt_log)


@pytest.mark.anyio
async def test_critic_criteria_carry_the_partial_context_notice_in_vector_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A critic judging a retrieved subset must not demand unretrieved concepts."""
    captured: dict = {}
    _stub(monkeypatch, _critique(success=True, score=95), captured)
    state = _unit_state()
    state.ontology_snapshot.assembly_mode = (
        OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE
    )

    await criticise_ontology_module.criticise_ontology(state, _tools())

    assert "PARTIAL CONTEXT" in captured["ontology_criteria"]


@pytest.mark.anyio
async def test_critic_criteria_stay_unqualified_for_full_copy_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    _stub(monkeypatch, _critique(success=True, score=95), captured)

    await criticise_ontology_module.criticise_ontology(_unit_state(), _tools())

    assert "PARTIAL CONTEXT" not in captured["ontology_criteria"]
